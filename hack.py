# os must be imported before google.generativeai
import os
import google.generativeai as genai
import streamlit as st
import uuid
import json
import time
from datetime import datetime

# --- Configuration and Initialization ---
try:
    # Load API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY environment variable not set. Please set it and restart."
        )
        st.stop()  # Halt execution if no API key

    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro-preview-03-25",  # Or another suitable model
        generation_config=generation_config,
    )
except ImportError:
    st.error(
        "Required library not found. Please install google-generativeai: pip install google-generativeai"
    )
    st.stop()
except ValueError as e:  # Specific check if key is None after getenv
    st.error(f"Configuration Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during Gemini setup: {e}")
    st.stop()


AGENT_TYPES = ["Agent", "Repository Agent", "Physical Agent"]

# Example initial agents (adjust IDs if needed)
INITIAL_AGENTS_DATA = {
    "repo-001": {
        "id": "repo-001",
        "type": "Repository Agent",
        "description": "Suggests agents based on query.",
        "system_prompt": None,
    },
    "gen-agent-001": {
        "id": "gen-agent-001",
        "type": "Agent",
        "description": "A general purpose AI agent.",
        "system_prompt": "You are a helpful AI assistant...",  # Keep it concise for example
    },
    "human-user-1": {
        "id": "human-user-1",
        "type": "Physical Agent",
        "description": "Human user interacting via Streamlit.",
        "system_prompt": None,
    },
    "human-user-2": {
        "id": "human-user-2",
        "type": "Physical Agent",
        "description": "Another human user via Streamlit.",
        "system_prompt": None,
    },
}


# --- State Management ---
def initialize_state():
    """Initializes session state variables if they don't exist."""
    if "agents" not in st.session_state:
        st.session_state.agents = INITIAL_AGENTS_DATA.copy()
        print("Initialized agents state")  # Debug print (appears in terminal)
    if "communication_history" not in st.session_state:
        st.session_state.communication_history = []
        print("Initialized history state")
    if "logged_in_agent_id" not in st.session_state:
        st.session_state.logged_in_agent_id = None
        print("Initialized logged_in_agent_id state")
    if "selected_chat_agent_id" not in st.session_state:
        st.session_state.selected_chat_agent_id = None
        print("Initialized selected_chat_agent_id state")


# --- Core Agent Logic (Adapted for Streamlit State) ---


def add_agent(agent_type, description, system_prompt=None):
    """Creates a new agent and adds it to the state."""
    if agent_type not in AGENT_TYPES:
        st.error(f"Invalid agent type '{agent_type}'.")
        return None
    agent_id = f"{agent_type.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    st.session_state.agents[agent_id] = {
        "id": agent_id,
        "type": agent_type,
        "description": description,
        "system_prompt": system_prompt,
    }
    st.success(f"Agent created: {agent_id} ({agent_type})")
    # No need to return ID if just updating state
    # st.rerun() # Consider if UI should immediately refresh after creation


def find_agent_by_id(agent_id):
    """Finds an agent by its ID from session state."""
    return st.session_state.agents.get(agent_id)


def format_history_for_prompt(limit=20):
    """Formats the recent communication history for the Gemini prompt."""
    formatted = "\n--- Communication History (most recent first) ---\n"
    # Access history from session state
    recent_history = st.session_state.communication_history[-(limit):]
    for msg in reversed(recent_history):
        formatted += f"From: {msg['sender']} | To: {msg['recipient']} | Time: {msg.get('timestamp', 'N/A')}\nMessage: {msg['message']}\n---\n"
    return formatted


# @st.cache_data # Caching might be complex with API calls and changing history
def call_gemini_api(agent, current_message):
    """Calls the Gemini API for an 'Agent' type."""
    # (Ensure this function uses the configured 'model' object)
    if not agent or agent["type"] != "Agent":
        print(
            f"Error: Cannot call Gemini for non-'Agent' type: {agent.get('id', 'N/A')}"
        )
        return "Error: Internal configuration issue."

    system_instruction = agent.get("system_prompt", "You are a helpful assistant.")
    history_context = format_history_for_prompt()  # Uses session state history
    prompt = f"{system_instruction}\n\n{history_context}\n--- Current Task/Message ---\nFrom: {current_message['sender']}\nTo: {current_message['recipient']}\nMessage: {current_message['message']}\n\n--- Your Response ---\n"

    print(f"DEBUG: Calling Gemini for {agent['id']}...")
    try:
        response = model.generate_content(prompt)
        print(f"DEBUG: Gemini response received for {agent['id']}.")
        # Make sure to handle potential blocks or safety issues if response.text is not straightforward
        if response.parts:
            return response.text.strip()
        else:
            # Handle cases where the response might be blocked due to safety settings or other issues
            print(f"WARN: Gemini response for {agent['id']} might be empty or blocked.")
            # Find the reason if possible (this might vary depending on API version/library details)
            try:
                reason = response.prompt_feedback.block_reason
                reason_msg = f"Response blocked due to: {reason}"
            except Exception:
                reason_msg = "Response may be empty or blocked."
            return (
                f"(Agent {agent['id']} did not provide a text response. {reason_msg})"
            )

    except Exception as e:
        print(f"Error calling Gemini API for {agent['id']}: {e}")
        st.error(f"Gemini API Error for {agent['id']}: {e}")  # Show error in UI
        return f"Error: Could not generate response due to API error."


def process_repository_request(query):
    """Handles requests for the Repository Agent using session state."""
    print(f"DEBUG: Repository Agent processing query: '{query}'")
    suggestions = []
    query_lower = query.lower()
    # Access agents from session state
    for agent_id, agent_data in st.session_state.agents.items():
        # Simple keyword matching
        search_text = f"{agent_data['description']} {agent_data['type']} {agent_id}"
        if query_lower in search_text.lower():
            # Exclude self (the repo agent) from suggestions
            if agent_data["type"] != "Repository Agent":
                suggestions.append(
                    f"- {agent_id}: ({agent_data['type']}) {agent_data['description']}"
                )

    if not suggestions:
        # Fallback: list all non-repository, non-physical agents
        suggestions = [
            f"- {aid}: ({adata['type']}) {adata['description']}"
            for aid, adata in st.session_state.agents.items()
            if adata["type"] not in ["Repository Agent", "Physical Agent"]
        ]
        if not suggestions:
            return "No suitable agents found in the repository based on your query."

    return f"Based on your query '{query}', consider contacting:\n" + "\n".join(
        suggestions
    )


def process_message(message):
    """Processes a message intended for a specific agent using session state."""
    recipient_id = message["recipient"]
    recipient_agent = find_agent_by_id(recipient_id)  # Uses session state lookup

    if not recipient_agent:
        print(f"Warning: Recipient agent '{recipient_id}' not found. Message dropped.")
        return  # Don't proceed

    print(f"DEBUG: Processing message for {recipient_id} ({recipient_agent['type']})")
    # print(f"DEBUG: Message content: {message['message'][:100]}...") # Avoid printing too much

    # --- Agent Logic ---
    response_message = None
    response_recipient = message["sender"]  # Default: respond back to sender

    if recipient_agent["type"] == "Agent":
        response_content = call_gemini_api(recipient_agent, message)
        response_message = response_content

    elif recipient_agent["type"] == "Repository Agent":
        response_content = process_repository_request(message["message"])
        response_message = response_content

    elif recipient_agent["type"] == "Physical Agent":
        # Messages arrive for physical agents (like the user's)
        # They are simply stored in history. The UI will display them.
        # Physical agents do not generate automatic responses here.
        print(
            f"DEBUG: Message delivered to Physical Agent {recipient_id}. User needs to respond via UI."
        )
        pass  # No automatic response needed

    # --- Send Response (if generated) ---
    if response_message and response_recipient:
        # Small delay - less critical in web UI unless API calls are very fast
        # time.sleep(0.1)
        send_message(
            recipient_id, response_recipient, response_message
        )  # Send the generated response


def send_message(sender_id, recipient_id, message_content):
    """Adds a message to the history (in state) and triggers processing for the recipient."""
    if not find_agent_by_id(sender_id):
        print(f"Error: Sender agent '{sender_id}' not found.")
        st.warning(f"Message not sent: Sender agent '{sender_id}' not found.")
        return
    # Note: We don't strictly need to check recipient existence here,
    # process_message handles non-existent recipients gracefully.

    print(f"DEBUG: Sending message: {sender_id} -> {recipient_id}")
    message = {
        "sender": sender_id,
        "recipient": recipient_id,
        "message": message_content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Append to history in session state
    st.session_state.communication_history.append(message)

    # Trigger processing by the recipient (this might chain reactions)
    # This happens sequentially within the same Streamlit script run
    process_message(message)
    # Note: No st.rerun() here; it should happen *after* the initial user action is fully processed.


# --- Streamlit UI ---

# Initialize state at the beginning of the script run
initialize_state()

st.set_page_config(layout="wide")
st.title("💬 Multi-Agent Chat")

# --- Sidebar ---
with st.sidebar:
    st.header("👤 Login & Agents")

    # 1. Select Logged-In Agent
    physical_agent_options = {
        aid: f"{data['description']} ({aid})"
        for aid, data in st.session_state.agents.items()
        if data["type"] == "Physical Agent"
    }

    if not physical_agent_options:
        st.warning("No Physical Agents found. Please create one.")
        # Set logged_in_agent_id to None if the current one is somehow invalid
        if st.session_state.logged_in_agent_id not in physical_agent_options:
            st.session_state.logged_in_agent_id = None
            st.session_state.selected_chat_agent_id = None  # Reset chat partner too
    else:
        # If nothing selected OR previously selected agent was removed, select the first available
        if (
            st.session_state.logged_in_agent_id is None
            or st.session_state.logged_in_agent_id not in physical_agent_options
        ):
            st.session_state.logged_in_agent_id = list(physical_agent_options.keys())[0]
            st.session_state.selected_chat_agent_id = (
                None  # Reset chat partner on login change
            )

        selected_login = st.selectbox(
            "Log in as:",
            options=list(physical_agent_options.keys()),
            format_func=lambda aid: physical_agent_options[aid],
            key="login_selectbox",  # Add key for stability
            index=list(physical_agent_options.keys()).index(
                st.session_state.logged_in_agent_id
            ),  # Set default index
        )

        # Update state if selection changed, reset chat partner
        if selected_login != st.session_state.logged_in_agent_id:
            st.session_state.logged_in_agent_id = selected_login
            st.session_state.selected_chat_agent_id = (
                None  # Reset chat partner on login change
            )
            st.rerun()  # Rerun immediately to reflect login change and update chat partner list

    st.markdown("---")

    # 2. Select Agent to Chat With
    if st.session_state.logged_in_agent_id:
        st.subheader(f"Chat Partners for {st.session_state.logged_in_agent_id}")

        # List all *other* agents
        chat_partner_options = {
            aid: f"{data['description']} ({aid})"
            for aid, data in st.session_state.agents.items()
            if aid != st.session_state.logged_in_agent_id
        }

        if not chat_partner_options:
            st.write("No other agents available to chat with.")
            st.session_state.selected_chat_agent_id = None
        else:
            # Determine default selection for radio. If previous selection is still valid, keep it.
            current_selection = st.session_state.selected_chat_agent_id
            valid_options = list(chat_partner_options.keys())
            default_index = 0  # Default to first option
            if current_selection and current_selection in valid_options:
                try:
                    default_index = valid_options.index(current_selection)
                except ValueError:
                    # Selection is somehow invalid, reset it
                    st.session_state.selected_chat_agent_id = None

            selected_partner = st.radio(
                "Select agent to chat with:",
                options=valid_options,
                format_func=lambda aid: chat_partner_options[aid],
                key="chat_partner_radio",
                index=default_index,  # Set default selection
            )

            # Update state if selection changed
            if selected_partner != st.session_state.selected_chat_agent_id:
                st.session_state.selected_chat_agent_id = selected_partner
                st.rerun()  # Rerun to load the chat history for the new partner

    st.markdown("---")

    # 3. Agent Management Expander
    with st.expander("🔧 Manage Agents"):
        st.subheader("Create New Agent")
        with st.form("create_agent_form", clear_on_submit=True):
            new_agent_type = st.selectbox("Type", AGENT_TYPES, index=0)
            new_agent_desc = st.text_input("Description")
            new_agent_prompt = None
            if new_agent_type == "Agent":  # Only show prompt field for 'Agent' type
                new_agent_prompt = st.text_area("System Prompt (optional)")

            submitted = st.form_submit_button("Create Agent")
            if submitted:
                if not new_agent_desc:
                    st.error("Description cannot be empty.")
                else:
                    add_agent(new_agent_type, new_agent_desc, new_agent_prompt)
                    # Adding agent updates state, rerun will refresh lists if needed later
                    # Consider adding st.rerun() here if list updates immediately are crucial

        st.subheader("All Agents")
        # Display all agents for reference
        if not st.session_state.agents:
            st.write("No agents defined.")
        else:
            for agent_id, agent_data in st.session_state.agents.items():
                st.text(f"- ID: {agent_id}")
                st.text(f"  Type: {agent_data['type']}")
                st.text(f"  Desc: {agent_data['description']}")
                if agent_data.get("system_prompt"):
                    st.text(f"  Prompt: {agent_data['system_prompt'][:30]}...")


# --- Main Chat Area ---
if not st.session_state.logged_in_agent_id:
    st.info("Please log in using the sidebar.")
elif not st.session_state.selected_chat_agent_id:
    st.info("Please select an agent to chat with from the sidebar.")
else:
    # Display chat history with the selected agent
    current_user = st.session_state.logged_in_agent_id
    chat_partner = st.session_state.selected_chat_agent_id
    st.header(f"Chat between {current_user} and {chat_partner}")

    # Filter history for messages between the two selected agents
    filtered_history = [
        msg
        for msg in st.session_state.communication_history
        if (msg["sender"] == current_user and msg["recipient"] == chat_partner)
        or (msg["sender"] == chat_partner and msg["recipient"] == current_user)
    ]

    # Display messages
    chat_container = st.container()  # Use container for better layout control maybe
    with chat_container:
        if not filtered_history:
            st.write("No messages yet in this conversation.")
        else:
            for msg in filtered_history:
                role = "user" if msg["sender"] == current_user else "assistant"
                # Use agent ID as avatar, or initials maybe
                avatar_icon = (
                    "👤"
                    if find_agent_by_id(msg["sender"])["type"] == "Physical Agent"
                    else "🤖"
                )
                with st.chat_message(name=msg["sender"], avatar=avatar_icon):
                    st.markdown(
                        f"**To: {msg['recipient']}** ({msg.get('timestamp', '')})"
                    )
                    st.markdown(
                        msg["message"]
                    )  # Render message content (supports markdown)

    # Chat Input - positioned at the bottom
    prompt = st.chat_input(
        f"Send message as {current_user} to {chat_partner}...", key="chat_input_box"
    )

    if prompt:
        # 1. Add user message immediately to history for visual feedback
        #    (We will process it right after)
        # user_msg_obj = {
        #     "sender": current_user, "recipient": chat_partner,
        #     "message": prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # }
        # st.session_state.communication_history.append(user_msg_obj)
        # Not strictly necessary as send_message below does this, but can improve perceived responsiveness

        # 2. Send the message and trigger processing
        #    This will add message to history and potentially trigger responses,
        #    which will also be added to history sequentially.
        send_message(current_user, chat_partner, prompt)

        # 3. Rerun the script to display the updated history
        #    This will show the message just sent and any responses generated.
        st.rerun()
