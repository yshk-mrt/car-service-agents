# os must be imported before google.generativeai
import os
import google.generativeai as genai
import uuid
import json
import time

# --- Configuration ---
try:
    # SECURE METHOD: Load API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    # Create the model
    # See https://ai.google.dev/gemini-api/docs/models/gemini
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
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

except ImportError:
    print("Error: google.generativeai library not found.")
    print("Please install it: pip install google-generativeai")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    print("Please set the GEMINI_API_KEY environment variable.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during Gemini setup: {e}")
    exit(1)


# --- Data Structures ---

agents = {}  # Dictionary to store agent_id -> agent_data
communication_history = []  # List to store message dictionaries
logged_in_agent_id = None  # Track the currently logged-in physical agent

# --- Agent Definitions ---

AGENT_TYPES = ["Agent", "Repository Agent", "Physical Agent"]

# Example initial agents
INITIAL_AGENTS = [
    {
        "id": "repo-001",
        "type": "Repository Agent",
        "description": "Suggests agents based on query. Ask for repository suggestions and this agent will suggest a recipient id.",
        "system_prompt": None,  # Repository agent doesn't use Gemini directly
    },
    {
        "id": "gen-agent-001",
        "type": "Agent",
        "description": "A general purpose AI agent.",
        "system_prompt": "You are a helpful AI assistant. Use the conversation history to understand the context. When unsure who to talk to, consider asking the Repository Agent (repo-001).",
    },
    {
        "id": "human-term-001",
        "type": "Physical Agent",
        "description": "Represents the human user interacting via this terminal.",
        "system_prompt": None,  # Physical agent is a proxy
    },
]

# --- Core Functions ---


def add_agent(agent_type, description, system_prompt=None):
    """Creates a new agent and adds it to the registry."""
    if agent_type not in AGENT_TYPES:
        print(f"Error: Invalid agent type '{agent_type}'. Must be one of {AGENT_TYPES}")
        return None
    agent_id = f"{agent_type.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    agents[agent_id] = {
        "id": agent_id,
        "type": agent_type,
        "description": description,
        "system_prompt": system_prompt,
    }
    print(f"Agent created: {agent_id} ({agent_type})")
    return agent_id


def find_agent_by_id(agent_id):
    """Finds an agent by its ID."""
    return agents.get(agent_id)


def format_history_for_prompt(limit=20):
    """Formats the recent communication history for the Gemini prompt."""
    formatted = "\n--- Communication History (most recent first) ---\n"
    # Take the last 'limit' messages
    recent_history = communication_history[-(limit):]
    for msg in reversed(recent_history):
        formatted += f"From: {msg['sender']} | To: {msg['recipient']} | Time: {msg.get('timestamp', 'N/A')}\nMessage: {msg['message']}\n---\n"
    return formatted


def call_gemini_api(agent, current_message):
    """Calls the Gemini API for an 'Agent' type."""
    if not agent or agent["type"] != "Agent":
        print("Error: Cannot call Gemini for non-'Agent' type or null agent.")
        return "Error: Internal configuration issue."

    system_instruction = agent.get("system_prompt", "You are a helpful assistant.")
    history_context = format_history_for_prompt()
    prompt = f"{system_instruction}\n\n{history_context}\n--- Current Task/Message ---\nFrom: {current_message['sender']}\nTo: {current_message['recipient']}\nMessage: {current_message['message']}\n\n--- Your Response ---\n"

    print(
        f"\nDEBUG: Calling Gemini for {agent['id']} with prompt excerpt:\n'{current_message['message']}'..."
    )
    try:
        # Start a chat session using the model history
        # Note: Gemini API typically manages history implicitly in a chat session.
        # For this structure, we explicitly pass history in the prompt.
        # A more robust implementation might use model.start_chat() if interactions
        # are expected to be long and conversational *for a single agent*.
        # Since agents talk to *each other*, passing explicit history is clearer here.

        response = model.generate_content(prompt)
        print(f"DEBUG: Gemini response received for {agent['id']}.")
        return response.text.strip()

    except Exception as e:
        print(f"Error calling Gemini API for {agent['id']}: {e}")
        return f"Error: Could not generate response due to API error: {e}"


def process_repository_request(query):
    """Handles requests for the Repository Agent."""
    print(f"DEBUG: Repository Agent processing query: '{query}'")
    suggestions = []
    query_lower = query.lower()
    for agent_id, agent_data in agents.items():
        # Simple keyword matching in description
        # A more advanced version could use embeddings or better NLP
        if query_lower in agent_data["description"].lower():
            suggestions.append(f"- {agent_id}: {agent_data['description']}")

    if not suggestions:
        # Fallback: list all non-repository agents if specific match fails
        suggestions = [
            f"- {aid}: {adata['description']}"
            for aid, adata in agents.items()
            if adata["type"] != "Repository Agent"
        ]
        if not suggestions:
            return "No other agents found in the repository."

    return (
        f"Found potential agents based on your query '{query}':\n"
        + "\n".join(suggestions)
        + "\n\nConsider sending your message directly to one of these agent IDs."
    )


def process_message(message):
    """Processes a message intended for a specific agent."""
    recipient_id = message["recipient"]
    recipient_agent = find_agent_by_id(recipient_id)

    if not recipient_agent:
        print(f"Warning: Recipient agent '{recipient_id}' not found. Message dropped.")
        return

    print(f"\n--- Message Received by {recipient_id} ({recipient_agent['type']}) ---")
    print(f"From: {message['sender']}")
    print(f"Message: {message['message']}")
    print("-" * (len(recipient_id) + len(recipient_agent["type"]) + 24))

    # --- Agent Logic ---
    response_message = None
    response_recipient = None  # Who should the agent respond to? Typically the sender.

    if recipient_agent["type"] == "Agent":
        # Call Gemini API
        response_content = call_gemini_api(recipient_agent, message)
        # Simple logic: respond back to the original sender
        response_message = response_content
        response_recipient = message["sender"]

    elif recipient_agent["type"] == "Repository Agent":
        # Handle repository logic
        response_content = process_repository_request(message["message"])
        response_message = response_content
        response_recipient = message["sender"]  # Respond back to asker

    elif recipient_agent["type"] == "Physical Agent":
        # Display the message if the user is logged into this agent
        if recipient_id == logged_in_agent_id:
            print(f"\n>>> Message for YOU ({recipient_id}):")
            print(f"    From: {message['sender']}")
            print(f"    Content: {message['message']}")
            print(">>>")
        else:
            # Message for another physical agent, just log it was received
            print(
                f"(Message intended for physical agent {recipient_id}, currently logged into {logged_in_agent_id})"
            )
        # Physical agents don't auto-respond, they wait for user input

    # --- Send Response (if generated) ---
    if response_message and response_recipient:
        # Small delay to simulate processing
        time.sleep(0.5)
        send_message(recipient_id, response_recipient, response_message)


def send_message(sender_id, recipient_id, message_content):
    """Adds a message to the history and triggers processing."""
    if not find_agent_by_id(sender_id):
        print(f"Error: Sender agent '{sender_id}' not found.")
        return
    if not find_agent_by_id(recipient_id):
        # Allow sending to potentially non-existent agents (e.g., user typo)
        # process_message will handle the warning.
        print(
            f"Warning: Recipient agent '{recipient_id}' not found. Message will be sent but may not be processed."
        )

    message = {
        "sender": sender_id,
        "recipient": recipient_id,
        "message": message_content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    communication_history.append(message)
    print(f"\n--- Message Sent ---")
    print(f"From: {sender_id}")
    print(f"To:   {recipient_id}")
    print(f"Msg:  {message_content}")
    print("--------------------")

    # Trigger processing by the recipient
    process_message(message)


# --- UI Functions ---


def display_agents():
    """Lists all current agents."""
    print("\n--- Registered Agents ---")
    if not agents:
        print("No agents registered.")
        return
    for agent_id, agent_data in agents.items():
        print(f"ID: {agent_id}")
        print(f"  Type: {agent_data['type']}")
        print(f"  Description: {agent_data['description']}")
        if agent_data.get("system_prompt"):
            print(
                f"  System Prompt: {agent_data['system_prompt'][:50]}..."
            )  # Show snippet
        print("-" * 10)


def handle_create_agent():
    """Handles the user flow for creating a new agent."""
    print("\n--- Create New Agent ---")
    print("Available types:", ", ".join(AGENT_TYPES))
    try:
        agent_type = input("Enter agent type: ")
        if agent_type not in AGENT_TYPES:
            print("Invalid type.")
            return
        description = input("Enter agent description: ")
        system_prompt = None
        if agent_type == "Agent":
            system_prompt = input("Enter system prompt for this 'Agent': ")
        add_agent(agent_type, description, system_prompt)
    except EOFError:
        print("\nCreation cancelled.")


def handle_login():
    """Handles logging into a Physical Agent."""
    global logged_in_agent_id
    print("\n--- Log In to Physical Agent ---")
    physical_agents = {
        aid: data for aid, data in agents.items() if data["type"] == "Physical Agent"
    }
    if not physical_agents:
        print("No Physical Agents available to log into.")
        return

    print("Available Physical Agents:")
    for aid, data in physical_agents.items():
        print(f"- {aid}: {data['description']}")

    try:
        agent_id_to_login = input("Enter ID of Physical Agent to log into: ")
        if agent_id_to_login in physical_agents:
            logged_in_agent_id = agent_id_to_login
            print(f"\nSuccessfully logged in as {logged_in_agent_id}.")
            logged_in_mode()  # Enter the interactive loop
        else:
            print("Invalid or non-physical agent ID.")
    except EOFError:
        print("\nLogin cancelled.")


def logged_in_mode():
    """Interactive loop when logged into a Physical Agent."""
    global logged_in_agent_id
    print(f"\n--- Logged In: {logged_in_agent_id} ---")
    print("Enter messages in the format 'recipient_id: message content'")
    print("Type 'history' to view full communication history.")
    print("Press Ctrl+D or type 'logout' to log out.")

    last_history_displayed_count = 0

    while True:
        try:
            # Display new messages (simple approach: check length)
            # A better way might involve timestamps or flags, but this is minimal
            if len(communication_history) > last_history_displayed_count:
                print("\n--- Recent System Activity ---")
                for msg in communication_history[last_history_displayed_count:]:
                    # Highlight messages involving the current agent
                    is_involved = (
                        msg["sender"] == logged_in_agent_id
                        or msg["recipient"] == logged_in_agent_id
                    )
                    prefix = ">>> " if is_involved else "    "
                    print(
                        f"{prefix}[{msg['timestamp']}] From: {msg['sender']} To: {msg['recipient']}"
                    )
                    if (
                        is_involved
                    ):  # Show content only if involved or if user explicitly asks
                        print(f"{prefix}    Msg: {msg['message']}")
                print("--- End Activity ---")
                last_history_displayed_count = len(communication_history)

            user_input = input(f"\n[{logged_in_agent_id}] Send message or command: ")
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() == "logout":
                break

            if user_input.lower() == "history":
                print("\n--- Full Communication History ---")
                if not communication_history:
                    print("(empty)")
                else:
                    for msg in communication_history:
                        print(
                            f"[{msg.get('timestamp', 'N/A')}] Sender: {msg['sender']}, Recipient: {msg['recipient']}"
                        )
                        print(f"  Message: {msg['message']}")
                        print("-" * 15)
                print("--- End History ---")
                continue

            if ":" not in user_input:
                print("Invalid format. Use 'recipient_id: message content'")
                continue

            recipient_id, message_content = user_input.split(":", 1)
            recipient_id = recipient_id.strip()
            message_content = message_content.strip()

            if not recipient_id or not message_content:
                print(
                    "Invalid format. Both recipient ID and message content are required."
                )
                continue

            # Send the message from the logged-in physical agent
            send_message(logged_in_agent_id, recipient_id, message_content)
            # Note: The response processing happens within send_message -> process_message

        except EOFError:  # Ctrl+D
            print("\nLogging out...")
            break
        except KeyboardInterrupt:  # Ctrl+C
            print("\nLogging out...")
            break

    # Log out
    print(f"Logged out from {logged_in_agent_id}.")
    logged_in_agent_id = None


# --- Main Execution ---


def main():
    """Main menu loop."""
    print("--- Minimal Multi-Agent System ---")

    # Load initial agents
    for agent_data in INITIAL_AGENTS:
        agents[agent_data["id"]] = agent_data
    print(f"Loaded {len(INITIAL_AGENTS)} initial agents.")

    while True:
        print("\n--- Main Menu ---")
        print("1. List Agents")
        print("2. Create Agent")
        print("3. Log into Physical Agent")
        print("4. View History")
        print("5. Exit")

        try:
            choice = input("Enter your choice: ")

            if choice == "1":
                display_agents()
            elif choice == "2":
                handle_create_agent()
            elif choice == "3":
                handle_login()
            elif choice == "4":
                print("\n--- Full Communication History ---")
                if not communication_history:
                    print("(empty)")
                else:
                    for msg in communication_history:
                        print(
                            f"[{msg.get('timestamp', 'N/A')}] Sender: {msg['sender']}, Recipient: {msg['recipient']}"
                        )
                        print(f"  Message: {msg['message']}")
                        print("-" * 15)
                print("--- End History ---")
            elif choice == "5":
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please try again.")

        except EOFError:  # Ctrl+D
            print("\nExiting.")
            break
        except KeyboardInterrupt:  # Ctrl+C
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
