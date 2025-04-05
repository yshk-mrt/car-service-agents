import os
import json
import random
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from langchain_openai import ChatOpenAI

# --- Environment Setup for OpenRouter ---
openrouter_api_key = os.environ.get("openrouter_api_key", "")
openrouter_model_name = "openrouter/openai/gpt-4o-2024-11-20"

print(f"--- Configuring LLM for OpenRouter: {openrouter_model_name} ---")
openrouter_llm = LLM(
    model="openrouter/openai/gpt-4o-2024-11-20",  # Format: provider/model
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    max_tokens=500,
    temperature=0.7
)

# --- Simple Crew Registry ---
# Holds references to instantiated crews for same-process communication
crew_registry = {}

# --- Define Custom Tools ---

# Tool for Crew 1 (Customer) to talk to Crew 2 (Shop)
class CrossCrewCommunicationTool(BaseTool):
    name: str = "Cross-Crew Communication Tool"
    description: str = (
        "Sends a message (e.g., a service request) to another specified Crew "
        "and returns its response. Input must be a dictionary with 'target_crew_id': string "
        "(the registered ID of the crew to talk to) and 'message': string (the content to send)."
    )

    def _run(self, **kwargs) -> str:
        target_crew_id = kwargs.get("target_crew_id")
        message = kwargs.get("message")

        if not target_crew_id or not message:
            return "Error: 'target_crew_id' and 'message' are required inputs."

        print(f"\n MOCK TOOL LOG: Attempting communication from '{self.agent.role}' to Crew ID '{target_crew_id}'...")

        target_crew = crew_registry.get(target_crew_id)
        if not target_crew:
            print(f" MOCK TOOL ERROR: Target Crew '{target_crew_id}' not found in registry.")
            return f"Error: Target Crew '{target_crew_id}' not found."

        # Trigger the target crew's kickoff method with the message as input
        # The target crew's tasks should be designed to handle an input named 'received_message' (or similar)
        try:
            # Note: This is synchronous. The current crew waits for the target crew to finish.
            response = target_crew.kickoff(inputs={'received_message': message})
            print(f" MOCK TOOL LOG: Received response from '{target_crew_id}': {response}")
            return f"Response from {target_crew_id}: {response}"
        except Exception as e:
            print(f" MOCK TOOL ERROR: Error during kickoff of target crew '{target_crew_id}': {e}")
            return f"Error communicating with {target_crew_id}: {e}"

# Simple Tools for the Shop Crew
class QuoteGeneratorTool(BaseTool):
    name: str = "Quote Generator Tool"
    description: str = "Generates a price quote and estimated repair time based on a car problem description."

    def _run(self, problem_description: str) -> str:
        print(f"\n MOCK TOOL LOG (Shop): Generating quote for '{problem_description}'...")
        cost = 300 + random.randint(0, 700)
        days = 1 + random.randint(0, 4)
        if "brake" in problem_description.lower():
            cost += 150
            days += 1
        elif "engine" in problem_description.lower() or "check light" in problem_description.lower():
            cost += 500
            days += 2
        elif "exhaust" in problem_description.lower() or "rattling" in problem_description.lower():
             cost += 200
             days += 0
        time.sleep(0.2) # Simulate work
        quote = f"Quote generated: Estimated cost ${cost}, Estimated time {days} business days."
        print(f" MOCK TOOL LOG (Shop): {quote}")
        return quote

class AvailabilityCheckerTool(BaseTool):
    name: str = "Availability Checker Tool"
    description: str = "Checks the shop's next available appointment slot."

    def _run(self, **kwargs) -> str:
        print(f"\n MOCK TOOL LOG (Shop): Checking availability...")
        # Simulate some simple availability logic
        available_slots = [
            "Next Monday afternoon",
            "Next Tuesday morning",
            "Next Wednesday all day",
            "Fully booked for the next few days, try next week.",
        ]
        availability = random.choice(available_slots)
        time.sleep(0.1)
        print(f" MOCK TOOL LOG (Shop): Availability: {availability}")
        return f"Shop Availability: {availability}"

# --- Instantiate Tools ---
cross_crew_tool = CrossCrewCommunicationTool()
quote_tool = QuoteGeneratorTool()
availability_tool = AvailabilityCheckerTool()


# ==================================
# --- Customer Crew Definition ---
# ==================================

print("\n--- Defining Customer Crew ---")

# 1. Load Customer Config (same as before)
config_file = "customer_config.json"
customer_data = {}
try:
    with open(config_file, 'r') as f:
        customer_data = json.load(f)
    print(f" Loaded customer data from {config_file}")
except Exception as e:
    print(f"ERROR loading {config_file}: {e}")
    exit()

# 2. Define Customer Agent
#    - Add the CrossCrewCommunicationTool to its tools
customer_agent = Agent(
    role='Car Owner',
    goal='Clearly describe car issues and availability, then send this request to the auto shop crew and relay the response.',
    backstory="Owner of a vehicle needing repair. Responsible for initiating contact with the shop and understanding their reply.",
    llm=openrouter_llm,
    tools=[cross_crew_tool], # Agent needs the tool to communicate
    verbose=True,
    allow_delegation=False
)

# 3. Define Customer Tasks
#    - Task 1: Prepare the request message (same as before)
#    - Task 2: Use the tool to send the message from Task 1 to the 'shop_crew'

task1_prepare_request = Task(
    description=f"""
        Review your current situation based on the information provided below.
        Prepare a concise request message suitable for sending to an auto repair shop.
        The message should clearly state:
        1. Your car model: {customer_data.get('car_model', 'Not Specified')}
        2. The specific issues you are experiencing: {customer_data.get('car_issue', 'Not Specified')}
        3. Your availability for bringing the car in: {', '.join(customer_data.get('availability', ['Not specified']))}

        Output ONLY the formatted request message text, nothing else.
    """,
    expected_output="A plain text message containing the car model, issues, and availability.",
    agent=customer_agent,
    # output_file="customer_request.txt" # Optional: save the prepared message
)

task2_send_request_and_get_response = Task(
    description=f"""
        Take the repair request message generated in the previous task.
        Use the 'Cross-Crew Communication Tool' to send this message to the crew registered
        with the ID 'shop_crew_main'.

        Your final output should be ONLY the response received back from the 'shop_crew_main'.
    """,
    expected_output="The response message received from the shop crew via the communication tool.",
    agent=customer_agent,
    context=[task1_prepare_request] # Depends on the output of the first task
)

# 4. Create Customer Crew
customer_crew = Crew(
    agents=[customer_agent],
    tasks=[task1_prepare_request, task2_send_request_and_get_response],
    process=Process.sequential,
    verbose=1 # Set verbosity as needed
)

# =============================
# --- Shop Crew Definition ---
# =============================

print("\n--- Defining Shop Crew ---")

# 1. Define Shop Agent(s)
shop_manager_agent = Agent(
    role='Auto Shop Service Manager',
    goal='Receive customer repair requests, generate quotes using available tools, check availability, and provide clear responses.',
    backstory="Experienced service manager responsible for handling customer inquiries, providing estimates, and scheduling appointments.",
    llm=openrouter_llm,
    tools=[quote_tool, availability_tool], # Tools the shop manager uses
    verbose=True,
    allow_delegation=False
)

# 2. Define Shop Task(s)
#    - Task: Process the incoming message, use tools, formulate response
task_process_request = Task(
    description=(
        "You have received a repair request message from a customer. The message content is provided in the input variable '{received_message}'.\n"
        "1. Analyze the customer's request to understand the car model and the problem.\n"
        "2. Use the 'Quote Generator Tool' with the problem description to get an estimated cost and time.\n"
        "3. Use the 'Availability Checker Tool' to find the next available slot.\n"
        "4. Synthesize this information into a clear response message for the customer. Include the quote, time estimate, and scheduling information/availability."
        "If the quote tool returns an error or the availability tool indicates no slots, mention that politely in the response."
    ),
    expected_output="A response message addressed to the customer containing the quote estimate, time estimate, and availability information.",
    agent=shop_manager_agent
    # Note: This task implicitly uses the '{received_message}' passed via kickoff inputs
)

# 3. Create Shop Crew
shop_crew = Crew(
    agents=[shop_manager_agent],
    tasks=[task_process_request],
    process=Process.sequential,
    verbose=1
)

# =============================
# --- Orchestration ---
# =============================

print("\n--- Registering Crews ---")
# Register the shop crew so the customer crew can find it via the tool
crew_registry['shop_crew_main'] = shop_crew
print(f" Registered crews: {list(crew_registry.keys())}")


print("\n--- Kicking Off Customer Crew ---")
# Running the customer crew will trigger the sequence:
# 1. Customer agent prepares the message.
# 2. Customer agent uses the tool to send the message to 'shop_crew_main'.
# 3. The tool looks up 'shop_crew_main' and calls its kickoff() with the message.
# 4. The Shop Crew runs its task (processing the message, using its tools).
# 5. The Shop Crew finishes and returns its result (the response message).
# 6. The tool receives this response and returns it to the Customer Agent.
# 7. The Customer Crew finishes, its final result being the shop's response.

final_response_from_shop = customer_crew.kickoff()

print("\n--- Primary Crew (Customer) Finished ---")
print("\nFinal Response Received by Customer Crew:")
print(final_response_from_shop)