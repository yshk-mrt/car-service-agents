import os
import json
import random
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

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

# Pydantic model for CrossCrewCommunicationTool input validation
class CrossCrewInput(BaseModel):
    """Input schema for CrossCrewCommunicationTool."""
    target_crew_id: str = Field(..., description="The registered ID of the crew to talk to")
    message: str = Field(..., description="The content to send to the target crew")

# Tool for Crew 1 (Customer) to talk to Crew 2 (Shop)
class CrossCrewCommunicationTool(BaseTool):
    name: str = "Cross-Crew Communication Tool"
    description: str = (
        "Sends a message (e.g., a service request) to another specified Crew "
        "and returns its response. Input must be a dictionary with 'target_crew_id': string "
        "(the registered ID of the crew to talk to) and 'message': string (the content to send)."
    )
    args_schema: type[BaseModel] = CrossCrewInput

    def _run(self, target_crew_id: str, message: str) -> str:
        if not target_crew_id or not message:
            return "Error: 'target_crew_id' and 'message' are required inputs."

        # Check if the agent is assigned to this tool
        agent_name = getattr(self, 'agent', None)
        sender = "Unknown" if agent_name is None else self.agent.role
        
        print(f"\n MOCK TOOL LOG: Attempting communication from '{sender}' to Crew ID '{target_crew_id}'...")

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

# Tool for Shop Manager to communicate with Mechanic
class ManagerToMechanicTool(BaseTool):
    name: str = "Manager-to-Mechanic Communication Tool"
    description: str = "Allows the shop manager to communicate with the mechanic."

    def _run(self, message_type: str, details: str) -> str:
        print(f"\n MOCK TOOL LOG (Manager): Sending {message_type} message to mechanic: {details}")
        time.sleep(0.2)  # Simulate communication delay
        
        responses = {
            "assignment": "Maintenance assignment received. Will prioritize accordingly.",
            "inquiry": "Will check status and report back soon.",
            "priority": "Priority update acknowledged and adjusted workflow."
        }
        
        response = responses.get(message_type.lower(), "Message received and acknowledged.")
        print(f" MOCK TOOL LOG (Mechanic): {response}")
        return f"Mechanic response: {response}"

# --- Instantiate Tools ---
cross_crew_tool = CrossCrewCommunicationTool()
quote_tool = QuoteGeneratorTool()
availability_tool = AvailabilityCheckerTool()
manager_to_mechanic_tool = ManagerToMechanicTool()


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

        When using the Cross-Crew Communication Tool, you MUST provide these two parameters:
        1. 'target_crew_id': which should be exactly 'shop_crew_main'
        2. 'message': which should be the complete message from your previous task

        Example of how to use the tool correctly:
        ```
        target_crew_id: "shop_crew_main"
        message: "Hello, I have a 2020 Subaru Outback and I'm experiencing issues..."
        ```

        Do not modify or summarize the message - send the exact message you created in the previous task.
        
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
    tools=[quote_tool, availability_tool, manager_to_mechanic_tool, cross_crew_tool], # Tools the shop manager uses
    verbose=True,
    allow_delegation=False
)

# 2. Define Shop Task(s)
#    - Task: Process the incoming message, use tools, formulate response
task_process_request = Task(
    description="""
        Process a customer repair request. If a message was received through cross-crew communication,
        it will be available in the context.
        
        1. Analyze the customer's request to understand the car model and the problem.
           If no specific message was received, assume this is a general inquiry.
        
        2. Use the 'Quote Generator Tool' with the problem description to get an estimated cost and time.
        
        3. Use the 'Availability Checker Tool' to find the next available slot.
        
        4. Synthesize this information into a clear response message for the customer. Include:
           - Acknowledgment of their specific vehicle and issue
           - The quote estimate (cost and time)
           - Scheduling information/availability
           - Next steps for the customer
        
        If any tool returns an error, handle it gracefully in your response.
        
        Your output should be a professional, customer-focused response.
    """,
    expected_output="A response message addressed to the customer containing the quote estimate, time estimate, and availability information.",
    agent=shop_manager_agent
)

# Add a new task for the shop manager to communicate with the mechanic
task_assign_to_mechanic = Task(
    description="""
        After providing a quote to the customer, you need to assign the maintenance work to a mechanic.
        Review the customer's request again and prepare instructions for the mechanic.
        
        The customer's request details are included in the context from the previous task.
        
        Use the Manager-to-Mechanic Communication Tool to send the assignment:
        - message_type: "assignment"
        - details: Include the car model, specific issues to address, priority level, and any special instructions.
        
        After receiving the mechanic's confirmation, check back on their progress using:
        - message_type: "inquiry"
        - details: Ask for a status update on the assignment.
        
        Your output should include the maintenance assignment you created and the mechanic's responses.
    """,
    expected_output="A summary of the assignment to the mechanic and their responses.",
    agent=shop_manager_agent,
    context=[task_process_request]  # This task depends on the previous customer request
)

# Add a new task for direct cross-crew communication between the shop manager and mechanic crew
task_direct_crew_communication = Task(
    description="""
        You need to follow up on the mechanic's work by directly communicating with the mechanic crew.
        
        Use the Cross-Crew Communication Tool to send a message to the mechanic crew:
        - target_crew_id: "mechanic_crew"
        - message: Craft a detailed message asking about the work progress, any additional repairs found, 
                  and expected completion time. Include relevant details from previous customer interactions.
        
        The mechanic crew will respond with their findings and status update.
        
        After receiving the mechanic's technical response:
        1. Process the information to separate customer-relevant details from technical details
        2. Translate technical information into customer-friendly language
        3. Format the final response to the customer with:
           - Service completion status
           - Any additional repairs found (if applicable)
           - Expected completion time
           - Next steps for the customer
        
        Your output should be a professional customer-ready message that includes only appropriate information.
    """,
    expected_output="A customer-friendly message based on the mechanic's technical response.",
    agent=shop_manager_agent,
    context=[task_assign_to_mechanic]  # This task follows after the mechanic assignment
)

# 3. Create Shop Crew
shop_crew = Crew(
    agents=[shop_manager_agent],
    tasks=[task_process_request, task_assign_to_mechanic, task_direct_crew_communication],
    process=Process.sequential,
    verbose=0
)

# =============================
# --- Mechanic Crew Definition ---
# =============================

print("\n--- Defining Mechanic Crew ---")

# Define a maintenance checker tool that sometimes reports additional repairs are needed
class MaintenanceCheckerTool(BaseTool):
    name: str = "Maintenance Checker Tool"
    description: str = "Performs maintenance checks and sometimes discovers additional repair needs."

    def _run(self, maintenance_task: str) -> str:
        print(f"\n MOCK TOOL LOG (Mechanic): Performing maintenance check for '{maintenance_task}'...")
        time.sleep(0.3)  # Simulate work
        
        # Randomly determine if additional repairs are needed (70% chance)
        additional_repair_needed = random.random() < 0.7
        
        if additional_repair_needed:
            additional_issues = [
                "discovered worn brake pads that need replacement",
                "found significant oil leak from the gasket",
                "noticed the timing belt is showing signs of wear and should be replaced soon",
                "detected an issue with the alternator that wasn't in the original assessment",
                "found rusted exhaust components that need attention"
            ]
            issue = random.choice(additional_issues)
            print(f" MOCK TOOL LOG (Mechanic): Additional repair needed: {issue}")
            return f"While performing {maintenance_task}, {issue}. Additional repair is necessary."
        else:
            print(f" MOCK TOOL LOG (Mechanic): No additional repairs needed.")
            return f"Maintenance task '{maintenance_task}' completed successfully. No additional issues found."

# Define a tool for mechanic to communicate with shop manager
class MechanicToManagerTool(BaseTool):
    name: str = "Mechanic-Manager Communication Tool"
    description: str = "Sends a message from the mechanic to the shop manager about availability or status updates."

    def _run(self, message_type: str, details: str) -> str:
        print(f"\n MOCK TOOL LOG (Mechanic): Sending {message_type} update to manager: {details}")
        time.sleep(0.2)  # Simulate communication delay
        
        # Could be enhanced to actually send to manager agent in a more complex setup
        responses = {
            "availability": "Availability update received and logged.",
            "status": "Status update received and logged.",
            "repair": "Additional repair notification received and processed."
        }
        
        response = responses.get(message_type.lower(), "Message received.")
        print(f" MOCK TOOL LOG (Manager): {response}")
        return f"Manager response: {response}"

# Instantiate mechanic tools
maintenance_tool = MaintenanceCheckerTool()
mechanic_comm_tool = MechanicToManagerTool()

# Define Mechanic Agent
mechanic_agent = Agent(
    role='Automotive Mechanic',
    goal='Perform maintenance tasks efficiently, identify additional repair needs, and communicate clearly with the shop manager.',
    backstory="Experienced mechanic with 15 years in the automotive repair industry. Known for thoroughness and attention to detail.",
    llm=openrouter_llm,
    tools=[maintenance_tool, mechanic_comm_tool],
    verbose=False,
    allow_delegation=False
)

# Define Mechanic Tasks
task_report_availability = Task(
    description="""
        Determine your current availability as a mechanic and report it to the shop manager.
        Consider factors like:
        1. Your current workload
        2. Complexity of jobs in your queue
        3. Skills required for upcoming repairs
        
        Use the Mechanic-Manager Communication Tool with:
        - message_type: "availability"
        - details: Your detailed availability information
        
        Your output should be the confirmation that the availability was reported.
    """,
    expected_output="Confirmation that availability was reported to the shop manager.",
    agent=mechanic_agent
)

task_perform_maintenance = Task(
    description="""
        You have been assigned a maintenance task for a customer's vehicle.
        
        Use the Maintenance Checker Tool to perform the maintenance and check if additional repairs are needed.
        Pass the specific maintenance task description from the maintenance_task parameter if available, 
        otherwise use "standard maintenance service".
        
        If the tool indicates additional repairs are needed, use the Mechanic-Manager Communication Tool to inform the manager about these additional repair needs:
        - message_type: "repair"
        - details: The specifics of the additional repair needs
        
        Your output should include what maintenance you performed and whether additional repairs are needed.
    """,
    expected_output="Description of maintenance performed and any additional repair needs identified.",
    agent=mechanic_agent,
    context=[task_report_availability]  # The availability task runs first
)

task_report_status = Task(
    description="""
        Report the status of the maintenance task to the shop manager.
        Based on the maintenance performed and any additional repairs identified,
        determine if the job is:
        - Completed: If no additional repairs were needed or they were minor
        - Delayed: If significant additional repairs were discovered
        
        Use the Mechanic-Manager Communication Tool with:
        - message_type: "status"
        - details: Your detailed status report, including whether the job is complete or delayed
        
        Your output should be a confirmation that the status was reported.
    """,
    expected_output="Confirmation that status was reported to the shop manager.",
    agent=mechanic_agent,
    context=[task_perform_maintenance]  # The maintenance task runs before this
)

# Define a task for the mechanic to respond to direct manager messages
task_respond_to_manager = Task(
    description="""
        You have received a direct message from the shop manager regarding a maintenance job.
        
        If a message was received through the cross-crew communication, it will be passed to you.
        If no specific message was received, assume the manager is asking for a general status update.
        
        1. Review the manager's message if available.
        2. If the manager is asking about work progress or if this is a general status update:
           - Use the Maintenance Checker Tool with an appropriate maintenance task description.
           - Include details about additional repairs needed (if any).
        3. If the manager is asking about completion time:
           - Estimate completion time based on the work done and any additional repairs found.
        
        Craft a detailed and professional response to the manager's inquiry.
        Include information about:
        - Current progress on the task
        - Any additional repairs you've identified using the Maintenance Checker Tool
        - Whether the job will be completed on time or delayed
        - Any parts or special tools needed
        
        Your output should be a complete response addressed to the manager.
    """,
    expected_output="A detailed response to the shop manager's inquiry.",
    agent=mechanic_agent
)

# Create Mechanic Crew
mechanic_crew = Crew(
    agents=[mechanic_agent],
    tasks=[task_report_availability, task_perform_maintenance, task_report_status, task_respond_to_manager],
    process=Process.sequential,
    verbose=0
)

# =============================
# --- Orchestration ---
# =============================

print("\n--- Registering Crews ---")
# Register the shop crew so the customer crew can find it via the tool
crew_registry['shop_crew_main'] = shop_crew
# Register the mechanic crew
crew_registry['mechanic_crew'] = mechanic_crew
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

response_from_shop = customer_crew.kickoff()

print("\n--- Primary Crew (Customer) Finished ---")
print("\nResponse Received by Customer:")
print(response_from_shop)

print("\n--- Running Mechanic Crew for Maintenance ---")
# In a real application, you would extract relevant data from the customer's request and shop's response
maintenance_info = {
    "maintenance_task": "Check engine light diagnostics and brake inspection", 
    "car_info": "2020 Subaru Outback",
    "priority": "Normal"
}

# Execute the mechanic crew with maintenance information as input
try:
    print("\nAssigning maintenance task to mechanic...")
    mechanic_result = mechanic_crew.kickoff(inputs=maintenance_info)
    
    print("\n--- Mechanic Crew Finished ---")
    print("\nFinal Report from Mechanic:")
    print(mechanic_result)
    
    print("\n--- Mechanic Report to Customer (via Shop Manager) ---")
    # In a real application, the shop manager would process the mechanic's report
    # and create a customer-friendly version to send back to the customer
    customer_friendly_report = f"""
    Thank you for bringing in your {maintenance_info['car_info']} for service.
    
    We've completed the diagnostics on the check engine light and inspection of your brakes.
    
    {mechanic_result}
    
    Please contact us if you have any questions.
    """
    print(customer_friendly_report)
    
except Exception as e:
    print(f"\nError running mechanic crew: {e}")