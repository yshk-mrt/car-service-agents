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

# --- ログ出力のためのユーティリティ関数 ---
def log_header(title):
    """Simple section header"""
    print(f"\n== {title} ==")

def log_step(sender, message):
    """Concise step log"""
    print(f"[{sender}] {message}")

def log_tool(tool_name, action, details=None):
    """Tool execution log - shortened"""
    if details and len(details) > 50:
        details = details[:47] + "..."
    print(f"→ {tool_name}: {action}" + (f" ({details})" if details else ""))

def log_response(title, message):
    """Display response message, handling different types"""
    # Handle CrewOutput objects and convert to string
    message_str = str(message)
    
    # Format the message
    lines = message_str.split('\n')
    formatted_message = lines[0]
    if len(lines) > 1:
        formatted_message += "..."
    print(f"{title}: {formatted_message}")

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

        # Fix: Get agent information from proper CrewAI context
        try:
            sender = self.agent.role if hasattr(self, 'agent') and self.agent else "Customer"
        except:
            sender = "Customer"  # Fallback name
        
        log_tool(self.name, f"{sender} → {target_crew_id}")

        target_crew = crew_registry.get(target_crew_id)
        if not target_crew:
            log_tool(self.name, "ERROR", f"Crew '{target_crew_id}' not found")
            return f"Error: Target Crew '{target_crew_id}' not found."

        try:
            # verbose=0でクルーを実行し、出力を抑制
            original_verbose = target_crew.verbose
            target_crew.verbose = 0
            response = target_crew.kickoff(inputs={'received_message': message})
            target_crew.verbose = original_verbose
            return response
        except Exception as e:
            log_tool(self.name, "ERROR", f"{e}")
            return f"Error communicating with {target_crew_id}: {e}"

# Simple Tools for the Shop Crew
class QuoteGeneratorTool(BaseTool):
    name: str = "Quote Generator Tool"
    description: str = "Generates a price quote and estimated repair time based on a car problem description."

    def _run(self, problem_description: str) -> str:
        log_tool(self.name, f"Generating quote")
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
        log_tool(self.name, "Quote ready", f"${cost}, {days} days")
        return quote

class AvailabilityCheckerTool(BaseTool):
    name: str = "Availability Checker Tool"
    description: str = "Checks the shop's next available appointment slot."

    def _run(self, **kwargs) -> str:
        log_tool(self.name, "Checking availability")
        # Simulate some simple availability logic
        available_slots = [
            "Next Monday afternoon",
            "Next Tuesday morning",
            "Next Wednesday all day",
            "Fully booked for the next few days, try next week.",
        ]
        availability = random.choice(available_slots)
        time.sleep(0.1)
        log_tool(self.name, "Found slot", availability)
        return f"Shop Availability: {availability}"

# Tool for Shop Manager to communicate with Mechanic
class ManagerToMechanicTool(BaseTool):
    name: str = "Manager-to-Mechanic Communication Tool"
    description: str = "Allows the shop manager to communicate with the mechanic."

    def _run(self, message_type: str, details: str) -> str:
        log_tool(self.name, f"Sending {message_type} message", details)
        time.sleep(0.2)  # Simulate communication delay
        
        responses = {
            "assignment": "Maintenance assignment received. Will prioritize accordingly.",
            "inquiry": "Will check status and report back soon.",
            "priority": "Priority update acknowledged and adjusted workflow."
        }
        
        response = responses.get(message_type.lower(), "Message received and acknowledged.")
        log_tool(self.name, "Mechanic response", response)
        return f"Mechanic response: {response}"

# Simple Tools for the Shop Crew
class MaintenanceCheckerTool(BaseTool):
    name: str = "Maintenance Checker Tool"
    description: str = "Performs maintenance checks and sometimes discovers additional repair needs."

    def _run(self, maintenance_task: str) -> str:
        log_tool(self.name, f"Checking '{maintenance_task}'")
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
            log_tool(self.name, "FOUND ISSUE", issue)
            return f"While performing {maintenance_task}, {issue}. Additional repair is necessary."
        else:
            log_tool(self.name, "No issues found")
            return f"Maintenance task '{maintenance_task}' completed successfully. No additional issues found."

# Define a tool for mechanic to communicate with shop manager
class MechanicToManagerTool(BaseTool):
    name: str = "Mechanic-Manager Communication Tool"
    description: str = "Sends a message from the mechanic to the shop manager about availability or status updates."

    def _run(self, message_type: str, details: str) -> str:
        log_tool(self.name, f"Sending {message_type} update", details)
        time.sleep(0.2)  # Simulate communication delay
        
        # Could be enhanced to actually send to manager agent in a more complex setup
        responses = {
            "availability": "Availability update received and logged.",
            "status": "Status update received and logged.",
            "repair": "Additional repair notification received and processed."
        }
        
        response = responses.get(message_type.lower(), "Message received.")
        log_tool(self.name, "Manager response", response)
        return f"Manager response: {response}"

# --- Instantiate Tools ---
cross_crew_tool = CrossCrewCommunicationTool()
quote_tool = QuoteGeneratorTool()
availability_tool = AvailabilityCheckerTool()
manager_to_mechanic_tool = ManagerToMechanicTool()
maintenance_tool = MaintenanceCheckerTool()
mechanic_comm_tool = MechanicToManagerTool()


# ==================================
# --- Customer Crew Definition ---
# ==================================

log_header("Defining Customer Crew")

# 1. Load Customer Config (same as before)
config_file = "customer_config.json"
customer_data = {}
try:
    with open(config_file, 'r') as f:
        customer_data = json.load(f)
    log_step("Config", f"Loaded customer data from {config_file}")
    # 顧客データをわかりやすく表示
    for key, value in customer_data.items():
        if key == "availability":
            print(f"  - {key}: {', '.join(value)}")
        else:
            print(f"  - {key}: {value}")
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
)

task2_send_request_and_get_response = Task(
    description=f"""
        Take the repair request message generated in the previous task.
        Use the 'Cross-Crew Communication Tool' to send this message to the crew registered
        with the ID 'shop_crew_main'.

        When using the Cross-Crew Communication Tool, you MUST provide these exact parameters:
        1. 'target_crew_id': "shop_crew_main"  (EXACTLY this string, do not attempt any other IDs)
        2. 'message': which should be the complete message from your previous task

        Do not attempt to use any other target_crew_id values like "auto_repair_shop_123" or "auto_shop_001".
        ONLY 'shop_crew_main' will work.
        
        Your final output should be ONLY the response received back from the 'shop_crew_main'.
    """,
    expected_output="The response message received from the shop crew via the communication tool.",
    agent=customer_agent,
    context=[task1_prepare_request]
)

# 4. Create Customer Crew
customer_crew = Crew(
    agents=[customer_agent],
    tasks=[task1_prepare_request, task2_send_request_and_get_response],
    process=Process.sequential,
    verbose=0  # 詳細出力を無効化
)

# =============================
# --- Shop Crew Definition ---
# =============================

log_header("Defining Shop Crew")

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
    verbose=0  # 詳細出力を無効化
)

# =============================
# --- Mechanic Crew Definition ---
# =============================

log_header("Defining Mechanic Crew")

# Define Mechanic Agent
mechanic_agent = Agent(
    role='Automotive Mechanic',
    goal='Perform maintenance tasks efficiently, identify additional repair needs, and communicate clearly with the shop manager.',
    backstory="Experienced mechanic with 15 years in the automotive repair industry. Known for thoroughness and attention to detail.",
    llm=openrouter_llm,
    tools=[maintenance_tool, mechanic_comm_tool, cross_crew_tool],
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
    verbose=0  # 詳細出力を無効化
)

# =============================
# --- Orchestration ---
# =============================

log_header("Registering Crews")
# Register the shop crew so the customer crew can find it via the tool
crew_registry['shop_crew_main'] = shop_crew
# Register the mechanic crew
crew_registry['mechanic_crew'] = mechanic_crew
# Add car_owner as an alias for customer_crew
crew_registry['car_owner'] = customer_crew
log_step("Registry", f"Registered crews: {list(crew_registry.keys())}")


log_header("Execution Start")
log_step("System", "Sending customer request...")

# Running the customer crew will trigger the sequence:
# 1. Customer agent prepares the message.
# 2. Customer agent uses the tool to send the message to 'shop_crew_main'.
# 3. The tool looks up 'shop_crew_main' and calls its kickoff() with the message.
# 4. The Shop Crew runs its task (processing the message, using its tools).
# 5. The Shop Crew finishes and returns its result (the response message).
# 6. The tool receives this response and returns it to the Customer Agent.
# 7. The Customer Crew finishes, its final result being the shop's response.

response_from_shop = customer_crew.kickoff()

log_header("Customer Request Completed")
log_response("Shop Response", response_from_shop)

log_header("Starting Maintenance")
maintenance_info = {
    "maintenance_task": "Check engine light diagnostics and brake inspection", 
    "car_info": "2020 Subaru Outback",
    "priority": "Normal"
}

# Simplified maintenance info
log_step("Assignment", f"{maintenance_info['car_info']} - {maintenance_info['maintenance_task']}")

try:
    mechanic_result = mechanic_crew.kickoff(inputs=maintenance_info)
    
    log_header("Maintenance Complete")
    log_response("Mechanic Report", mechanic_result)
    
    # Simplified final report
    customer_friendly_report = f"Service completed: {maintenance_info['car_info']} {maintenance_info['maintenance_task']} is done. {str(mechanic_result).split('.')[0]}."
    
    log_response("Customer Report", customer_friendly_report)
    
except Exception as e:
    print(f"\nError: {e}")