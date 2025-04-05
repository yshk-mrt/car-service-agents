import os
import json
import random
import time
from typing import ClassVar, Optional
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

def log_tool(*args):
    """Log tool usage with consistent formatting."""
    if len(args) == 1:
        # Single argument format: log_tool("message")
        print(f"→ {args[0]}")
    elif len(args) == 2:
        # Two argument format: log_tool("tool_name", "message")
        print(f"→ {args[0]}: {args[1]}")
    elif len(args) == 3:
        # Three argument format: log_tool("tool_name", "status", "message")
        print(f"→ {args[0]}: {args[1]} ({args[2]})")
    else:
        # Default case
        print(f"→ Tool log: {' '.join(str(arg) for arg in args)}")

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
class CrossCrewCommunicationSchema(BaseModel):
    """Input schema for CrossCrewCommunicationTool."""
    target_crew_id: str = Field(..., description="The registered ID of the crew to talk to")
    message: str = Field(..., description="The content to send to the target crew")

# Tool for Crew 1 (Customer) to talk to Crew 2 (Shop)
class CrossCrewCommunicationTool(BaseTool):
    """Tool for sending messages between crews."""
    name: str = "Cross-Crew Communication Tool"
    description: str = "Sends a message (e.g., a service request) to another specified Crew and returns its response. Input must be a dictionary with 'target_crew_id': string (the registered ID of the crew to talk to) and 'message': string (the content to send)."
    args_schema: type = CrossCrewCommunicationSchema
    
    # Class variables to track communication state
    customer_original_message: ClassVar[str] = ""
    shop_initial_proposal_sent: ClassVar[bool] = False
    customer_agreed_to_service: ClassVar[bool] = False
    car_dropped_off: ClassVar[bool] = False
    service_completed: ClassVar[bool] = False
    customer_to_shop_count: ClassVar[int] = 0
    shop_to_customer_count: ClassVar[int] = 0
    
    def _run(self, target_crew_id: str, message: str) -> str:
        """
        Run the tool with the given input.
        """
        # Determine sender based on the target crew
        if target_crew_id == "shop_crew_main":
            sender = "Car Owner"  # If sending to shop, sender must be customer
        elif target_crew_id == "car_owner":
            sender = "Auto Shop Service Manager"  # If sending to customer, sender must be shop
        elif target_crew_id == "mechanic_crew":
            sender = "Auto Shop Service Manager"  # If sending to mechanic, sender must be shop
        else:
            return f"Error: Target Crew '{target_crew_id}' not found. Valid crew IDs are: {list(crew_registry.keys())}. DO NOT try to use this tool again - simply write your direct response."
        
        # Get the target crew
        target_crew = crew_registry.get(target_crew_id)
        if not target_crew:
            return f"Error: Target Crew '{target_crew_id}' not found. Valid crew IDs are: {list(crew_registry.keys())}. DO NOT try to use this tool again - simply write your direct response."
        
        # Log the communication
        log_tool(f"{sender} → {target_crew_id}")
        
        # In the orchestration flow, we're handling the communication separately
        # This tool should only be used for the initial customer request and mechanic communication
        # For all other communications, just acknowledge receipt
        
        # Handle customer to shop communication - only for initial request
        if sender == "Car Owner" and target_crew_id == "shop_crew_main":
            # Only process the very first message through this tool
            if CrossCrewCommunicationTool.customer_to_shop_count == 0:
                CrossCrewCommunicationTool.customer_to_shop_count += 1
                CrossCrewCommunicationTool.customer_original_message = message
                return "Message received. This will be processed in the orchestration flow."
            else:
                # For follow-up messages, don't process them here
                return "DO NOT use this tool. Simply write your direct response without using any tools."
            
        # Handle shop to mechanic communication
        elif sender == "Auto Shop Service Manager" and target_crew_id == "mechanic_crew":
            original_tasks = target_crew.tasks
            target_crew.tasks = [task_perform_service]
            response = target_crew.kickoff(inputs={'service_request': message})
            target_crew.tasks = original_tasks
            
            # Mark car service as completed after mechanic responds
            CrossCrewCommunicationTool.service_completed = True
            
            return str(response)
        
        # Default case for any other communication
        else:
            # Just acknowledge receipt
            return "DO NOT use this tool. Simply write your direct response without using any tools."

# Simple Tools for the Shop Crew
class QuoteGeneratorTool(BaseTool):
    name: str = "Quote Generator Tool"
    description: str = "Generates a price quote and estimated repair time based on a car problem description."
    
    def _run(self, problem_description: str) -> str:
        """Generate a quote based on the problem description."""
        log_tool(self.name, f"Generating quote")
        
        # Simulate quote generation with random values
        cost = random.randint(800, 2000)
        days = random.randint(3, 10)
        
        # Ensure the quote is over $1000 for negotiation purposes
        if cost < 1000:
            cost = 1000 + random.randint(0, 500)
        
        log_tool(self.name, f"Quote ready (${cost}, {days} days)")
        return f"Quote generated: Estimated cost ${cost}, Estimated time {days} business days."

class AvailabilityCheckerTool(BaseTool):
    name: str = "Availability Checker Tool"
    description: str = "Checks the shop's next available appointment slot."
    
    def _run(self) -> str:
        """Check the next available appointment slot."""
        log_tool(self.name, f"Checking availability")
        
        # Simulate availability checking with random options
        options = [
            "Next Monday afternoon",
            "Tuesday morning, 9-11 AM",
            "Wednesday afternoon, 2-4 PM",
            "This Friday, any time",
            "Fully booked for the next few days, try next week."
        ]
        availability = random.choice(options)
        
        log_tool(self.name, f"Found slot ({availability})")
        return f"Shop Availability: {availability}"

# Tool for Shop Manager to communicate with Mechanic
class ManagerToMechanicCommunicationTool(BaseTool):
    name: str = "Manager-to-Mechanic Communication Tool"
    description: str = "Allows the shop manager to communicate with the mechanic."
    
    class ManagerToMechanicSchema(BaseModel):
        message_type: str = Field(..., description="Type of message (e.g., 'assignment', 'inquiry', 'update')")
        details: str = Field(..., description="The content of the message")
    
    args_schema: type = ManagerToMechanicSchema
    
    def _run(self, message_type: str, details: str) -> str:
        """Send a message from the manager to the mechanic."""
        log_tool(self.name, f"Sending {message_type} update ({details[:30]}...)")
        
        # Simulate mechanic response
        responses = {
            "assignment": "Assignment received and logged. Will begin work immediately.",
            "inquiry": "Received your inquiry. Based on my assessment, the issue is related to...",
            "update": "Update received and acknowledged. Will adjust the work accordingly.",
            "availability": "Availability update received and logged.",
            "repair": "Additional repair notification received and processed.",
            "status_update": "Status update received and logged.",
            "status": "Status update received and logged."
        }
        
        response = responses.get(message_type, "Message received.")
        log_tool(self.name, f"Manager response ({response[:30]}...)")
        return response

# Tools for the Mechanic Crew
class MaintenanceCheckerTool(BaseTool):
    name: str = "Maintenance Checker Tool"
    description: str = "Checks if a maintenance task reveals additional repair needs."
    
    def _run(self, maintenance_task: str) -> str:
        """Check if a maintenance task reveals additional repair needs."""
        log_tool(self.name, f"Checking '{maintenance_task}'")
        
        # 30% chance of finding an issue
        if random.random() < 0.3:
            issue = random.choice([
                "detected an issue with the alternator that wasn't initially reported",
                "found excessive wear on the brake pads that should be addressed",
                "discovered a small oil leak that should be fixed",
                "noticed the air filter needs replacement",
                "found that the transmission fluid needs to be changed"
            ])
            log_tool(self.name, f"FOUND ISSUE ({issue[:30]}...)")
            return f"During the {maintenance_task}, I {issue}. This will require additional work."
        else:
            log_tool(self.name, "No issues found")
            return f"The maintenance task \"{maintenance_task}\" was successfully performed, and no additional repairs are needed."

# Define a tool for mechanic to communicate with shop manager
class MechanicManagerCommunicationTool(BaseTool):
    name: str = "Mechanic-Manager Communication Tool"
    description: str = "Sends a message from the mechanic to the shop manager about availability or status updates."
    
    class MechanicToManagerSchema(BaseModel):
        message_type: str = Field(..., description="Type of message (e.g., 'availability', 'status_update', 'repair')")
        details: str = Field(..., description="The content of the message")
    
    args_schema: type = MechanicToManagerSchema
    
    def _run(self, message_type: str, details: str) -> str:
        """Send a message from the mechanic to the shop manager."""
        log_tool(self.name, f"Sending {message_type} update ({details[:30]}...)")
        
        # Simulate manager response
        responses = {
            "availability": "Availability update received and logged.",
            "status_update": "Status update received and logged.",
            "repair": "Additional repair notification received and processed.",
            "status": "Status update received and logged."
        }
        
        response = responses.get(message_type, "Message received.")
        log_tool(self.name, f"Manager response ({response[:30]}...)")
        return f"Manager response: {response}"

# Initialize the tools
quote_tool = QuoteGeneratorTool()
availability_tool = AvailabilityCheckerTool()
maintenance_tool = MaintenanceCheckerTool()
manager_to_mechanic_tool = ManagerToMechanicCommunicationTool()
mechanic_to_manager_tool = MechanicManagerCommunicationTool()

# 1. Create Customer Agent
customer_agent = Agent(
    role='Car Owner',
    goal='Clearly describe car issues and availability, then send this request to the auto shop crew and relay the response.',
    backstory="You own a 2020 Subaru Outback with some issues that need repair. You need to communicate these issues to an auto shop.",
    llm=openrouter_llm,
    tools=[],  # We'll add tools after agent creation
    verbose=True,
    allow_delegation=False
)

# Create a CrossCrewCommunicationTool instance for the customer agent
customer_comm_tool = CrossCrewCommunicationTool()
customer_agent.tools = [customer_comm_tool]

# 2. Create Shop Manager Agent
shop_manager_agent = Agent(
    role='Auto Shop Service Manager',
    goal='Receive customer repair requests, generate quotes using available tools, check availability, and provide clear responses.',
    backstory="Experienced service manager responsible for handling customer inquiries, providing estimates, and scheduling appointments.",
    llm=openrouter_llm,
    tools=[],  # We'll add tools after agent creation
    verbose=True,
    allow_delegation=False
)

# Create a CrossCrewCommunicationTool instance for the shop manager agent
shop_manager_comm_tool = CrossCrewCommunicationTool()
shop_manager_agent.tools = [
    quote_tool,
    availability_tool,
    manager_to_mechanic_tool,
    shop_manager_comm_tool
]

# 3. Create Mechanic Agent
mechanic_agent = Agent(
    role='Auto Mechanic',
    goal='Fix cars properly and efficiently',
    backstory="You are an experienced auto mechanic with expertise in various car models. Your job is to diagnose and fix car issues accurately and efficiently.",
    llm=openrouter_llm,
    tools=[],  # We'll add tools after agent creation
    verbose=True,
    allow_delegation=False
)

# Create a CrossCrewCommunicationTool instance for the mechanic agent
mechanic_comm_tool = CrossCrewCommunicationTool()
mechanic_agent.tools = [
    maintenance_tool,
    mechanic_to_manager_tool,
    mechanic_comm_tool
]

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

# 2. Define Customer Tasks
#    - Task 1: Prepare the request message (same as before)
#    - Task 2: Use the tool to send the message from Task 1 to the 'shop_crew'
#    - Task 3: Agree to service and drop off the car
#    - Task 4: Receive car ready notification

task1_prepare_request = Task(
    description=f"""
        Review your current situation based on the information provided below.
        Prepare a concise request message suitable for sending to an auto repair shop.
        The message should clearly state:
        1. Your car model: 2020 Subaru Outback
        2. The specific issues you are experiencing: Check engine light came on yesterday. Also, there's a slight squealing noise when I brake, especially downhill.
        3. Your availability for bringing the car in: Monday, April 7th, 2025 - Morning, Wednesday, April 9th, 2025 - All Day, Thursday, April 10th, 2025 - Afternoon (after 1 PM), Friday, April 11th, 2025 - Morning (before 1 PM)

        IMPORTANT: DO NOT use any tools or attempt to communicate with anyone. Simply write your direct response.
        
        Output ONLY the formatted request message text, nothing else.
    """,
    expected_output="A concise repair request message containing car model, issues, and availability.",
    agent=customer_agent,
)

task2_send_request_and_get_response = Task(
    description=f"""
        You are responding to the shop's initial proposal.
        
        The shop has sent you a proposal with pricing and appointment details.
        
        Create a follow-up message that addresses:
        
        1. The pricing being over $1,000 - ask for a discount
        
        2. Confirm that you need shuttle service
        
        You MUST use EXACTLY this template, replacing only the [placeholders]:
        
        ```
        Dear [Auto Shop Name],

        Thank you for getting back to me with the estimate and appointment details. I noticed that the repair cost of $[EXACT AMOUNT] is over $1,000. Is there any possibility of a discount, as the price seems a bit high?

        Also, I would like to confirm that I will need shuttle service during the appointment. Thank you for offering this convenience.

        Please let me know if these adjustments can be accommodated. I appreciate your help!

        Best regards,
        [Customer Name]
        ```
        
        IMPORTANT: 
        - Replace [Auto Shop Name] with "Superior Auto Shop"
        - Replace [EXACT AMOUNT] with the exact dollar amount from the shop's proposal (e.g., "1,747")
        - Replace [Customer Name] with "Alex Johnson"
        - DO NOT use any tools or attempt to communicate with anyone. Simply write your direct response using the template above.
        
        Your final output should be ONLY the follow-up message you created (using the exact template above).
    """,
    expected_output="""A follow-up message addressing both price concerns and shuttle service.""",
    agent=customer_agent
)

task3_agree_and_drop_off = Task(
    description=f"""
        After receiving the shop's response to your follow-up message about pricing and shuttle service,
        you need to confirm the appointment and drop off your car.
        
        Create a message that:
        1. Thanks the shop for the discount and shuttle service confirmation
        2. Confirms that you accept the final price
        3. Confirms that you will drop off your car at the scheduled appointment time
        4. Asks about the estimated completion time
        
        Your message should be polite and clear.
        
        IMPORTANT: DO NOT use any tools or attempt to communicate with anyone. Simply write your direct response.
        
        Your final output should be your confirmation and car drop-off message.
    """,
    expected_output="A polite confirmation message accepting the shop's terms and confirming car drop-off.",
    agent=customer_agent
)

task4_receive_car_ready_notification = Task(
    description=f"""
        You have received a notification from the auto shop that your car is ready for pickup.
        
        Review the message from the shop, which will be provided to you as 'shop_notification'.
        
        Respond with a message that:
        1. Thanks the shop for completing the service
        2. Confirms when you will pick up the car
        3. Asks if there's anything you need to know about the repairs
        
        Your message should be polite and appreciative.
        
        IMPORTANT: DO NOT use any tools or attempt to communicate with anyone. Simply write your direct response to the message that was provided to you.
        
        Your final output should be your response to the car ready notification.
    """,
    expected_output="A polite response to the car ready notification, confirming pickup details.",
    agent=customer_agent
)

# 3. Create Customer Crew
customer_crew = Crew(
    agents=[customer_agent],
    tasks=[task1_prepare_request, task2_send_request_and_get_response, task3_agree_and_drop_off, task4_receive_car_ready_notification],
    process=Process.sequential,
    verbose=0  # 詳細出力を無効化
)

# =============================
# --- Shop Crew Definition ---
# =============================

log_header("Defining Shop Crew")

# 2. Create Shop Manager Agent
shop_manager_agent = Agent(
    role='Auto Shop Service Manager',
    goal='Receive customer repair requests, generate quotes using available tools, check availability, and provide clear responses.',
    backstory="Experienced service manager responsible for handling customer inquiries, providing estimates, and scheduling appointments.",
    llm=openrouter_llm,
    tools=[
        quote_tool,
        availability_tool,
        manager_to_mechanic_tool,
        shop_manager_comm_tool
    ],
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
        "3. If the quote is less than $1,000, manually increase it to be over $1,000 to ensure the customer will negotiate the price.\n"
        "4. Use the 'Availability Checker Tool' to find the next available slot.\n"
        "5. Synthesize this information into a clear initial proposal for the customer that includes:\n"
        "   - A greeting and acknowledgment of their request\n"
        "   - The quote estimate and time estimate (make sure the price is over $1,000)\n"
        "   - The next available appointment slot\n"
        "   - An explicit question asking if they require shuttle service\n"
        "6. Make sure your response is professional, clear, and structured as a formal proposal.\n"
        "If the quote tool returns an error or the availability tool indicates no slots, mention that politely in the response."
    ),
    expected_output="An initial proposal message addressed to the customer containing the quote estimate (over $1,000), time estimate, availability information, and explicitly asking if they require shuttle service.",
    agent=shop_manager_agent,
)

# Add a new task for the shop manager to handle the customer's follow-up message about price and shuttle service
task_respond_to_followup = Task(
    description=f"""
        You have received a follow-up message from a customer responding to your initial proposal. The message content is provided in the input variable 'follow_up_message'.
        
        The original price quoted and the discounted price (15% off) are provided in the inputs 'original_price' and 'discounted_price'.
        
        1. Carefully analyze the follow-up message to understand the customer's concerns or requests.
        2. The customer has mentioned that the price is too high:
           - Offer a 15% discount on the original quote
           - Clearly state both the original price and the new discounted price
           - Explain that this is a special discount for valued customers
        3. The customer has confirmed they need shuttle service:
           - Acknowledge this request
           - Confirm that complimentary shuttle service will be provided
           - Mention the shuttle service hours (8am-5pm)
        4. Provide a revised, comprehensive final response that includes:
           - A friendly greeting
           - The discount applied to the original quote
           - Confirmation of shuttle service
           - Reconfirmation of the appointment availability
           - A thank you message for choosing your shop
           - A clear next step (e.g., 'Please confirm this appointment time works for you')
        5. DO NOT use any tools to generate a new quote or check availability again - simply reference the information from the initial proposal.
        6. Make sure your response is friendly, professional, and addresses all the customer's concerns in a single, coherent message.
        
        IMPORTANT:
        - Replace [Customer Name] with "Alex Johnson"
        - Use the exact original price of $[original_price] in your response
        - Use the exact discounted price of $[discounted_price] in your response
        - Replace [Your Name] with "Michael Rodriguez"
        - Replace [Auto Shop Name] with "Superior Auto Shop"
        - Replace [Contact Information] with "Phone: (555) 123-4567, Email: service@superiorauto.com"
    """,
    expected_output="""A comprehensive response addressing the customer's concerns about pricing and shuttle service.""",
    agent=shop_manager_agent,
)

# Add a new task for the shop manager to process car drop-off
task_process_car_dropoff = Task(
    description=(
        "You have received a message from a customer confirming their appointment and dropping off their car. "
        "The message content is provided in the input variable '{received_message}'.\n"
        "1. Thank the customer for confirming the appointment and dropping off their car.\n"
        "2. Acknowledge receipt of their vehicle.\n"
        "3. Provide an estimated completion time (e.g., 'end of the day' or 'by tomorrow afternoon').\n"
        "4. Confirm that you will notify them when the car is ready for pickup.\n"
        "5. Remind them about the shuttle service if they mentioned needing it.\n"
        "6. Make sure your response is professional, clear, and reassuring.\n"
    ),
    expected_output="A professional acknowledgment of the car drop-off with estimated completion time.",
    agent=shop_manager_agent
)

# Add a new task for the shop manager to communicate with the mechanic about the service
task_assign_to_mechanic_for_service = Task(
    description=(
        "A customer has dropped off their car for service. You need to communicate with the mechanic to get the work done.\n"
        "You will be provided with a message to send to the mechanic in the 'mechanic_message' input.\n"
        "1. Use the Cross-Crew Communication Tool to send the message to the mechanic crew ('mechanic_crew').\n"
        "2. Wait for the mechanic's response confirming the work has been completed.\n"
        "3. Your final output should include the mechanic's response about the completed service.\n"
        "Make sure to use the Cross-Crew Communication Tool correctly with these parameters:\n"
        "- target_crew_id: 'mechanic_crew'\n"
        "- message: The mechanic_message provided to you\n"
    ),
    expected_output="A report from the mechanic about the completed service.",
    agent=shop_manager_agent
)

# 3. Create Shop Crew
shop_crew = Crew(
    agents=[shop_manager_agent],
    tasks=[task_process_request, task_respond_to_followup, task_process_car_dropoff, task_assign_to_mechanic_for_service],
    process=Process.sequential,
    verbose=0  # 詳細出力を無効化
)

# =============================
# --- Mechanic Crew Definition ---
# =============================

log_header("Defining Mechanic Crew")

# 3. Create Mechanic Agent
mechanic_agent = Agent(
    role='Auto Mechanic',
    goal='Fix cars properly and efficiently',
    backstory="You are an experienced auto mechanic with expertise in various car models. Your job is to diagnose and fix car issues accurately and efficiently.",
    llm=openrouter_llm,
    tools=[
        maintenance_tool,
        mechanic_to_manager_tool,
        mechanic_comm_tool
    ],
    verbose=True,
    allow_delegation=False
)

# Define Mechanic Tasks
task_report_availability = Task(
    description="""
        Report your current availability to the shop manager.
        
        Use the Mechanic-Manager Communication Tool to send an availability update:
        - message_type: "availability"
        - details: A brief description of your current workload and availability
        
        Your output should confirm that you've reported your availability.
    """,
    expected_output="Confirmation that availability has been reported to the shop manager.",
    agent=mechanic_agent
)

task_perform_service = Task(
    description="""
        You have been assigned a service task for a customer's vehicle.
        The service request details are provided in the 'service_request' input.
        
        Use the Maintenance Checker Tool to perform the maintenance and check if additional repairs are needed.
        Pass the specific maintenance task description from the service_request parameter if available, 
        otherwise use "standard maintenance service".
        
        If the tool indicates additional repairs are needed, use the Mechanic-Manager Communication Tool to inform the manager about these additional repair needs:
        - message_type: "repair"
        - details: The specifics of the additional repair needs
        
        After completing the service, use the Mechanic-Manager Communication Tool to update the manager:
        - message_type: "status_update"
        - details: A summary of the work completed
        
        Your output should include what maintenance you performed and whether additional repairs are needed.
    """,
    expected_output="Description of maintenance performed and any additional repair needs identified.",
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
    tasks=[task_perform_service, task_perform_maintenance],
    verbose=True,
    process=Process.sequential,
    name="mechanic_crew"
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


log_header("Starting Service Process")
log_step("System", "Phase 1: Initial customer request...")

# PHASE 1: Customer initial request
# First, get the customer's initial request
log_step("System", "Customer generating request...")
# Set up the customer crew to only run the first task
original_tasks = customer_crew.tasks
customer_crew.tasks = [task1_prepare_request]
customer_request_output = customer_crew.kickoff()
customer_crew.tasks = original_tasks

# Convert CrewOutput to string
customer_request = str(customer_request_output)
log_response("Customer Request", customer_request)

# PHASE 2: Shop manager initial proposal
# Send the customer request to the shop manager
log_step("System", "Phase 2: Shop manager initial proposal...")
# Reset communication counters
CrossCrewCommunicationTool.customer_to_shop_count = 0
CrossCrewCommunicationTool.shop_to_customer_count = 0
CrossCrewCommunicationTool.customer_original_message = customer_request

# Use the task directly
original_tasks = shop_crew.tasks
shop_crew.tasks = [task_process_request]
shop_proposal_output = shop_crew.kickoff(inputs={"received_message": customer_request})
shop_crew.tasks = original_tasks

# Convert CrewOutput to string
shop_proposal = str(shop_proposal_output)

# Mark that the shop has sent the initial proposal
CrossCrewCommunicationTool.shop_initial_proposal_sent = True
log_response("Shop Initial Proposal", shop_proposal)

# PHASE 3: Customer follow-up about price and shuttle
# Create a follow-up message from the customer about price and shuttle service
log_step("System", "Phase 3: Customer follow-up about price and shuttle...")

# Extract the price from the shop proposal
import re
price_match = re.search(r'\$(\d+,?\d*)', shop_proposal)
price = price_match.group(1) if price_match else "1,747"

# Set up the customer crew to run the second task
original_tasks = customer_crew.tasks
customer_crew.tasks = [task2_send_request_and_get_response]
# Pass the shop proposal and extracted price to the customer as input
follow_up_output = customer_crew.kickoff(inputs={
    "shop_proposal": shop_proposal,
    "exact_amount": price
})
customer_crew.tasks = original_tasks

# Extract the follow-up message from the output
follow_up_message = str(follow_up_output)
# Replace any remaining [EXACT AMOUNT] placeholders with the actual price
follow_up_message = follow_up_message.replace("[EXACT AMOUNT]", price)
log_response("Customer Follow-up", follow_up_message)

# PHASE 4: Shop manager responds to follow-up
# Create a response from the shop manager to the customer's follow-up
log_step("System", "Phase 4: Shop manager responds to follow-up...")

# Calculate the discounted price (15% off)
try:
    original_price = float(price.replace(",", ""))
    discounted_price = original_price * 0.85
    discounted_price_formatted = f"{discounted_price:,.0f}"
except ValueError:
    # Fallback if price extraction fails
    original_price = 1747
    discounted_price_formatted = "1,485"

# Set up the shop crew to run the task_respond_to_followup task
original_tasks = shop_crew.tasks
shop_crew.tasks = [task_respond_to_followup]
# Pass the follow-up message to the shop manager as input
shop_response = shop_crew.kickoff(inputs={
    "follow_up_message": follow_up_message,
    "original_price": price,
    "discounted_price": discounted_price_formatted
})
shop_crew.tasks = original_tasks

# Extract the shop response from the output
shop_response_str = str(shop_response)
# Replace any remaining placeholders with actual values
shop_response_str = shop_response_str.replace("[original_price]", price)
shop_response_str = shop_response_str.replace("[discounted_price]", discounted_price_formatted)
log_response("Shop Response to Follow-up", shop_response_str)

# PHASE 5: Customer agrees and drops off car
log_step("System", "Phase 5: Customer agrees and drops off car...")

# Set up the customer crew to run the third task
original_tasks = customer_crew.tasks
customer_crew.tasks = [task3_agree_and_drop_off]
# Pass the shop response to the customer as input
car_dropoff_output = customer_crew.kickoff(inputs={"shop_response": shop_response_str})
customer_crew.tasks = original_tasks

# Extract the car drop-off message from the output
car_dropoff_message = str(car_dropoff_output)
log_response("Customer Agreement", car_dropoff_message)

# Set car dropped off flag
CrossCrewCommunicationTool.customer_agreed_to_service = True
CrossCrewCommunicationTool.car_dropped_off = True

# PHASE 6: Shop acknowledges car drop-off
# Shop acknowledges car drop-off
original_tasks = shop_crew.tasks
shop_crew.tasks = [task_process_car_dropoff]
shop_dropoff_response_output = shop_crew.kickoff(inputs={"received_message": car_dropoff_message})
shop_crew.tasks = original_tasks

# Convert CrewOutput to string
shop_dropoff_response = str(shop_dropoff_response_output)
log_response("Shop Acknowledges Drop-off", shop_dropoff_response)

# PHASE 7: Car Service
log_step("System", "Phase 7: Car Service...")

# Shop manager assigns car to mechanic
mechanic_request_message = f"""
Vehicle Details: 2020 Subaru Outback
Issues to address: Check engine light and squealing noise during braking.
Service Details: Perform inspection and repairs as necessary to resolve the stated issues.
Special instructions: Please prioritize the work to meet the estimated completion time.
Let me know once the work is completed.
"""

# Set up the shop crew to communicate with mechanic
original_tasks = shop_crew.tasks
shop_crew.tasks = [task_assign_to_mechanic_for_service]
mechanic_service_output = shop_crew.kickoff(inputs={"mechanic_message": mechanic_request_message})
shop_crew.tasks = original_tasks

# Convert CrewOutput to string
mechanic_service_report = str(mechanic_service_output)
log_response("Mechanic Service Report", mechanic_service_report)

# PHASE 8: Shop manager informs customer car is ready
log_step("System", "Phase 8: Shop manager informs customer car is ready...")

# Prepare car ready notification
car_ready_notification = f"""
Dear Customer,

Great news! Your 2020 Subaru Outback is now ready for pickup. 

Our mechanic has completed all the necessary repairs:
- Diagnosed and resolved the check engine light issue
- Fixed the squealing noise when braking
- Performed a complete safety inspection

Everything has been completed according to the agreed price of {shop_response_str.split('$')[1].split(' ')[0] if '$' in shop_response_str else '$1,417.80'}.

You can pick up your vehicle anytime during our business hours (8:00 AM - 6:00 PM). 
Please bring your ID and the original work order.

If you need our shuttle service for pickup, please let us know in advance.

Thank you for choosing our shop. We look forward to seeing you soon!

Best regards,
Auto Shop Service Manager
"""

# Set up the customer crew to receive car ready notification
original_tasks = customer_crew.tasks
customer_crew.tasks = [task4_receive_car_ready_notification]
pickup_response_output = customer_crew.kickoff(inputs={"shop_notification": car_ready_notification})
customer_crew.tasks = original_tasks

# Convert CrewOutput to string
pickup_response = str(pickup_response_output)
log_response("Customer Pickup Response", pickup_response)

# Service process complete
log_step("System", "Full service cycle completed successfully")