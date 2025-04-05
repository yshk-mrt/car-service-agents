import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get OpenRouter API key
openrouter_api_key = os.getenv("OpenRouterKey")
if not openrouter_api_key:
    raise ValueError("OpenRouterKey environment variable is required")

# Configure OpenAI client to use OpenRouter
client = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/openrouter-dev/openrouter-examples",
        "X-Title": "Car Service Agents CrewAI"
    }
)

# Test that the OpenRouter connection works
try:
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Say hello!"}]
    )
    print("OpenRouter test successful:", response.choices[0].message.content)
except Exception as e:
    print(f"OpenRouter test failed: {e}")
    raise

# Set up environment variables for CrewAI to use OpenRouter
os.environ["OPENAI_API_KEY"] = openrouter_api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_MODEL_NAME"] = "openai/gpt-4o"  # Changed from anthropic to openai model

# Define your agents with roles, goals, and additional attributes
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory=(
        "You are a Senior Research Analyst at a leading tech think tank. "
        "Your expertise lies in identifying emerging trends and technologies in AI and data science. "
        "You have a knack for dissecting complex data and presenting actionable insights."
    ),
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory=(
        "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation. "
        "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
    ),
    verbose=True,
    allow_delegation=True,
    cache=False  # Disable cache for this agent
)

# Create tasks for your agents
task1 = Task(
    description=(
        "Create a comprehensive analysis of the latest advancements in AI in 2025. "
        "Identify key trends, breakthrough technologies, and potential industry impacts. "
        "Compile your findings in a detailed report. "
        "Make sure to check with a human if the draft is good before finalizing your answer."
    ),
    expected_output='A comprehensive full report on the latest AI advancements in 2025, leave nothing out',
    agent=researcher,
    human_input=True
)

task2 = Task(
    description=(
        "Using the insights from the researcher\'s report, develop an engaging blog post that highlights the most significant AI advancements. "
        "Your post should be informative yet accessible, catering to a tech-savvy audience. "
        "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
    ),
    expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2025',
    agent=writer,
    human_input=True
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    memory=True,
    planning=True  # Enable planning feature for the crew
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)