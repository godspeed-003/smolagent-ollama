from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool
from ollama_model import OllamaModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
import random

@tool
def my_custom_tool() -> str:
    """A tool that returns a random fun fact to surprise and delight users."""
    fun_facts = [
        "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs!",
        "Bananas are berries, but strawberries are not.",
        "Octopuses have three hearts.",
        "A group of flamingos is called a 'flamboyance'.",
        "There are more stars in the universe than grains of sand on Earth.",
        "Some turtles can breathe through their butts!",
        "The Eiffel Tower can be 15 cm taller during hot days.",
        "Wombat poop is cube-shaped."
    ]
    return random.choice(fun_facts)

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
visit_webpage_tool = VisitWebpageTool()
web_search_tool = DuckDuckGoSearchTool()

model = OllamaModel(model_name="llama3.1:latest", max_tokens=2096, temperature=0.5)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Instantiate local tools
visit_webpage_tool = VisitWebpageTool()
web_search_tool = DuckDuckGoSearchTool()

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# Ensure final_answer is structured with pre_messages and post_messages for smolagents
prompt_templates["final_answer"] = {
    "pre_messages": (
        "# Final Answer Prompt\n"
        "You are now ready to provide the final answer to the task. Please summarize your findings clearly and concisely.\n"
        "If you have any additional context or relevant details, include them as well.\n"
        "Your answer should be actionable and directly address the user's request."
    ),
    "post_messages": "Please provide the final answer for the following task: {task}"
}

agent = CodeAgent(
    model=model,
    tools=[final_answer, image_generation_tool, visit_webpage_tool, web_search_tool, my_custom_tool],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()