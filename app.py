from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool
from ollama_model import OllamaModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool


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
        timezone: A string representing a valid timezone (e.g., 'America/New_York', 'kolkata', 'asia', etc.).
    Handles common aliases and fuzzy matches.
    """
    import difflib
    import pytz

    # Normalize input
    tz_input = timezone.strip().replace(" ", "_").lower()

    # Expanded alias map (add more as needed)
    aliases = {
        "india": "Asia/Kolkata",
        "kolkata": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "usa": "America/New_York",
        "new york": "America/New_York",
        "london": "Europe/London",
        "england": "Europe/London",
        "uk": "Europe/London",
        "paris": "Europe/Paris",
        "france": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "germany": "Europe/Berlin",
        "china": "Asia/Shanghai",
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "hong kong": "Asia/Hong_Kong",
        "singapore": "Asia/Singapore",
        "sydney": "Australia/Sydney",
        "australia": "Australia/Sydney",
        "moscow": "Europe/Moscow",
        "russia": "Europe/Moscow",
        "brazil": "America/Sao_Paulo",
        "sao paulo": "America/Sao_Paulo",
        "canada": "America/Toronto",
        "toronto": "America/Toronto",
        "vancouver": "America/Vancouver",
        "tokyo": "Asia/Tokyo",
        "japan": "Asia/Tokyo",
        "osaka": "Asia/Tokyo",
        "kyoto": "Asia/Tokyo",
        "asia": "Asia/Kolkata",  # Default Asia to Kolkata, can prompt user to clarify
        "europe": "Europe/London", # Default Europe to London
        # Add more as needed
    }

    # Try alias map first
    if tz_input in aliases:
        tz_name = aliases[tz_input]
    else:
        # Try exact match in pytz
        all_timezones = [tz.lower() for tz in pytz.all_timezones]
        if tz_input in all_timezones:
            tz_name = pytz.all_timezones[all_timezones.index(tz_input)]
        else:
            # Try partial/fuzzy match
            candidates = [tz for tz in pytz.all_timezones if tz_input in tz.lower()]
            if not candidates:
                # Use difflib for closest match
                close = difflib.get_close_matches(tz_input, all_timezones, n=1, cutoff=0.7)
                if close:
                    tz_name = pytz.all_timezones[all_timezones.index(close[0])]
                else:
                    # Give helpful error
                    sample_tzs = ', '.join(pytz.all_timezones[:10])
                    return (
                        f"Error: Unknown timezone '{timezone}'. "
                        f"Try a valid timezone string like 'Asia/Kolkata', 'America/New_York', etc. "
                        f"Examples: {sample_tzs}"
                    )
            else:
                tz_name = candidates[0]
    try:
        tz = pytz.timezone(tz_name)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {tz_name} is: {local_time}"
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

