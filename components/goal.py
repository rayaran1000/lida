import json
import time
import logging
from local_packages.lida.utils import clean_code_snippet
from llmx import TextGenerator
from local_packages.lida.datamodel import Goal, TextGenerationConfig, Persona


SYSTEM_INSTRUCTIONS = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")


class GoalExplorer:
    """Generate goals given a summary of data."""

    def __init__(self) -> None:
        pass

    def generate(self, summary: dict, textgen_config: TextGenerationConfig,
                 text_gen: TextGenerator, n=5, persona: Persona = None, retries=25, retry_delay=1) -> list[Goal]:
        """
        Generate goals given a summary of data with retry logic in case of errors.

        Args:
            summary (dict): Summary of data.
            textgen_config (TextGenerationConfig): Text generation configuration.
            text_gen (TextGenerator): Text generator.
            n (int): Number of goals to generate.
            persona (Persona): Persona information.
            retries (int): Number of retry attempts. Defaults to 3.
            retry_delay (int): Delay between retries in seconds. Defaults to 2.

        Returns:
            list[Goal]: List of generated goals.
        """
        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n.
        {summary} \n\n"""

        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona, who is interested in complex, insightful goals about the data. \n"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant",
             "content":
             f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n "}
        ]

        for attempt in range(retries):
            result = text_gen.generate(messages=messages, config=textgen_config)
            try:
                # Attempt to parse the result as JSON
                json_string = clean_code_snippet(result.text[0]["content"])
                result = json.loads(json_string)
                # Cast each item in the list to a Goal object
                if isinstance(result, dict):
                    result = [result]
                return [Goal(**x) for x in result]  # Return parsed goals
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Attempt {attempt + 1}: Failed to decode JSON. Error: {e}")
                print(f"Attempt {attempt + 1}: Failed to decode JSON. Retrying...")
                time.sleep(retry_delay)  # Wait before retrying

        # If all retries fail, raise an exception
        logger.error("All retry attempts failed to generate valid JSON.")
        raise ValueError(
            "The model did not return a valid JSON object while attempting to generate goals. Consider using a larger model or a model with higher max token length."
        )