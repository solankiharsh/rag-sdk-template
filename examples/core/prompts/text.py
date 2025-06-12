"""
Ragsdk Core Example: Text Prompt

This example shows how to use the `Prompt` class to generate themed text using an LLM.
We define an `AnimalPrompt` that generates names for a given animal type.

The script performs the following steps:

    1. Define input and output formats using Pydantic models.
    2. Implement the `AnimalPrompt` class with a structured system prompt.
    3. Initialize the `LiteLLM` class to generate text.
    4. Generate a name based on the specified animal.
    5. Print the generated name.

To run the script, execute the following command:

    ```bash
    uv run examples/core/prompt/text.py
    ```
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ragsdk-core",
# ]
# ///
import asyncio
import os

from dotenv import load_dotenv  # type: ignore
from pydantic import BaseModel  # type: ignore

from ragsdk.core.llms import LiteLLM  # type: ignore
from ragsdk.core.prompt import Prompt  # type: ignore

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")


class AnimalPromptInput(BaseModel):
    """
    Input format for the AnimalPrompt.
    """

    animal: str


class AnimalPromptOutput(BaseModel):
    """
    Output format for the AnimalPrompt.
    """

    name: str


class AnimalPrompt(Prompt[AnimalPromptInput, AnimalPromptOutput]):
    """
    Prompt that generates animal names.
    """

    system_prompt = """
    You are an animal name generator. Use provided animal kind as a base.
    """

    user_prompt = """
    Animal: {{ animal }}
    """


async def main() -> None:
    """
    Run the example.
    """
    llm = LiteLLM(model_name="gpt-4o-2024-08-06", use_structured_output=True)
    prompt = AnimalPrompt(AnimalPromptInput(animal="cat"))
    response = await llm.generate(prompt)
    print(response.name)


if __name__ == "__main__":
    asyncio.run(main())
