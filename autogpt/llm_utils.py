from ast import List
import time
from typing import Dict, Optional

from colorama import Fore
import requests
import json
import spacy
from autogpt.config import Config

CFG = Config()


def call_ai_function(
    function: str, args: List, description: str, model: Optional[str] = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = CFG.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)

api_url = 'http://localhost:3030/send-message'

def create_chat_completion(
    messages: List,
    model: Optional[str] = None,
    temperature: float = CFG.temperature,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the custom API"""
    response = None
    num_retries = 5
    headers = {"Content-Type": "application/json"}

    if CFG.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
            f" max_tokens {max_tokens}" + Fore.RESET
        )

    # Combine messages into a single string
    message_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    for attempt in range(num_retries):
        try:
            data = {"message": message_str}
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=120)

            if response.status_code == 200:
                return response.json()["data"]
        except requests.exceptions.RequestException as e:
            if attempt == num_retries - 1:
                raise
            if CFG.debug_mode:
                print(
                    Fore.RED + "Error: ",
                    f"API request failed. Retrying in {2 ** (attempt + 2)} seconds..." + Fore.RESET,
                )
            time.sleep(2 ** (attempt + 2))


def create_embedding_with_spacy(text: str) -> list:
    """Create a text embedding using spacy's en_core_web_md model"""
    nlp = spacy.load("en_core_web_md")
    text = text.replace("\n", " ")
    doc = nlp(text)
    return doc.vector
