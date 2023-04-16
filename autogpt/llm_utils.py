import time
import requests
import json
from openai.error import APIError, RateLimitError
from colorama import Fore

from autogpt.config import Config

cfg = Config()

api_url = "http://localhost:3000/send-message"


def create_chat_completion(
    messages, model=None, temperature=cfg.temperature, max_tokens=None
) -> str:
    """Create a chat completion using the custom API"""
    response = None
    num_retries = 5
    headers = {"Content-Type": "application/json"}

    if cfg.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
            f" max_tokens {max_tokens}" + Fore.RESET
        )

    for attempt in range(num_retries):
        try:
            data = {"message": json.dumps(messages)}
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=120)

            if response.status_code == 200:
                break
            else:
                raise APIError("Non-200 status code received")
        except RateLimitError:
            if cfg.debug_mode:
                print(
                    Fore.RED + "Error: ",
                    "API Rate Limit Reached. Waiting 20 seconds..." + Fore.RESET,
                )
            time.sleep(20)
        except APIError as e:
            if e.http_status == 502:
                if cfg.debug_mode:
                    print(
                        Fore.RED + "Error: ",
                        "API Bad gateway. Waiting 20 seconds..." + Fore.RESET,
                    )
                time.sleep(20)
            else:
                raise
            if attempt == num_retries - 1:
                raise

    if response is None:
        raise RuntimeError("Failed to get response after 5 retries")

    response_json = response.json()
    return response_json["data"]
