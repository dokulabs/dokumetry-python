"""
Module for monitoring Anthropic API calls.
"""

import time
from .__helpers import send_data

def init(func, doku_url, token, environment, applicationName):
    """
    Initialize Anthropic integration with Doku.

    Args:
        func: The Anthropic function to be patched.
        doku_url (str): Doku URL.
        token (str): Authentication token.
    """

    original_completions_create = func.completions.create

    def patched_completions_create(*args, **kwargs):
        """
        Patched version of Anthropic's completions.create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            AnthropicResponse: The response from Anthropic's completions.create.
        """

        start_time = time.time()
        response = original_completions_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        model = kwargs.get('model') if 'model' in kwargs else args[0]
        prompt = kwargs.get('prompt') if 'prompt' in kwargs else args[2]

        prompt_tokens = func.count_tokens(prompt)
        completion_tokens = func.count_tokens(response.completion)

        data = {
                "environment": environment,
                "applicationName": applicationName,
                "sourceLanguage": "python",
                "endpoint": "anthropic.completions",
                "completionTokens": completion_tokens,
                "promptTokens": prompt_tokens,
                "requestDuration": duration,
                "model": model,
                "prompt": prompt,
                "finishReason": response.stop_reason,
                "response": response.completion
        }

        send_data(data, doku_url, token)

        return response

    func.completions.create = patched_completions_create
