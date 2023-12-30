"""
Module for monitoring Cohere API calls.
"""

import time
from .__helpers import send_data

def count_tokens(text):
    """
    Count the number of tokens in the given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    tokens_per_word = 2.5

    # Split the text into words
    words = text.split()

    # Calculate the number of tokens
    num_tokens = round(len(words) * tokens_per_word)

    return num_tokens

# pylint: disable=too-many-arguments
def init(llm, doku_url, token, environment, application_name, skip_resp):
    """
    Initialize Cohere monitoring for Doku.

    Args:
        llm: The Cohere function to be patched.
        doku_url (str): Doku URL.
        token (str): Doku Authentication token.
        environment (str): Doku environment.
        application_name (str): Doku application name.
        skip_resp (bool): Skip response processing.
    """

    original_generate = llm.generate
    original_embed = llm.embed
    original_chat = llm.chat
    original_summarize = llm.summarize

    def patched_generate(*args, **kwargs):
        """
        Patched version of Cohere's generate method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            CohereResponse: The response from Cohere's generate method.
        """

        start_time = time.time()
        response = original_generate(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model') if 'model' in kwargs else "command"
        prompt = kwargs.get('prompt')

        for generation in response:
            data = {
                "environment": environment,
                "applicationName": application_name,
                "sourceLanguage": "python",
                "endpoint": "cohere.generate",
                "skipResp": skip_resp,
                "completionTokens": count_tokens(generation.text),
                "promptTokens": count_tokens(prompt),
                "requestDuration": duration,
                "model": model,
                "prompt": prompt,
                "response": generation.text,
            }

            if "stream" not in kwargs:
                data["finishReason"] = generation.finish_reason

            send_data(data, doku_url, token)

        return response

    def embeddings_generate(*args, **kwargs):
        """
        Patched version of Cohere's embeddings generate method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            CohereResponse: The response from Cohere's embeddings generate method.
        """

        start_time = time.time()
        response = original_embed(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model') if 'model' in kwargs else "embed-english-v2.0"
        prompt = ' '.join(kwargs.get('texts', []))

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "cohere.embed",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
        }

        send_data(data, doku_url, token)

        return response

    def chat_generate(*args, **kwargs):
        """
        Patched version of Cohere's chat generate method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            CohereResponse: The response from Cohere's chat generate method.
        """

        start_time = time.time()
        response = original_chat(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "command")
        prompt = kwargs.get('message')
        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "cohere.chat",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "prompt": prompt,
            "model": model,
        }
        if "stream" not in kwargs or kwargs["stream"] == False:
            data["completionTokens"] = response.meta["billed_units"]["output_tokens"]
            data["promptTokens"] = response.meta["billed_units"]["input_tokens"]
            data["totalTokens"] = response.token_count["billed_tokens"]
            data["response"] = response.text
        else:
            data["response"] = ""
            for event in response:
                if event.event_type == "text-generation":   
                    data["response"] += event.text
            data["promptTokens"] = count_tokens(prompt)
            data["completionTokens"] = count_tokens(data["response"])
            data["totalTokens"] = data["promptTokens"] + data["completionTokens"]

        send_data(data, doku_url, token)

        return response

    def summarize_generate(*args, **kwargs):
        """
        Patched version of Cohere's summarize generate method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            CohereResponse: The response from Cohere's summarize generate method.
        """

        start_time = time.time()
        response = original_summarize(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model') if 'model' in kwargs else "command"
        prompt = kwargs.get('text')

        if "stream" not in kwargs:

            data = {
                    "environment": environment,
                    "applicationName": application_name,
                    "sourceLanguage": "python",
                    "endpoint": "cohere.chat",
                    "skipResp": skip_resp,
                    "requestDuration": duration,
                    "completionTokens": count_tokens(response.summary),
                    "promptTokens": count_tokens(prompt),
                    "model": model,
                    "prompt": prompt,
                    "response": response.summary
            }

        send_data(data, doku_url, token)

        return response

    llm.generate = patched_generate
    llm.embed = embeddings_generate
    llm.chat = chat_generate
    llm.summarize = summarize_generate
