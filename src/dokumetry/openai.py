"""
Module for monitoring OpenAI API calls.
"""

import time
from .__helpers import send_data

# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
def init(llm, doku_url, token, environment, application_name, skip_resp):
    """
    Initialize OpenAI monitoring for Doku.

    Args:
        llm: The OpenAI function to be patched.
        doku_url (str): Doku URL.
        token (str): Doku Authentication token.
        environment (str): Doku environment.
        application_name (str): Doku application name.
        skip_resp (bool): Skip response processing.
    """

    original_chat_create = llm.chat.completions.create
    original_completions_create = llm.completions.create
    original_embeddings_create = llm.embeddings.create
    original_fine_tuning_jobs_create = llm.fine_tuning.jobs.create
    original_images_create = llm.images.generate
    original_images_create_variation = llm.images.create_variation
    original_audio_speech_create = llm.audio.speech.create

    def patched_chat_create(*args, **kwargs):
        """
        Patched version of OpenAI's chat completions create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's chat completions create method.
        """

        start_time = time.time()
        response = original_chat_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        message_prompt = kwargs.get('messages', "No prompt provided")
        formatted_messages = []

        for message in message_prompt:
            role = message["role"]
            content = message["content"]

            if isinstance(content, list):
                content_str = ", ".join(
                    f"{item['type']}: {item['text'] if 'text' in item else item['image_url']}"
                    if 'type' in item else f"text: {item['text']}"
                    for item in content
                )
                formatted_messages.append(f"{role}: {content_str}")
            else:
                formatted_messages.append(f"{role}: {content}")

        prompt = "\n".join(formatted_messages)

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "openai.chat.completions",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
        }

        if "stream" not in kwargs:
            data["completionTokens"] = response.usage.completion_tokens
            data["promptTokens"] = response.usage.prompt_tokens
            data["totalTokens"] = response.usage.total_tokens
            data["finishReason"] = response.choices[0].finish_reason

        if "tools" not in kwargs:
            if "n" not in kwargs or kwargs["n"] == 1:
                data["response"] = response.choices[0].message.content
            else:
                i = 0
                while i < kwargs["n"]:
                    data["response"] = response.choices[i].message.content
                    i += 1
                    send_data(data, doku_url, token)
                return response
        else:
            data["response"] = "Function called with tools"

        send_data(data, doku_url, token)

        return response

    def patched_completions_create(*args, **kwargs):
        """
        Patched version of OpenAI's completions create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's completions create method.
        """

        start_time = time.time()
        response = original_completions_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        prompt = kwargs.get('prompt', "No prompt provided")

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "openai.completions",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
        }

        if "stream" not in kwargs:
            data["completionTokens"] = response.usage.completion_tokens
            data["promptTokens"] = response.usage.prompt_tokens
            data["totalTokens"] = response.usage.total_tokens
            data["finishReason"] = response.choices[0].finish_reason

        if "tools" not in kwargs:
            if "n" not in kwargs or kwargs["n"] == 1:
                data["response"] = response.choices[0].text
            else:
                i = 0
                while i < kwargs["n"]:
                    data["response"] = response.choices[i].text
                    i += 1
                    send_data(data, doku_url, token)
                return response

        send_data(data, doku_url, token)

        return response

    def patched_embeddings_create(*args, **kwargs):
        """
        Patched version of OpenAI's embeddings create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's embeddings create method.
        """

        start_time = time.time()
        response = original_embeddings_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        prompt = kwargs.get('input', "No prompt provided")

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "openai.emdeddings",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
            "promptTokens": response.usage.prompt_tokens,
            "totalTokens": response.usage.total_tokens
        }

        send_data(data, doku_url, token)

        return response

    def patched_fine_tuning_create(*args, **kwargs):
        """
        Patched version of OpenAI's fine-tuning jobs create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's fine-tuning jobs create method.
        """

        start_time = time.time()
        response = original_fine_tuning_jobs_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "openai.fine_tuning",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "finetuneJobId": response.id,
            "finetuneJobStatus": response.status,
        }

        send_data(data, doku_url, token)

        return response

    def patched_image_create(*args, **kwargs):
        """
        Patched version of OpenAI's images generate method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's images generate method.
        """

        start_time = time.time()
        response = original_images_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        prompt = kwargs.get('prompt', "No prompt provided")
        size = kwargs.get('size', '10324x1024') if 'size' not in kwargs else kwargs['size']

        if model is None:
            model = "dall-e-2"

        if 'response_format' in kwargs and kwargs['response_format'] == 'b64_json':
            image = "b64_json"
        else:
            image = "url"

        if 'quality' not in kwargs:
            quality = "standard"
        else:
            quality = kwargs['quality']

        for items in response.data:
            data = {
                "environment": environment,
                "applicationName": application_name,
                "sourceLanguage": "python",
                "endpoint": "openai.images.create",
                "skipResp": skip_resp,
                "requestDuration": duration,
                "model": model,
                "prompt": prompt,
                "imageSize": size,
                "imageQuality": quality,
                "revisedPrompt": items.revised_prompt,
                "image": getattr(items, image)
            }

            send_data(data, doku_url, token)

        return response

    def patched_image_create_variation(*args, **kwargs):
        """
        Patched version of OpenAI's images create variation method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's images create variation method.
        """

        start_time = time.time()
        response = original_images_create_variation(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        size = kwargs.get('size', '10324x1024') if 'size' not in kwargs else kwargs['size']

        if model is None:
            model = "dall-e-2"

        if 'response_format' in kwargs and kwargs['response_format'] == 'b64_json':
            image = "b64_json"
        else:
            image = "url"

        for items in response.data:

            data = {
                "environment": environment,
                "applicationName": application_name,
                "sourceLanguage": "python",
                "endpoint": "openai.images.create.variations",
                "skipResp": skip_resp,
                "requestDuration": duration,
                "model": model,
                "imageSize": size,
                "imageQuality": "standard",
                "revisedPrompt": items.revised_prompt,
                "image": getattr(items, image)
            }

            send_data(data, doku_url, token)

        return response

    def patched_audio_speech_create(*args, **kwargs):
        """
        Patched version of OpenAI's audio speech create method.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            OpenAIResponse: The response from OpenAI's audio speech create method.
        """

        start_time = time.time()
        response = original_audio_speech_create(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        model = kwargs.get('model', "No Model provided")
        prompt = kwargs.get('input', "No prompt provided")
        voice = kwargs.get('voice')

        data = {
            "environment": environment,
            "applicationName": application_name,
            "sourceLanguage": "python",
            "endpoint": "openai.audio.speech.create",
            "skipResp": skip_resp,
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
            "audioVoice": voice,
            "promptTokens": len(prompt)
        }

        send_data(data, doku_url, token)

        return response

    llm.chat.completions.create = patched_chat_create
    llm.completions.create = patched_completions_create
    llm.embeddings.create = patched_embeddings_create
    llm.fine_tuning.jobs.create = patched_fine_tuning_create
    llm.images.generate = patched_image_create
    llm.images.create_variation = patched_image_create_variation
    llm.audio.speech.create = patched_audio_speech_create
