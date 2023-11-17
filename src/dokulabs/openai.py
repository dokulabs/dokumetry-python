"""
Module for monitoring OpenAI API calls.
"""

import time
from .__helpers import send_data , get_prompt_and_model

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def init(func, doku_url, token):
    """
    Initialize OpenAI monitoring for Doku.

    Args:
        func: The OpenAI function to be patched.
        doku_url (str): Doku URL.
        token (str): Doku Authentication token.
    """

    original_chat_create = func.chat.completions.create
    original_completions_create = func.completions.create
    original_embeddings_create = func.embeddings.create
    original_fine_tuning_jobs_create = func.fine_tuning.jobs.create
    original_images_create = func.images.generate
    original_images_create_variation = func.images.create_variation
    original_audio_speech_create = func.audio.speech.create

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
        prompt, model = get_prompt_and_model(args, kwargs)

        data = {
            "source": "python",
            "endpoint": "openai.chat.completions",
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
        }

        if "stream" not in kwargs:
            data["completionTokens"] = response.usage.completion_tokens
            data["promptTokens"] = response.usage.prompt_tokens
            data["totalTokens"] = response.usage.total_tokens
            data["finishReason"] = response.choices[0].finish_reason
            data["response"] = response.choices[0].message.content

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
        prompt, model = get_prompt_and_model(args, kwargs)

        data = {
            "source": "python",
            "endpoint": "openai.completions",
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
        }

        if "stream" not in kwargs:
            data["completionTokens"] = response.usage.completion_tokens
            data["promptTokens"] = response.usage.prompt_tokens
            data["totalTokens"] = response.usage.total_tokens
            data["finishReason"] = response.choices[0].finish_reason
            data["response"] = response.choices[0].text

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
        prompt, model = get_prompt_and_model(args, kwargs)

        data = {
            "source": "python",
            "endpoint": "openai.emdeddings",
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
        _ , model = get_prompt_and_model(args, kwargs)

        data = {
            "source": "python",
            "endpoint": "openai.fine_tuning",
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
        prompt , model = get_prompt_and_model(args, kwargs)
        size = kwargs.get('size', '10324x1024') if 'size' not in kwargs else kwargs['size']

        if model is None:
            model = "dall-e-2"

        if 'response_format' in kwargs and kwargs['response_format'] == 'b64_json':
            image = "b64_json"
        else:
            image = "url"

        for items in response.data:
            data = {
                "source": "python",
                "endpoint": "openai.images.create",
                "requestDuration": duration,
                "model": model,
                "prompt": prompt,
                "imageSize": size,
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
        _ , model = get_prompt_and_model(args, kwargs)
        size = kwargs.get('size', '10324x1024') if 'size' not in kwargs else kwargs['size']

        if model is None:
            model = "dall-e-2"

        if 'response_format' in kwargs and kwargs['response_format'] == 'b64_json':
            image = "b64_json"
        else:
            image = "url"

        for items in response.data:

            data = {
                "source": "python",
                "endpoint": "openai.images.create.variations",
                "requestDuration": duration,
                "model": model,
                "imageSize": size,
                "revisedPrompt": items.revised_prompt,
                "image": getattr(items, image)
            }

            send_data(data, doku_url, token)

        return response

    def patched_audio_speech_variation(*args, **kwargs):
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
        prompt , model = get_prompt_and_model(args, kwargs)
        voice = kwargs.get('voice')

        data = {
            "source": "python",
            "endpoint": "openai.audio.speech.create",
            "requestDuration": duration,
            "model": model,
            "prompt": prompt,
            "audioVoice": voice,
            "promptTokens": len(prompt)
        }

        send_data(data, doku_url, token)

        return response

    func.chat.completions.create = patched_chat_create
    func.completions.create = patched_completions_create
    func.embeddings.create = patched_embeddings_create
    func.fine_tuning.jobs.create = patched_fine_tuning_create
    func.images.generate = patched_image_create
    func.images.create_variation = patched_image_create_variation
    func.audio.speech.create = patched_audio_speech_variation
