"""
This moduel has get_prompt_and_model and send_data functions to be used by other modules.
"""

import logging
import requests

def get_prompt_and_model(args, kwargs):
    """
    Get prompt and model from arguments and keyword arguments.

    Args:
        args (list): List of arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        tuple: Tuple containing prompt and model.
    """

    prompt = (
        args[1] if args and len(args) > 1 and isinstance(args[1], str)
        else kwargs.get('prompt',
                        kwargs.get('input',
                        kwargs.get('messages', [{"content": None}])[0]['content'])
        )
    )

    model = kwargs.get('model')

    return prompt, model

def send_data(data, doku_url, doku_token):
    """
    Send data to the specified Doku URL.

    Args:
        data (dict): Data to be sent.
        api_url (str): URL of the API endpoint.
        auth_token (str): Authentication token.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the request.
    """

    try:
        headers = {
            'Authorization': doku_token,
            'Content-Type': 'application/json',
        }

        response = requests.post(doku_url.rstrip("/") + "/data",
                                 json=data,
                                 headers=headers,
                                 timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as req_err:
        logging.error("Error sending data to Doku: %s", req_err)
        raise  # Re-raise the exception after logging
