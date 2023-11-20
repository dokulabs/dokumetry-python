"""
__init__ module for dokulabs package.
"""

from .openai import init as init_openai
from .anthropic import init as init_anthropic
from .cohere import init as init_cohere

# pylint: disable=too-few-public-methods
class DokuConfig:
    """
    Configuration class for Doku initialization.
    """

    func = None
    doku_url = None
    token = None
    environment = None
    application_name = None

def init(func, doku_url, token, environment="default", application_name="default"):
    """
    Initialize Doku configuration based on the provided function.

    Args:
        func: The function to determine the platform (OpenAI, Cohere, Anthropic).
        doku_url (str): Doku URL.
        token (str): Doku Authentication token.
    """

    DokuConfig.func = func
    DokuConfig.doku_url = doku_url
    DokuConfig.token = token
    DokuConfig.environment = environment
    DokuConfig.application_name = application_name

    # pylint: disable=no-else-return, line-too-long
    if hasattr(func.chat, 'completions') and callable(func.chat.completions.create) and ('.openai.azure.com/' not in str(func.base_url)):
        init_openai(func, doku_url, token, environment, application_name)
        return
    # pylint: disable=no-else-return
    elif hasattr(func, 'generate') and callable(func.generate):
        init_cohere(func, doku_url, token, environment, application_name)
        return
    elif hasattr(func, 'count_tokens') and callable(func.count_tokens):
        init_anthropic(func, doku_url, token, environment, application_name)
        return
