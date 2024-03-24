"""
Mistral Test Suite

This module contains a suite of tests for Mistral functionality 
using the Mistral Python library. It includes tests for various 
Mistral API endpoints such as text summarization, text generation 
with a prompt,text embeddings creation, and chat-based 
language understanding.

The tests are designed to cover different aspects of Mistral's 
capabilities and serve as a validation mechanism for the integration 
with the Doku monitoring system.

Global Mistral client and initialization are set up for the 
Mistral client and Doku monitoring.

Environment Variables:
    - Mistral_API_TOKEN: Mistral API api_key for authentication.
    - DOKU_URL: Doku URL for monitoring data submission.
    - DOKU_TOKEN: Doku authentication api_key.

Note: Ensure the environment variables are properly set before running the tests.
"""

import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import dokumetry

# Global Mistral client
client = MistralClient(
    api_key=os.getenv("MISTRAL_API_TOKEN")
)

# Global Mistral initialization
# pylint: disable=line-too-long
dokumetry.init(llm=client, doku_url=os.getenv("DOKU_URL"), api_key=os.getenv("DOKU_TOKEN"), environment="dokumetry-testing", application_name="dokumetry-python-test", skip_resp=False)

def test_chat():
    """
    Test the 'chat' function of the Mistral client.
    """
    messages = [
        ChatMessage(role="user", content="What is the best French cheese?")
    ]

    # No streaming
    message = client.chat(
        model="mistral-large-latest",
        messages=messages,
    )
    assert message.object == 'chat.completion'

def test_embeddings():
    """
    Test the 'embeddings' function of the Mistral client.
    """
    response = client.embeddings(
      model="mistral-embed",
      input=["Embed this sentence.", "As well as this one."],
    )
    assert response.object == 'list'
