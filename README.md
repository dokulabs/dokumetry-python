# Doku Python SDK - dokumetry

[![Doku Python Package](https://img.shields.io/badge/Doku-orange)](https://github.com/dokulabs/doku)
[![Library Version](https://img.shields.io/github/tag/dokulabs/python-sdk.svg?&label=Library%20Version&logo=pypi)](https://github.com/dokulabs/python-sdk/tags)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/dokulabs/python-sdk)](https://github.com/dokulabs/python-sdk/pulse)
[![GitHub Contributors](https://img.shields.io/github/contributors/dokulabs/python-sdk)](https://github.com/dokulabs/python-sdk/graphs/contributors)

[![Tests](https://github.com/dokulabs/python-sdk/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/dokulabs/python-sdk/actions/workflows/tests.yml)
[![Pylint](https://github.com/dokulabs/python-sdk/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/dokulabs/python-sdk/actions/workflows/pylint.yml)
[![CodeQL](https://github.com/dokulabs/python-sdk/actions/workflows/github-code-scanning/codeql/badge.svg?branch=main)](https://github.com/dokulabs/python-sdk/actions/workflows/github-code-scanning/codeql)

Doku Python SDK (`dokumetry`) empowers you to effortlessly track and monitor language learning model (LLM) usage data and metrics from your Python code. It seamlessly integrates with major LLM Platforms:

 - ✅ OpenAI
 - ✅ Anthropic
 - ✅ Cohere

All LLM observability usage data is sent directly to the Doku Platform for streamlined tracking. Get started with Doku Python SDK for simplified and effective observability.

## Features

- **User-friendly UI Logs:** Log all your LLM requests in just two lines of code.

- **Cost and Latency Tracking:** Track costs and latencies based on users and custom properties for better analysis.

- **Prompt and Response Feedback:** Iterate on prompts and chat conversations directly in the UI.

- **Collaboration and Sharing:** Share results and collaborate with friends or teammates for more effective teamwork.

- **Very Low Latency Impact** We know latency of your Large-Language Model usage is important to your application's success, that's why we designed Doku SDKs to impact latency as little as possible.

## Installation

```bash
pip install dokumetry
```

## Quick Start ⚡️

### OpenAI

```
from openai import OpenAI
import dokumetry

client = OpenAI(
    api_key="YOUR_OPENAI_KEY"
)

# Pass the above `client` object along with your DOKU URL and Token and this will make sure that all OpenAI calls are automatically tracked.
dokumetry.init(client, doku_url="YOUR_DOKU_URL", token="YOUR_DOKU_TOKEN")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is LLM Observability",
        }
    ],
    model="gpt-3.5-turbo",
)
```

### Anthropic

```
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import dokumetry

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my api key",
)

# Pass the above `anthropic` object along with your DOKU URL and Token and this will make sure that all Anthropic calls are automatically tracked.
dokumetry.init(anthropic, doku_url="YOUR_DOKU_URL", token="YOUR_DOKU_TOKEN")

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} What is LLM Observability?{AI_PROMPT}",
)
print(completion.completion)
```

### Cohere

```
import cohere
import dokumetry

# initialize the Cohere Client with an API Key
co = cohere.Client('YOUR_API_KEY')

# Pass the above `co` object along with your DOKU URL and Token and this will make sure that all Cohere calls are automatically tracked.
dokumetry.init(co, doku_url="YOUR_DOKU_URL", token="YOUR_DOKU_TOKEN")

# generate a prediction for a prompt
prediction = co.chat(message='What is LLM Observability?', model='command')

# print the predicted text
print(f'Chatbot: {prediction.text}')
```

## Supported Parameters

| Parameter         | Description                                               | Required      |
|-------------------|-----------------------------------------------------------|---------------|
| llm              | Language Learning Model (LLM) Object to track             | Yes           |
| doku_url          | URL of your Doku Instance                                 | Yes           |
| token             | Your Doku Token                                           | Yes           |
| environment       | Custom environment tag to include in your metrics         | Optional      |
| application_name  | Custom application name tag for your metrics              | Optional      |
| skip_resp         | Skip response from the Doku Ingester for faster execution | Optional      |


## Semantic Versioning
This package generally follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions, though certain backwards-incompatible changes may be released as minor versions:

Changes that only affect static types, without breaking runtime behavior.
Changes to library internals which are technically public but not intended or documented for external use. (Please open a GitHub issue to let us know if you are relying on such internals).
Changes that we do not expect to impact the vast majority of users in practice.
We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

## Requirements
Python >= 3.7 is supported.

If you are interested in other runtime environments, please open or upvote an issue on GitHub.

## Security

Doku Python Library (`dokumetry`) sends the observability data over HTTP/HTTPS to the Doku Ingester which uses key based authentication mechanism to ensure the security of your data. Be sure to keep your API keys confidential and manage permissions diligently. Refer to our [Security Policy](SECURITY)

## Contributing

We welcome contributions to the Doku Python Library (`dokumetry`) project. Please refer to [CONTRIBUTING](CONTRIBUTING) for detailed guidelines on how you can participate.

## License

Doku Python Library (`dokumetry`) is available under the [GPL-3.0](LICENSE) License.

## Support

For support, issues, or feature requests, submit an issue through the [GitHub issues](https://github.com/dokulabs/python-sdk/issues) associated with this repository.
