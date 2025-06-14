# RagSDK CLI

RagSDK CLI provides the `ragsdk` command-line interface (CLI) tool that allows you to interact with RagSDK from the terminal. Other packages can extend the CLI by adding their own commands, so the exact set of available commands may vary depending on the installed packages.

## Installation

To use the complete RagSDK stack, install the `ragsdk` package:

```sh
pip install ragsdk
```

## Example Usage
The following example assumes that `ragsdk-core` is installed and that the current ddirectory contains a `song_prompt.py` file with a prompt class named `SongPrompt`, as defined in the [Quickstart Guide](https://github.com/solankiharsh/rag-sdk-50-days-of-learningquickstart/quickstart1_prompts/#making-the-prompt-dynamic).

The example demonstrates how to execute the prompt using the `ragsdk` CLI tool.
The left side of the table shows the system and user prompts (rendered with placeholders replaced by the provided values), and the right side shows the generated response from the Large Language Model.

```sh
$ ragsdk prompts exec song_prompt:SongPrompt --payload '{"subject": "unicorns", "age_group": 12, "genre": "pop"}'

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Question                              ┃ Answer                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ [{'role': 'system', 'content': 'You   │ (Verse 1)                             │
│ are a professional songwriter.        │ In a land of rainbows and glitter,    │
│ You only use language that is         │ Where flowers bloom and skies are     │
│ appropriate for children.'},          │ brighter,                             │
│ {'role': 'user', 'content': 'Write a  │ There's a magical creature so rare,   │
│ song about a unicorns for 12 years    │ With a horn that sparkles in the air. │
│ old pop fans.'}]                      │                                       │
│                                       │ (Chorus)                              │
│                                       │ Unicorns, unicorns, oh so divine,     │
│                                       │ With their mane that shines and eyes  │
│                                       │ that shine,                           │
│                                       │ Gallop through the meadows, so free,  │
│                                       │ In a world of wonder, just you and    │
│                                       │ me.                                   │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

## Documentation
* [Documentation of the `ragsdk` CLI](https://github.com/solankiharsh/rag-sdk-50-days-of-learningcli/main/)
