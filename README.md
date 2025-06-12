<div align="center">

# Rag at Scale
**Building Blocks for Rapid GenAI Applications**

[🔗 Homepage](https://github.com/solankiharsh) • [📖 Docs](https://github.com/solankiharsh) • [✉️ Contact](mailto:hvsolanki27@gmail.com)

[![License](https://img.shields.io/pypi/l/ragsdk)](https://pypi.org/project/ragsdk)
[![Version](https://img.shields.io/pypi/v/ragsdk)](https://pypi.org/project/ragsdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/ragsdk)](https://pypi.org/project/ragsdk)

</div>

---

## 🌟 Key Features

### 1. Core Capabilities
- **Swap LLMs at will**
  Access 100+ hosted models via LiteLLM or run local LLMs seamlessly.
- **Type-safe prompts**
  Leverage Python generics to enforce input/output schemas on all LLM calls.
- **Modular vector stores**
  Plug in Qdrant, PgVector, or any other store via the unified API.
- **CLI developer tools**
  Manage stores, inspect pipelines, and test prompts right from your terminal.

### 2. Fast & Flexible RAG
- **20+ ingestion formats**
  PDFs, HTML, spreadsheets, PPTs—and more via Docling or Unstructured.
- **Structured data handling**
  Tables, images, and layouts extracted with built-in VLM support.
- **Universal connectors**
  S3, GCS, Azure, or roll your own data source integration.
- **Scalable pipelines**
  Ray-powered parallel ingestion for massive datasets.

### 3. Reliable Deployment & Monitoring
- **Real-time observability**
  Integrated OpenTelemetry metrics and CLI dashboards.
- **Automated testing**
  Pre-deploy prompt validation with Promptfoo.
- **Continuous optimization**
  Auto-evaluate and refine model performance over time.
- **Chat UI boilerplate**
  Out-of-the-box chatbot interface with persistence and feedback loops.

---

## ⚙️ Installation

```bash
pip install ragsdk
```

This is a starter bundle of packages, containing:

- [`ragsdk-core`](https://github.com/solankiharsh/ragsdk/tree/main/packages/ragsdk-core) - fundamental tools for working with prompts, LLMs and vector databases.
- [`ragsdk-agents`](https://github.com/solankiharsh/ragsdk/tree/main/packages/ragsdk-agents) - abstractions for building agentic systems.
- [`ragsdk-document-search`](https://github.com/solankiharsh/ragsdk/ragsdk/tree/main/packages/ragsdk-document-search) - retrieval and ingestion piplines for knowledge bases.
- [`ragsdk-evaluate`](https://github.com/solankiharsh/ragsdk/tree/main/packages/ragsdk-evaluate) - unified evaluation framework for Rag components.
- [`ragsdk-chat`](https://github.com/solankiharsh/ragsdk/tree/main/packages/ragsdk-chat) - full-stack infrastructure for building conversational AI applications.
- [`ragsdk-cli`](https://github.com/solankiharsh/ragsdk/tree/main/packages/ragsdk-cli) - `ragsdk` shell command for interacting with R components.

Alternatively, you can use individual components of the stack by installing their respective packages.

## Quickstart

### Basics

To define a prompt and run LLM:

```python
import asyncio
from pydantic import BaseModel
from ragsdk.core.llms import LiteLLM
from ragsdk.core.prompt import Prompt

class QuestionAnswerPromptInput(BaseModel):
    question: str

class QuestionAnswerPromptOutput(BaseModel):
    answer: str

class QuestionAnswerPrompt(Prompt[QuestionAnswerPromptInput, QuestionAnswerPromptOutput]):
    system_prompt = """
    You are a question answering agent. Answer the question to the best of your ability.
    """
    user_prompt = """
    Question: {{ question }}
    """

llm = LiteLLM(model_name="gpt-4.1-nano", use_structured_output=True)

async def main() -> None:
    prompt = QuestionAnswerPrompt(QuestionAnswerPromptInput(question="What are high memory and low memory on linux?"))
    response = await llm.generate(prompt)
    print(response.answer)

if __name__ == "__main__":
    asyncio.run(main())
```

### Document Search

To build and query a simple vector store index:

```python
import asyncio
from ragsdk.core.embeddings import LiteLLMEmbedder
from ragsdk.core.vector_stores import InMemoryVectorStore
from ragsdk.document_search import DocumentSearch

embedder = LiteLLMEmbedder(model_name="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedder=embedder)
document_search = DocumentSearch(vector_store=vector_store)

async def run() -> None:
    await document_search.ingest("web://https://arxiv.org/pdf/1706.03762")
    result = await document_search.search("What are the key findings presented in this paper?")
    print(result)

if __name__ == "__main__":
    asyncio.run(run())
```

### RAG Pipeline

To build a simple RAG pipeline:

```python
import asyncio
from pydantic import BaseModel
from ragsdk.core.embeddings import LiteLLMEmbedder
from ragsdk.core.llms import LiteLLM
from ragsdk.core.prompt import Prompt
from ragsdk.core.vector_stores import InMemoryVectorStore
from ragsdk.document_search import DocumentSearch

class QuestionAnswerPromptInput(BaseModel):
    question: str
    context: list[str]

class QuestionAnswerPromptOutput(BaseModel):
    answer: str

class QuestionAnswerPrompt(Prompt[QuestionAnswerPromptInput, QuestionAnswerPromptOutput]):
    system_prompt = """
    You are a question answering agent. Answer the question that will be provided using context.
    If in the given context there is not enough information refuse to answer.
    """
    user_prompt = """
    Question: {{ question }}
    Context: {% for item in context %}
        {{ item }}
    {%- endfor %}
    """

embedder = LiteLLMEmbedder(model_name="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedder=embedder)
document_search = DocumentSearch(vector_store=vector_store)
llm = LiteLLM(model_name="gpt-4.1-nano", use_structured_output=True)

async def run() -> None:
    question = "What are the key findings presented in this paper?"

    await document_search.ingest("web://https://arxiv.org/pdf/1706.03762")
    result = await document_search.search(question)

    prompt = QuestionAnswerPrompt(QuestionAnswerPromptInput(
        question=question,
        context=[element.text_representation for element in result],
    ))
    response = await llm.generate(prompt)
    print(response.answer)

if __name__ == "__main__":
    asyncio.run(run())
```

### Chatbot interface with UI

To expose your RAG application through Ragsdk UI:

```python
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from ragsdk.chat.api import RagsdkAPI
from ragsdk.chat.interface import ChatInterface
from ragsdk.chat.interface.types import ChatContext, ChatResponse
from ragsdk.core.embeddings import LiteLLMEmbedder
from ragsdk.core.llms import LiteLLM
from ragsdk.core.prompt import Prompt
from ragsdk.core.prompt.base import ChatFormat
from ragsdk.core.vector_stores import InMemoryVectorStore
from ragsdk.document_search import DocumentSearch


class QuestionAnswerPromptInput(BaseModel):
    question: str
    context: list[str]


class QuestionAnswerPrompt(Prompt[QuestionAnswerPromptInput, str]):
    system_prompt = """
    You are a question answering agent. Answer the question that will be provided using context.
    If in the given context there is not enough information refuse to answer.
    """
    user_prompt = """
    Question: {{ question }}
    Context: {% for item in context %}{{ item }}{%- endfor %}
    """


class MyChat(ChatInterface):
    """Chat interface for fullapp application."""

    async def setup(self) -> None:
        self.embedder = LiteLLMEmbedder(model_name="text-embedding-3-small")
        self.vector_store = InMemoryVectorStore(embedder=self.embedder)
        self.document_search = DocumentSearch(vector_store=self.vector_store)
        self.llm = LiteLLM(model_name="gpt-4.1-nano", use_structured_output=True)

        await self.document_search.ingest("web://https://arxiv.org/pdf/1706.03762")

    async def chat(
        self,
        message: str,
        history: ChatFormat | None = None,
        context: ChatContext | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        # Search for relevant documents
        result = await self.document_search.search(message)

        prompt = QuestionAnswerPrompt(
            QuestionAnswerPromptInput(
                question=message,
                context=[element.text_representation for element in result],
            )
        )

        # Stream the response from the LLM
        async for chunk in self.llm.generate_streaming(prompt):
            yield self.create_text_response(chunk)


if __name__ == "__main__":
    RagsdkAPI(MyChat).run()
```