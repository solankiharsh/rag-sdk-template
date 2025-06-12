"""
Ragsdk Core Example: OpenTelemetry

This example demonstrates how to collect traces and metrics using Ragsdk audit module.
We run the LLM generation several times to collect telemetry data, and then export it to the OpenTelementry collector and visualize it in Grafana.

The script exports traces to the local OTLP collector running on http://localhost:4317.
The recommended way to run it is using the official Docker image:

    ```bash
    docker run \
        --mount type=bind,src=./examples/core/audit/config/grafana/ragsdk-dashboard.json,dst=/otel-lgtm/ragsdk-dashboard.json \
        --mount type=bind,src=./examples/core/audit/config/grafana/grafana-dashboards.yaml,dst=/otel-lgtm/grafana/conf/provisioning/dashboards/grafana-dashboards.yaml \
        -p 3000:3000 -p 4317:4317 -p 4318:4318 --rm -ti grafana/otel-lgtm
    ```

To run the script, execute the following command:

    ```bash
    uv run examples/core/audit/otel.py
    ```

To visualize the metrics collected by Ragsdk, follow these steps:

    1. Open your browser and navigate to http://localhost:3000.
    2. To check collected metrics, go to the Dashboards section and select Ragsdk (make sure auto refresh is enabled).
    3. To check collected traces, go to the Drilldown/Traces section.
"""  # noqa: E501

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ragsdk-core[otel]",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-grpc",
#     "tqdm",
# ]
# ///
import asyncio
import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv  # type: ignore
from opentelemetry import metrics, trace  # type: ignore
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,  # type: ignore
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore
from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
from pydantic import BaseModel  # type: ignore
from tqdm.asyncio import tqdm  # type: ignore

from ragsdk.core.audit import set_metric_handlers, set_trace_handlers, traceable  # type: ignore
from ragsdk.core.llms import LiteLLM  # type: ignore
from ragsdk.core.prompt import Prompt  # type: ignore

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")


resource = Resource({SERVICE_NAME: "ragsdk-example"})

# Otel tracer provider setup
span_exporter = OTLPSpanExporter("http://localhost:4317", insecure=True)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter, max_export_batch_size=1))
trace.set_tracer_provider(tracer_provider)

# Otel meter provider setup
metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4317", insecure=True)
reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=1000)
meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
metrics.set_meter_provider(meter_provider)

# ragsdk observability setup
set_trace_handlers("otel")
set_metric_handlers("otel")


class PhilosopherPromptInput(BaseModel):  # noqa: D101
    philosopher_type: str
    question: str


class PromptOutput(BaseModel):  # noqa: D101
    answer: str


class PhilosopherPrompt(Prompt[PhilosopherPromptInput, PromptOutput]):  # noqa: D101
    system_prompt = "You are an ancient {{ philosopher_type }} philosopher."
    user_prompt = "Question: {{ question }}"

class AssistantPromptInput(BaseModel):  # noqa: D101
    knowledge: list[str]
    question: str

class AssistantPrompt(Prompt[AssistantPromptInput, PromptOutput]):  # noqa: D101
    system_prompt = "Answer based on the knowledge provided."
    user_prompt = """
    Question: {{ question }}

    Knowledge:
    {% for item in knowledge %}{{ item }}
    {% endfor %}
    """


@traceable
async def process_request() -> None:  # noqa: D101
    print("▶ process_request: start")
    question = "What's the meaning of life?"
    philosophers = [
        LiteLLM(model_name="gpt-4.1-2025-04-14", api_key=api_key, use_structured_output=True),
        LiteLLM(model_name="gpt-4.1-2025-04-14", api_key=api_key, use_structured_output=True),
        LiteLLM(model_name="gpt-4.1-2025-04-14", api_key=api_key, use_structured_output=True),
    ]
    prompts = []
    for p_type in ["nihilist", "stoic", "existentialist"]:
        print(f"  • building prompt for philosopher={p_type}")
        prompts.append(
            PhilosopherPrompt(
                PhilosopherPromptInput(question=question, philosopher_type=p_type)
            )
        )

    print("  • calling philosophers concurrently...")
    responses = await asyncio.gather(
        *[llm.generate(prompt) for llm, prompt in zip(philosophers, prompts, strict=False)]
    )
    print("  • philosopher responses received:")
    for i, resp in enumerate(responses, 1):
        print(f"    #{i}: {resp.answer[:60]}{'…' if len(resp.answer)>60 else ''}")

    print("  • preparing assistant prompt")
    assistant = LiteLLM(model_name="o3", api_key=api_key, use_structured_output=True)
    assistant_prompt = AssistantPrompt(
        AssistantPromptInput(
            question=question,
            knowledge=[r.answer for r in responses],
        )
    )
    print("  • streaming assistant response:")
    async for chunk in assistant.generate_streaming(assistant_prompt):
        print(f"    [chunk] {chunk}")
    print("◀ process_request: end\n")


async def main() -> None:  # noqa: D101
    iteration = 0

    async def run() -> AsyncGenerator[None, None]:
        nonlocal iteration
        while True:
            iteration += 1
            print(f"=== main: starting iteration {iteration} ===")
            await process_request()
            print(f"=== main: finished iteration {iteration} ===\n")
            yield

    # wrap with tqdm to see timing
    async for _ in tqdm(run(), desc="overall loop"):
        # you can also break after a few iterations if you like:
        if iteration >= 5:
            print("Reached 5 iterations, stopping.")
            break

if __name__ == "__main__":
    asyncio.run(main())
