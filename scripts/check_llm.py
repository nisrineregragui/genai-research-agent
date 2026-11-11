"""Smoke test for the LLM proxy.

Verifies every capability the agent relies on: embeddings, chat,
tool calling and structured (JSON schema) output.

Run from the project root:  python -m scripts.check_llm
"""

from __future__ import annotations

import json
import sys

from app.config import Settings, load_settings
from app.llm import get_client


def check_embeddings(client, settings: Settings) -> str:
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=["warfarin", "apixaban"],
    )
    dims = len(resp.data[0].embedding)
    return f"{len(resp.data)} vectors, {dims} dimensions each"


def check_chat(client, settings: Settings) -> str:
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "In one sentence, what is a contraindication?"}],
    )
    return resp.choices[0].message.content.strip()


def check_tool_calling(client, settings: Settings) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_glossary",
                "description": "Look up the definition of a pharmacology term.",
                "parameters": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                },
            },
        }
    ]
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "What does INR mean? Use the glossary tool."}],
        tools=tools,
    )
    calls = resp.choices[0].message.tool_calls
    if not calls:
        raise RuntimeError("The model answered without calling the tool.")
    call = calls[0].function
    return f"{call.name}({call.arguments})"


def check_structured_output(client, settings: Settings) -> str:
    schema = {
        "type": "object",
        "properties": {
            "drug": {"type": "string"},
            "is_anticoagulant": {"type": "boolean"},
        },
        "required": ["drug", "is_anticoagulant"],
        "additionalProperties": False,
    }
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "Is warfarin an anticoagulant?"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "drug_check", "strict": True, "schema": schema},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    return json.dumps(data)


def main() -> int:
    settings = load_settings()
    print(f"Chat model:      {settings.chat_model}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Base URL:        {settings.openai_base_url or '(OpenAI default)'}\n")

    client = get_client()
    checks = [
        ("Embeddings", check_embeddings),
        ("Chat", check_chat),
        ("Tool calling", check_tool_calling),
        ("Structured output", check_structured_output),
    ]

    failures = 0
    for name, check in checks:
        try:
            print(f"[PASS] {name}: {check(client, settings)}")
        except Exception as exc:  # report every failure instead of stopping at the first one
            failures += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())