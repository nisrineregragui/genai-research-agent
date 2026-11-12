"""FastAPI application.

Run locally:  python -m uvicorn app.api.main:app --port 8000

Endpoints:
- GET  /            minimal web interface (app/web/index.html)
- GET  /health      liveness check
- POST /ask         the full answer as JSON (the interface required by the assignment)
- POST /ask/stream  the same answer as Server-Sent Events: one "step" event per graph
                    node as it completes, then one "result" event (same payload as /ask)

The agent is built once at startup (corpus + embeddings). Tests inject a fake agent
through create_app(agent=...), so they never call the LLM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAIError

from app.agent import AgentResult, ResearchAgent
from app.api.schemas import AskRequest, AskResponse, to_response
from app.bootstrap import build_agent
from app.config import DEFAULT_MAX_QUESTION_CHARS, load_settings

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"
MODEL_ERROR = "The language model service failed. Please try again."
INTERNAL_ERROR = "Internal error while answering the question."


def create_app(agent: ResearchAgent | None = None, max_question_chars: int | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            settings = load_settings()
            logger.info("Building the agent (corpus, embeddings, graph)...")
            app.state.agent = build_agent(settings)
            app.state.max_question_chars = settings.max_question_chars
            logger.info("Agent ready")
        else:
            app.state.agent = agent
            app.state.max_question_chars = max_question_chars or DEFAULT_MAX_QUESTION_CHARS
        yield

    app = FastAPI(
        title="Pharmacology Research Agent",
        description="Agentic research assistant answering questions about a corpus of FDA drug labels. "
                    "Not medical advice.",
        lifespan=lifespan,
    )

    def validated_question(body: AskRequest, request: Request) -> str:
        question = body.question.strip()
        limit = request.app.state.max_question_chars
        if not question:
            raise HTTPException(status_code=422, detail="The question is empty.")
        if len(question) > limit:
            raise HTTPException(status_code=422, detail=f"The question is too long (maximum {limit} characters).")
        return question

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Plain `def` endpoints: FastAPI runs them in a thread pool, so a slow question
    # does not block the server for other users.
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = validated_question(body, request)
        try:
            result = request.app.state.agent.ask(question)
        except OpenAIError as exc:
            logger.exception("Language model provider error")
            raise HTTPException(status_code=502, detail=MODEL_ERROR) from exc
        except Exception as exc:  # API boundary: log everything, never leak internals to the client
            logger.exception("Unexpected error while answering")
            raise HTTPException(status_code=500, detail=INTERNAL_ERROR) from exc
        return to_response(result)

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
        question = validated_question(body, request)  # invalid input -> 422 before streaming starts
        running_agent = request.app.state.agent

        def events() -> Iterator[str]:
            try:
                for event in running_agent.stream(question):
                    if isinstance(event, AgentResult):
                        yield _sse("result", to_response(event).model_dump())
                    else:
                        yield _sse("step", {"node": event.node, "seconds": event.seconds})
            except OpenAIError:
                logger.exception("Language model provider error")
                yield _sse("error", {"detail": MODEL_ERROR})
            except Exception:  # never leak internals, even mid-stream
                logger.exception("Unexpected error while answering")
                yield _sse("error", {"detail": INTERNAL_ERROR})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event; the JSON payload is a single line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)

app = create_app()