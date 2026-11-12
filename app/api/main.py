"""FastAPI application.

Run locally:  python -m uvicorn app.api.main:app --port 8000

The agent is built once at startup (corpus + embeddings). Tests inject a fake agent
through create_app(agent=...), so they never call the LLM.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAIError

from app.agent import ResearchAgent
from app.api.schemas import AskRequest, AskResponse, to_response
from app.bootstrap import build_agent
from app.config import DEFAULT_MAX_QUESTION_CHARS, load_settings

logger = logging.getLogger(__name__)


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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain `def`: FastAPI runs it in a thread pool, so a slow question does not block the server.
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = body.question.strip()
        limit = request.app.state.max_question_chars
        if not question:
            raise HTTPException(status_code=422, detail="The question is empty.")
        if len(question) > limit:
            raise HTTPException(status_code=422, detail=f"The question is too long (maximum {limit} characters).")

        try:
            result = request.app.state.agent.ask(question)
        except OpenAIError as exc:
            logger.exception("Language model provider error")
            raise HTTPException(
                status_code=502, detail="The language model service failed. Please try again."
            ) from exc
        except Exception as exc:  # API boundary: log everything, never leak internals to the client
            logger.exception("Unexpected error while answering")
            raise HTTPException(status_code=500, detail="Internal error while answering the question.") from exc
        return to_response(result)

    return app


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)

app = create_app()