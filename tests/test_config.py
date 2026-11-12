"""Configuration tests: the environment exactly as `docker run --env-file` delivers it."""

from __future__ import annotations

import pytest

from app.config import load_settings


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A clean, minimal environment (the local .env loaded at import is overridden)."""
    for name in ("OPENAI_BASE_URL", "OPENAI_BASE_UR", "MAX_REVISIONS", "TASK_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "model-x")
    return monkeypatch


def test_quotes_passed_literally_by_docker_are_stripped(env: pytest.MonkeyPatch) -> None:
    env.setenv("OPENAI_BASE_URL", "'https://proxy.example.com'")
    env.setenv("OPENAI_MODEL", '"gpt-test"')
    settings = load_settings()
    assert settings.openai_base_url == "https://proxy.example.com"
    assert settings.chat_model == "gpt-test"


def test_misspelled_base_url_variable_is_accepted(env: pytest.MonkeyPatch) -> None:
    env.setenv("OPENAI_BASE_UR", "'https://proxy.example.com'")
    assert load_settings().openai_base_url == "https://proxy.example.com"


def test_correct_spelling_wins_over_the_typo(env: pytest.MonkeyPatch) -> None:
    env.setenv("OPENAI_BASE_URL", "https://correct.example.com")
    env.setenv("OPENAI_BASE_UR", "https://typo.example.com")
    assert load_settings().openai_base_url == "https://correct.example.com"


def test_missing_required_variable_is_named_in_the_error(env: pytest.MonkeyPatch) -> None:
    env.delenv("OPENAI_MODEL")
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        load_settings()


def test_quoted_empty_value_counts_as_missing(env: pytest.MonkeyPatch) -> None:
    env.setenv("OPENAI_MODEL", "''")
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        load_settings()


def test_integer_settings_are_validated(env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_REVISIONS", "0")
    assert load_settings().max_revisions == 0  # 0 is allowed: it disables revision
    env.setenv("TASK_TOP_K", "zero")
    with pytest.raises(RuntimeError, match="TASK_TOP_K"):
        load_settings()

def test_llm_retries_are_configurable_and_zero_is_allowed(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_MAX_RETRIES", "0")
    assert load_settings().llm_max_retries == 0
    env.setenv("LLM_MAX_RETRIES", "many")
    with pytest.raises(RuntimeError, match="LLM_MAX_RETRIES"):
        load_settings()
