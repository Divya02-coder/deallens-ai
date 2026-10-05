import os
import pytest


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not configured",
)
def test_gemini_connection():

    from ai_engine.llm import GeminiLLM

    llm = GeminiLLM()

    result = llm.generate(
        "Reply with exactly: DealLens AI OK"
    )

    assert result