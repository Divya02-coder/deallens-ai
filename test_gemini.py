import os
import pytest


@pytest.mark.integration
def test_gemini_api():
    """
    Optional live Gemini API test.

    Normal pytest runs skip this test.
    Run with:
        set RUN_GEMINI_TEST=1
        pytest -q -m integration
    """

    if os.getenv("RUN_GEMINI_TEST") != "1":
        pytest.skip("Gemini integration test disabled.")

    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured.")

    from ai_engine.llm import GeminiLLM

    llm = GeminiLLM()

    response = llm.generate(
        "Reply with exactly: DealLens Gemini integration OK"
    )

    assert response is not None
    assert len(response.strip()) > 0