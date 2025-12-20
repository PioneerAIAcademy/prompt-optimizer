"""
End-to-end tests with real API calls for utils.py.

Run with: pytest tests/test_e2e.py -v -s

Requires:
- OPENAI_API_KEY environment variable set
- Network access to OpenAI API

Note: Config-specific E2E tests are in test_config_e2e.py.
"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Skip all tests in this file if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestLLMCalls:
    """E2E tests for LLM utility functions."""

    def test_call_llm_basic(self):
        from utils import call_llm

        response = call_llm(
            system_prompt="You are a helpful assistant. Respond in exactly one word.",
            user_prompt="What is 2+2?",
            model="openai/gpt-4o-mini",
        )

        assert response is not None
        assert len(response) > 0
        assert "4" in response.lower() or "four" in response.lower()

    def test_call_llm_single_prompt(self):
        from utils import call_llm_single_prompt

        response = call_llm_single_prompt(
            prompt="What is the capital of France? Answer in one word.",
            model="openai/gpt-4o-mini",
        )

        assert "paris" in response.lower()
