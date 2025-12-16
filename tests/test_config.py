"""
Unit tests for config.py

These tests use mocking to avoid actual LLM calls.
"""

from unittest.mock import MagicMock, patch

import config


class TestStratify:
    """Tests for stratify function."""

    def test_finds_category_column(self):
        df = MagicMock()
        df.columns = ["text", "category", "value"]
        result = config.stratify(df)
        assert result == "category"

    def test_finds_label_column(self):
        df = MagicMock()
        df.columns = ["text", "label"]
        result = config.stratify(df)
        assert result == "label"

    def test_returns_none_when_no_match(self):
        df = MagicMock()
        df.columns = ["text", "value", "other"]
        result = config.stratify(df)
        assert result is None


class TestEval:
    """Tests for eval function."""

    @patch("config.call_llm")
    def test_basic_eval(self, mock_llm):
        mock_llm.return_value = (
            "Analysis complete.\n**Score:** 4.5\nReasoning: The answer is good."
        )

        row = {"question": "What is AI?", "answer": "Artificial Intelligence"}
        result = config.eval(
            row,
            system_prompt="You are a grader.",
            user_prompt_template="Question: {question}\nAnswer: {answer}",
            model="test-model",
        )

        assert "llm_response" in result
        assert result["extracted_score"] == 4.5
        assert "extracted_reasoning" in result

    @patch("config.call_llm")
    def test_missing_score_in_response(self, mock_llm):
        mock_llm.return_value = "This response has no score."

        row = {"question": "Test"}
        result = config.eval(
            row,
            system_prompt="Test",
            user_prompt_template="{question}",
            model="test-model",
        )

        assert "llm_response" in result
        assert "extracted_score" not in result


class TestScore:
    """Tests for score function."""

    def test_accuracy_calculation(self):
        row = {
            "extracted_score": 4.0,
            "expected_score": 4.5,
            "llm_response": "Some response",
        }

        result = config.score(row, grader_prompt=None, model="test-model")

        assert "accuracy" in result
        assert "accuracy_reason" in result
        assert 0.8 < result["accuracy"] < 1.0  # Difference of 0.5 on 1-5 scale

    def test_perfect_score(self):
        row = {
            "extracted_score": 5.0,
            "expected_score": 5.0,
            "llm_response": "Response",
        }

        result = config.score(row, grader_prompt=None, model="test-model")

        assert result["accuracy"] == 1.0

    @patch("config.call_llm_single_prompt")
    def test_with_grader_prompt(self, mock_llm):
        mock_llm.return_value = "Quality: 8/10. Good response."

        row = {
            "llm_response": "Test response",
            "question": "Test question",
        }
        grader_prompt = "Rate this: {{ row.llm_response }}"

        _result = config.score(row, grader_prompt=grader_prompt, model="test-model")

        # Should have called the LLM
        mock_llm.assert_called_once()
        assert _result is not None


class TestOptimize:
    """Tests for optimize function."""

    @patch("config.call_llm_single_prompt")
    def test_extracts_from_tags(self, mock_llm):
        mock_llm.return_value = """
        Here's my analysis...

        <optimized_prompt>
        You are an improved assistant.
        </optimized_prompt>
        """

        result = config.optimize(
            optimizer_prompt_template="Optimize: {{ system_prompt }}",
            system_prompt="You are an assistant.",
            user_prompt_template="{question}",
            examples=[{"question": "test", "score": 0.5}],
            analysis=None,
            model="test-model",
        )

        assert result == "You are an improved assistant."

    @patch("config.call_llm_single_prompt")
    def test_extracts_from_header(self, mock_llm):
        mock_llm.return_value = """
        Analysis of issues...

        Optimized Prompt:
        You are a better assistant that handles edge cases.
        """

        result = config.optimize(
            optimizer_prompt_template="{{ system_prompt }}",
            system_prompt="Original",
            user_prompt_template="",
            examples=[],
            analysis=None,
            model="test-model",
        )

        assert "better assistant" in result


class TestAnalyze:
    """Tests for analyze function."""

    @patch("config.call_llm_single_prompt")
    def test_basic_analysis(self, mock_llm):
        mock_llm.return_value = "Common issues found: 1. Vague responses 2. Missing details"

        rows = [
            {"question": "Q1", "accuracy": 0.5, "accuracy_reason": "poor"},
            {"question": "Q2", "accuracy": 0.6, "accuracy_reason": "okay"},
        ]

        result = config.analyze(
            rows,
            analysis_prompt_template="Analyze: {% for row in rows %}{{ row.question }}{% endfor %}",
            model="test-model",
        )

        assert "Common issues" in result
        mock_llm.assert_called_once()
