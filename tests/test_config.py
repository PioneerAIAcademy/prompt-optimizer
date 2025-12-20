"""
Unit tests for config.py

These tests use mocking to avoid actual LLM calls.
This file tests the sample emotion classification task - students should
modify these tests when adapting config.py for their own use case.
"""

from unittest.mock import MagicMock, patch

import config
from config import EmotionResponse


class TestStratify:
    """Tests for stratify function."""

    def test_returns_emotion(self):
        """stratify() returns 'emotion' for this project's dataset."""
        df = MagicMock()
        df.columns = ["text", "emotion"]
        result = config.stratify(df)
        assert result == "emotion"


class TestPrimaryScore:
    """Tests for primary_score function."""

    def test_returns_accuracy(self):
        """primary_score() returns 'accuracy' for this project's dataset."""
        df = MagicMock()
        df.columns = ["accuracy", "accuracy_reason"]
        result = config.primary_score(df)
        assert result == "accuracy"


class TestEval:
    """Tests for eval function."""

    @patch("config.call_llm_structured")
    def test_basic_eval_joy(self, mock_llm):
        """Test eval with valid 'joy' emotion."""
        mock_llm.return_value = EmotionResponse(emotion="joy")

        row = {"text": "I'm so happy today!"}
        result = config.eval(
            row,
            system_prompt="You are an emotion classifier.",
            user_prompt_template="{text}",
        )

        assert result["response"] == "joy"
        assert result["predicted_emotion"] == "joy"

    @patch("config.call_llm_structured")
    def test_eval_normalizes_case(self, mock_llm):
        """Test that emotion is normalized to lowercase."""
        mock_llm.return_value = EmotionResponse(emotion="ANGER")

        row = {"text": "This is outrageous!"}
        result = config.eval(
            row,
            system_prompt="Classify emotion",
            user_prompt_template="{text}",
        )

        assert result["predicted_emotion"] == "anger"

    @patch("config.time.sleep")  # Skip actual sleep during tests
    @patch("config.call_llm_structured")
    def test_invalid_emotion_retries(self, mock_llm, mock_sleep):
        """Test that invalid emotions trigger retries."""
        # First two calls return invalid, third returns valid
        mock_llm.side_effect = [
            EmotionResponse(emotion="happy"),  # Invalid
            EmotionResponse(emotion="excited"),  # Invalid
            EmotionResponse(emotion="joy"),  # Valid
        ]

        row = {"text": "I'm so happy!"}
        result = config.eval(
            row,
            system_prompt="Classify",
            user_prompt_template="{text}",
        )

        assert result["predicted_emotion"] == "joy"
        assert mock_llm.call_count == 3

    @patch("config.time.sleep")
    @patch("config.call_llm_structured")
    def test_invalid_emotion_after_all_retries(self, mock_llm, mock_sleep):
        """Test behavior when all retries return invalid emotions."""
        mock_llm.return_value = EmotionResponse(emotion="happy")  # Invalid

        row = {"text": "I'm so happy!"}
        result = config.eval(
            row,
            system_prompt="Classify",
            user_prompt_template="{text}",
        )

        # Should still return the invalid emotion (will score 0.0)
        assert result["predicted_emotion"] == "happy"
        assert mock_llm.call_count == 3


class TestScore:
    """Tests for score function."""

    def test_exact_match(self):
        """Test exact match scoring."""
        row = {
            "predicted_emotion": "joy",
            "emotion": "joy",
        }

        result = config.score(row, grader_prompt=None)

        assert result["accuracy"] == 1.0
        assert "joy" in result["accuracy_reason"]

    def test_mismatch(self):
        """Test mismatch scoring."""
        row = {
            "predicted_emotion": "anger",
            "emotion": "joy",
        }

        result = config.score(row, grader_prompt=None)

        assert result["accuracy"] == 0.0
        assert "anger" in result["accuracy_reason"]
        assert "joy" in result["accuracy_reason"]

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        row = {
            "predicted_emotion": "JOY",
            "emotion": "joy",
        }

        result = config.score(row, grader_prompt=None)

        assert result["accuracy"] == 1.0

    def test_missing_fields(self):
        """Test behavior with missing fields."""
        row = {}

        result = config.score(row, grader_prompt=None)

        assert result["accuracy"] == 1.0  # Empty strings match


class TestOptimize:
    """Tests for optimize function."""

    @patch("config.call_llm_single_prompt")
    def test_extracts_from_tags(self, mock_llm):
        mock_llm.return_value = """
        Here's my analysis...

        <optimized_prompt>
        You are an improved emotion classifier.
        </optimized_prompt>
        """

        result = config.optimize(
            optimizer_prompt_template="Optimize: {{ system_prompt }}",
            system_prompt="You are an emotion classifier.",
            user_prompt_template="{text}",
            examples=[{"text": "happy text", "predicted_emotion": "anger"}],
            analysis=None,
            model="test-model",
        )

        assert result == "You are an improved emotion classifier."

    @patch("config.call_llm_single_prompt")
    def test_extracts_from_header(self, mock_llm):
        mock_llm.return_value = """
        Analysis of issues...

        Optimized Prompt:
        You are a better emotion classifier that handles edge cases.
        """

        result = config.optimize(
            optimizer_prompt_template="{{ system_prompt }}",
            system_prompt="Original",
            user_prompt_template="",
            examples=[],
            analysis=None,
            model="test-model",
        )

        assert "better emotion classifier" in result


class TestAnalyze:
    """Tests for analyze function."""

    @patch("config.call_llm_single_prompt")
    def test_basic_analysis(self, mock_llm):
        mock_llm.return_value = "Common issues: 1. Confusion between joy and surprise"

        rows = [
            {"text": "Wow!", "emotion": "surprise", "predicted_emotion": "joy"},
            {"text": "Amazing!", "emotion": "surprise", "predicted_emotion": "joy"},
        ]

        result = config.analyze(
            rows,
            analysis_prompt_template="Analyze: {% for row in rows %}{{ row.text }}{% endfor %}",
            model="test-model",
        )

        assert "Common issues" in result
        mock_llm.assert_called_once()
