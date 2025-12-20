"""
End-to-end tests for config.py with real API calls.

Run with: pytest tests/test_config_e2e.py -v -s

Requires:
- OPENAI_API_KEY environment variable set
- Network access to OpenAI API

This file tests the sample emotion classification task - students should
modify these tests when adapting config.py for their own use case.
"""

import os

import pytest
from dotenv import load_dotenv

import config

# Load environment variables from .env file
load_dotenv()

# Skip all tests in this file if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestEvalE2E:
    """E2E tests for eval function with real LLM calls."""

    def test_eval_returns_valid_emotion_joy(self):
        """Test that eval returns a valid emotion for happy text."""
        row = {"text": "I'm so happy today! This is the best day ever!"}

        result = config.eval(
            row,
            system_prompt="You are an emotion classifier. Classify the emotion as exactly one of: joy, anger, sadness, surprise. Output ONLY the emotion label.",
            user_prompt_template="{text}",
        )

        assert "response" in result
        assert "predicted_emotion" in result
        assert result["predicted_emotion"] in config.VALID_EMOTIONS
        # Happy text should likely be classified as joy
        assert result["predicted_emotion"] == "joy"

    def test_eval_returns_valid_emotion_anger(self):
        """Test that eval returns a valid emotion for angry text."""
        row = {"text": "This is absolutely outrageous! I can't believe they did this!"}

        result = config.eval(
            row,
            system_prompt="You are an emotion classifier. Classify the emotion as exactly one of: joy, anger, sadness, surprise. Output ONLY the emotion label.",
            user_prompt_template="{text}",
        )

        assert result["predicted_emotion"] in config.VALID_EMOTIONS
        assert result["predicted_emotion"] == "anger"

    def test_eval_returns_valid_emotion_sadness(self):
        """Test that eval returns a valid emotion for sad text."""
        row = {"text": "I feel so lonely and heartbroken. Everything feels hopeless."}

        result = config.eval(
            row,
            system_prompt="You are an emotion classifier. Classify the emotion as exactly one of: joy, anger, sadness, surprise. Output ONLY the emotion label.",
            user_prompt_template="{text}",
        )

        assert result["predicted_emotion"] in config.VALID_EMOTIONS
        assert result["predicted_emotion"] == "sadness"

    def test_eval_returns_valid_emotion_surprise(self):
        """Test that eval returns a valid emotion for surprised text."""
        row = {"text": "Oh my goodness! I never expected this! What a shock!"}

        result = config.eval(
            row,
            system_prompt="You are an emotion classifier. Classify the emotion as exactly one of: joy, anger, sadness, surprise. Output ONLY the emotion label.",
            user_prompt_template="{text}",
        )

        assert result["predicted_emotion"] in config.VALID_EMOTIONS
        assert result["predicted_emotion"] == "surprise"


class TestScoreE2E:
    """E2E tests for score function (no LLM calls for exact match)."""

    def test_score_exact_match(self):
        """Test scoring with exact match."""
        row = {"predicted_emotion": "joy", "emotion": "joy"}
        result = config.score(row, grader_prompt=None)
        assert result["accuracy"] == 1.0

    def test_score_mismatch(self):
        """Test scoring with mismatch."""
        row = {"predicted_emotion": "anger", "emotion": "joy"}
        result = config.score(row, grader_prompt=None)
        assert result["accuracy"] == 0.0


class TestAnalyzeE2E:
    """E2E tests for analyze function with real LLM calls."""

    def test_analyze_identifies_patterns(self):
        """Test that analyze identifies error patterns."""
        rows = [
            {"text": "Wow!", "emotion": "surprise", "predicted_emotion": "joy", "accuracy": 0.0},
            {"text": "Amazing!", "emotion": "surprise", "predicted_emotion": "joy", "accuracy": 0.0},
            {"text": "Unbelievable!", "emotion": "surprise", "predicted_emotion": "joy", "accuracy": 0.0},
        ]

        analysis_template = """Analyze these emotion classification failures:

{% for row in rows %}
- Text: "{{ row.text }}"
- Expected: {{ row.emotion }}
- Predicted: {{ row.predicted_emotion }}
{% endfor %}

Identify the common pattern in these failures in 1-2 sentences."""

        result = config.analyze(
            rows,
            analysis_template,
            model="openai/gpt-4o-mini",
        )

        assert result is not None
        assert len(result) > 20


class TestClusterFailuresE2E:
    """E2E tests for clustering with real LLM calls."""

    @pytest.fixture
    def sample_failures(self):
        """Sample failure rows for testing."""
        return [
            {
                "_example_id": 1,
                "text": "Wow! I can't believe it!",
                "emotion": "surprise",
                "predicted_emotion": "joy",
                "accuracy": 0.0,
                "accuracy_reason": "Predicted joy instead of surprise"
            },
            {
                "_example_id": 2,
                "text": "Amazing! This is incredible!",
                "emotion": "surprise",
                "predicted_emotion": "joy",
                "accuracy": 0.0,
                "accuracy_reason": "Predicted joy instead of surprise"
            },
            {
                "_example_id": 3,
                "text": "I feel so down today",
                "emotion": "sadness",
                "predicted_emotion": "anger",
                "accuracy": 0.0,
                "accuracy_reason": "Predicted anger instead of sadness"
            },
            {
                "_example_id": 4,
                "text": "Everything is terrible",
                "emotion": "sadness",
                "predicted_emotion": "anger",
                "accuracy": 0.0,
                "accuracy_reason": "Predicted anger instead of sadness"
            },
        ]

    @pytest.fixture
    def clustering_template(self):
        """Load the clustering template."""
        with open("clustering-prompt.jinja2", "r") as f:
            return f.read()

    def test_cluster_failures_returns_valid_structure(self, sample_failures, clustering_template):
        """Test that cluster_failures returns properly structured output."""
        result = config.cluster_failures(
            rows=sample_failures,
            clustering_prompt_template=clustering_template,
            score_column="accuracy",
            model="openai/gpt-4o-mini",
            max_clusters=3
        )

        # Check structure
        assert "clusters" in result
        assert len(result["clusters"]) >= 1
        assert len(result["clusters"]) <= 3

        for cluster in result["clusters"]:
            assert "label" in cluster
            assert "description" in cluster
            assert "example_ids" in cluster
            assert isinstance(cluster["example_ids"], list)

        # Verify all example IDs are from our input
        all_ids = set()
        for cluster in result["clusters"]:
            all_ids.update(cluster["example_ids"])
        valid_ids = {1, 2, 3, 4}
        assert all_ids.issubset(valid_ids), f"Invalid IDs found: {all_ids - valid_ids}"

        print("\n=== Clustering Results ===")
        for i, cluster in enumerate(result["clusters"]):
            print(f"Cluster {i+1}: {cluster['label']}")
            print(f"  Description: {cluster['description']}")
            print(f"  IDs: {cluster['example_ids']}")
