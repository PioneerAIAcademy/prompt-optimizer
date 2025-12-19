"""
End-to-end tests with real API calls.

Run with: pytest tests/test_e2e.py -v -s

Requires:
- OPENAI_API_KEY environment variable set
- Network access to OpenAI API
"""

import os

import pandas as pd
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


class TestConfigFunctions:
    """E2E tests for config.py functions."""

    def test_eval_function(self):
        import config

        row = {
            "question": "What is photosynthesis?",
            "context": "Biology topic about plants",
            "answer": "Photosynthesis is how plants make food from sunlight.",
        }

        system_prompt = """You are a grading assistant. Evaluate the answer and provide:
1. A score from 1-5
2. Brief reasoning

Format your response as:
**Score:** [number]
Reasoning: [your reasoning]"""

        user_prompt_template = (
            "Question: {question}\nContext: {context}\nStudent Answer: {answer}"
        )

        result = config.eval(
            row,
            system_prompt,
            user_prompt_template,
            model="openai/gpt-4o-mini",
        )

        assert "response" in result
        assert len(result["response"]) > 0
        # Score extraction is best-effort, may or may not succeed

    def test_score_function_without_grader(self):
        import config

        row = {
            "score": 4.0,
            "expected_score": 4.5,
            "response": "Some response text",
        }

        result = config.score(row, grader_prompt=None, model="openai/gpt-4o-mini")

        assert "accuracy" in result
        assert "accuracy_reason" in result
        assert 0 <= result["accuracy"] <= 1

    def test_analyze_function(self):
        import config

        rows = [
            {
                "question": "What is AI?",
                "llm_response": "AI is artificial intelligence.",
                "accuracy": 0.6,
                "accuracy_reason": "Too brief",
            },
            {
                "question": "Explain gravity",
                "llm_response": "Gravity pulls things down.",
                "accuracy": 0.5,
                "accuracy_reason": "Incomplete explanation",
            },
        ]

        # Use a simple inline template for testing
        analysis_template = """Analyze these evaluation results:
{% for row in rows %}
- Question: {{ row.question }}
- Response: {{ row.llm_response }}
- Score: {{ row.accuracy }}
{% endfor %}

Identify common issues in 2-3 sentences."""

        result = config.analyze(
            rows,
            analysis_template,
            model="openai/gpt-4o-mini",
        )

        assert result is not None
        assert len(result) > 50  # Should be a substantial analysis

    def test_optimize_function(self):
        import config

        optimizer_template = """Improve this prompt based on the examples.

Current prompt:
{{ system_prompt }}

Examples showing issues:
{% for ex in examples %}
- Input: {{ ex.question }}
- Score: {{ ex.accuracy }}
{% endfor %}

{% if analysis %}
Analysis: {{ analysis }}
{% endif %}

Provide improved prompt between <optimized_prompt> tags."""

        examples = [
            {"question": "What is AI?", "accuracy": 0.5, "accuracy_reason": "Too vague"},
        ]

        result = config.optimize(
            optimizer_prompt_template=optimizer_template,
            system_prompt="You are a helpful assistant.",
            user_prompt_template="{question}",
            examples=examples,
            analysis="Responses are too brief and lack detail.",
            model="openai/gpt-4o-mini",
        )

        assert result is not None
        assert len(result) > 20


class TestFullWorkflow:
    """E2E test of the complete workflow."""

    def test_complete_workflow(self, tmp_path):
        """Test creating a project, evaluating, and optimizing."""
        import config
        from utils import (
            calculate_score_averages,
            ensure_dir,
            extract_score_columns,
            save_prompt_file,
            split_dataset,
        )

        # 1. Create a small dataset
        df = pd.DataFrame(
            {
                "question": [
                    "What is AI?",
                    "Explain gravity",
                    "What is DNA?",
                ],
                "expected_score": [4.0, 4.5, 4.0],
                "category": ["tech", "physics", "biology"],
            }
        )

        # 2. Split the dataset
        train_df, dev_df, test_df = split_dataset(
            df, "33/33/34", stratify_column=None  # Too few samples for stratification
        )

        assert len(train_df) + len(dev_df) + len(test_df) == 3

        # 3. Set up project structure
        project_path = tmp_path / "test-project"
        run_path = project_path / "baseline"
        ensure_dir(str(run_path))

        # 4. Save prompts
        system_prompt = """You are a grading assistant. For each question, provide:
1. A score from 1-5
2. Brief reasoning

Format: **Score:** [number]
Reasoning: [text]"""

        user_prompt_template = "Evaluate this question: {question}"

        save_prompt_file(str(run_path / "system_prompt.txt"), system_prompt)
        save_prompt_file(str(run_path / "user_prompt.txt"), user_prompt_template)

        # 5. Run evaluation on train split
        results = []
        for _, row in train_df.iterrows():
            row_dict = row.to_dict()

            # Eval
            eval_result = config.eval(
                row_dict,
                system_prompt,
                user_prompt_template,
                model="openai/gpt-4o-mini",
            )
            row_dict.update(eval_result)

            # Score
            score_result = config.score(
                row_dict,
                grader_prompt=None,
                model="openai/gpt-4o-mini",
            )
            row_dict.update(score_result)

            results.append(row_dict)

        results_df = pd.DataFrame(results)

        # Verify we have results
        assert len(results_df) > 0
        assert "response" in results_df.columns

        # 6. Calculate averages
        score_cols = extract_score_columns(results_df)
        if score_cols:
            averages = calculate_score_averages(results_df, score_cols)
            assert len(averages) > 0

        print("\n=== E2E Test Complete ===")
        print(f"Evaluated {len(results_df)} rows")
        print(f"Columns: {list(results_df.columns)}")
        if score_cols:
            print(f"Score averages: {averages}")


class TestClusterFailuresE2E:
    """E2E tests for clustering with real LLM calls."""

    @pytest.fixture
    def sample_failures(self):
        """Sample failure rows for testing."""
        return [
            {
                "_example_id": 1,
                "question": "What is 2+2?",
                "expected": "4",
                "llm_response": "The answer is 5",
                "accuracy": 0.0,
                "accuracy_reason": "Wrong answer"
            },
            {
                "_example_id": 2,
                "question": "What is the capital of France?",
                "expected": "Paris",
                "llm_response": "I don't know",
                "accuracy": 0.0,
                "accuracy_reason": "No answer provided"
            },
            {
                "_example_id": 3,
                "question": "Explain quantum entanglement",
                "expected": "Particles remain connected...",
                "llm_response": "It's when particles are close together",
                "accuracy": 0.3,
                "accuracy_reason": "Partial understanding, missing key concepts"
            },
            {
                "_example_id": 4,
                "question": "What is 10*10?",
                "expected": "100",
                "llm_response": "The answer is 1000",
                "accuracy": 0.0,
                "accuracy_reason": "Wrong calculation"
            },
        ]

    @pytest.fixture
    def clustering_template(self):
        """Load the clustering template."""
        with open("clustering-prompt.jinja2", "r") as f:
            return f.read()

    def test_cluster_failures_returns_valid_structure(self, sample_failures, clustering_template):
        """Test that cluster_failures returns properly structured output."""
        import config

        # Use the default model
        model = os.environ.get("TEST_MODEL", "openai/gpt-4o-mini")

        result = config.cluster_failures(
            rows=sample_failures,
            clustering_prompt_template=clustering_template,
            score_column="accuracy",
            model=model,
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
