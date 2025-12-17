"""
Unit tests for utils.py
"""

import os

import pandas as pd
import pytest

from utils import (
    add_example_ids,
    bootstrap_ci,
    calculate_score_averages,
    detect_regressions,
    ensure_dir,
    extract_score_columns,
    format_score_with_ci,
    format_user_prompt,
    get_trend_label,
    load_project_metadata,
    load_prompt_file,
    paired_bootstrap_test,
    parse_cluster_json,
    render_jinja_template,
    sample_size_guidance,
    save_project_metadata,
    save_prompt_file,
    split_dataset,
)


class TestFormatUserPrompt:
    """Tests for format_user_prompt function."""

    def test_simple_format(self):
        template = "Question: {question}"
        row = {"question": "What is AI?"}
        result = format_user_prompt(template, row)
        assert result == "Question: What is AI?"

    def test_multiple_placeholders(self):
        template = "Question: {question}\nContext: {context}"
        row = {"question": "What is AI?", "context": "Technology basics"}
        result = format_user_prompt(template, row)
        assert result == "Question: What is AI?\nContext: Technology basics"

    def test_missing_placeholder_raises(self):
        template = "Question: {question}\nAnswer: {answer}"
        row = {"question": "What is AI?"}
        with pytest.raises(KeyError):
            format_user_prompt(template, row)

    def test_extra_columns_ignored(self):
        template = "Question: {question}"
        row = {"question": "What is AI?", "extra": "ignored"}
        result = format_user_prompt(template, row)
        assert result == "Question: What is AI?"


class TestRenderJinjaTemplate:
    """Tests for render_jinja_template function."""

    def test_simple_render(self):
        template = "Hello {{ name }}!"
        result = render_jinja_template(template, name="World")
        assert result == "Hello World!"

    def test_loop_render(self):
        template = "{% for item in items %}{{ item }},{% endfor %}"
        result = render_jinja_template(template, items=["a", "b", "c"])
        assert result == "a,b,c,"

    def test_conditional_render(self):
        template = "{% if show %}Visible{% endif %}"
        assert render_jinja_template(template, show=True) == "Visible"
        assert render_jinja_template(template, show=False) == ""

    def test_dict_access(self):
        template = "{{ row.name }} - {{ row.score }}"
        result = render_jinja_template(template, row={"name": "Test", "score": 5})
        assert result == "Test - 5"


class TestSplitDataset:
    """Tests for split_dataset function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "text": [f"text_{i}" for i in range(100)],
                "category": ["A"] * 50 + ["B"] * 50,
                "score": [1, 2, 3, 4, 5] * 20,
            }
        )

    def test_split_ratios(self, sample_df):
        train, dev, test = split_dataset(sample_df, "40/40/20")
        total = len(train) + len(dev) + len(test)
        assert total == 100
        assert abs(len(train) - 40) <= 2  # Allow small variance
        assert abs(len(dev) - 40) <= 2
        assert abs(len(test) - 20) <= 2

    def test_different_ratios(self, sample_df):
        train, dev, test = split_dataset(sample_df, "60/20/20")
        assert len(train) > len(dev)
        assert len(train) > len(test)

    def test_stratification(self, sample_df):
        train, dev, test = split_dataset(sample_df, "40/40/20", stratify_column="category")
        # Check that both categories are in all splits
        assert set(train["category"].unique()) == {"A", "B"}
        assert set(dev["category"].unique()) == {"A", "B"}
        assert set(test["category"].unique()) == {"A", "B"}

    def test_reproducibility(self, sample_df):
        train1, dev1, test1 = split_dataset(sample_df, "40/40/20", random_state=42)
        train2, dev2, test2 = split_dataset(sample_df, "40/40/20", random_state=42)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(dev1, dev2)
        pd.testing.assert_frame_equal(test1, test2)


class TestScoreUtilities:
    """Tests for score-related utilities."""

    def test_calculate_score_averages(self):
        df = pd.DataFrame(
            {
                "accuracy": [0.8, 0.9, 1.0],
                "accuracy_reason": ["good", "great", "perfect"],
                "relevance": [0.7, 0.8, 0.9],
            }
        )
        averages = calculate_score_averages(df, ["accuracy", "relevance"])
        assert abs(averages["accuracy"] - 0.9) < 0.01
        assert abs(averages["relevance"] - 0.8) < 0.01

    def test_extract_score_columns(self):
        df = pd.DataFrame(
            {
                "text": ["a", "b"],
                "accuracy": [0.8, 0.9],
                "accuracy_reason": ["good", "great"],
                "count": [1, 2],
            }
        )
        score_cols = extract_score_columns(df)
        assert "accuracy" in score_cols
        assert "accuracy_reason" not in score_cols
        assert "text" not in score_cols


class TestFileUtilities:
    """Tests for file I/O utilities."""

    def test_save_and_load_prompt(self, tmp_path):
        prompt_path = tmp_path / "test_prompt.txt"
        content = "This is a test prompt\nWith multiple lines"

        save_prompt_file(str(prompt_path), content)
        loaded = load_prompt_file(str(prompt_path))

        assert loaded == content

    def test_save_and_load_metadata(self, tmp_path):
        metadata = {"project_name": "test", "created_at": "2024-01-01T00:00:00"}

        save_project_metadata(str(tmp_path), metadata)
        loaded = load_project_metadata(str(tmp_path))

        assert loaded == metadata

    def test_ensure_dir_creates_nested(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c"
        ensure_dir(str(nested_path))
        assert os.path.exists(nested_path)

    def test_ensure_dir_idempotent(self, tmp_path):
        path = tmp_path / "test"
        ensure_dir(str(path))
        ensure_dir(str(path))  # Should not raise
        assert os.path.exists(path)


class TestStatisticalUtilities:
    """Tests for bootstrap CI and significance testing."""

    def test_bootstrap_ci_returns_tuple(self):
        scores = [0.7, 0.8, 0.75, 0.85, 0.9]
        lower, upper = bootstrap_ci(scores)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_bootstrap_ci_empty_list(self):
        lower, upper = bootstrap_ci([])
        assert lower == 0.0
        assert upper == 0.0

    def test_bootstrap_ci_single_value(self):
        lower, upper = bootstrap_ci([0.5])
        assert lower == 0.5
        assert upper == 0.5

    def test_bootstrap_ci_contains_mean(self):
        scores = [0.7, 0.8, 0.75, 0.85, 0.9]
        lower, upper = bootstrap_ci(scores)
        mean = sum(scores) / len(scores)
        assert lower <= mean <= upper

    def test_paired_bootstrap_test_significant_difference(self):
        # Clear improvement
        scores_a = [0.5, 0.5, 0.5, 0.5, 0.5]
        scores_b = [0.9, 0.9, 0.9, 0.9, 0.9]
        result = paired_bootstrap_test(scores_a, scores_b)
        assert result["significant"] is True
        assert result["observed_diff"] > 0

    def test_paired_bootstrap_test_no_difference(self):
        # Same scores
        scores = [0.7, 0.8, 0.75, 0.85, 0.9]
        result = paired_bootstrap_test(scores, scores)
        assert result["significant"] is False
        assert abs(result["observed_diff"]) < 0.01

    def test_paired_bootstrap_test_length_mismatch(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test([0.5, 0.6], [0.5])

    def test_sample_size_guidance_small(self):
        assert "Too few" in sample_size_guidance(10)

    def test_sample_size_guidance_medium(self):
        assert "large effects" in sample_size_guidance(30)

    def test_sample_size_guidance_good(self):
        assert "medium effects" in sample_size_guidance(75)

    def test_sample_size_guidance_excellent(self):
        assert "small effects" in sample_size_guidance(150)

    def test_format_score_with_ci(self):
        scores = [0.7, 0.8, 0.75, 0.85, 0.9]
        result = format_score_with_ci(scores)
        assert "+/-" in result
        assert "0.8" in result  # Mean is 0.8


class TestExampleTracking:
    """Tests for example ID generation and regression detection."""

    def test_add_example_ids_generates_sequential(self):
        df = pd.DataFrame({"question": ["Q1", "Q2", "Q3"]})
        result = add_example_ids(df)
        assert "_example_id" in result.columns
        assert list(result["_example_id"]) == [0, 1, 2]

    def test_add_example_ids_uses_existing_id(self):
        df = pd.DataFrame({"id": ["a", "b", "c"], "question": ["Q1", "Q2", "Q3"]})
        result = add_example_ids(df)
        assert "_example_id" in result.columns
        assert list(result["_example_id"]) == ["a", "b", "c"]

    def test_add_example_ids_preserves_existing(self):
        df = pd.DataFrame({"_example_id": [10, 20, 30], "question": ["Q1", "Q2", "Q3"]})
        result = add_example_ids(df)
        assert list(result["_example_id"]) == [10, 20, 30]

    def test_get_trend_label_improving(self):
        scores = [0.5, 0.6, 0.7]
        runs = ["v1", "v2", "v3"]
        assert "Improving" in get_trend_label(scores, runs)

    def test_get_trend_label_regressed(self):
        scores = [0.8, 0.6]
        runs = ["v1", "v2"]
        assert "Regressed" in get_trend_label(scores, runs)

    def test_get_trend_label_oscillating(self):
        scores = [0.5, 0.8, 0.5]
        runs = ["v1", "v2", "v3"]
        assert "Oscillating" in get_trend_label(scores, runs)

    def test_get_trend_label_stable(self):
        scores = [0.75, 0.76]
        runs = ["v1", "v2"]
        assert "Stable" in get_trend_label(scores, runs)

    def test_detect_regressions_finds_broke(self):
        history_df = pd.DataFrame({
            "_example_id": [1, 2],
            "baseline": [0.9, 0.4],
            "v2": [0.3, 0.8]
        })
        result = detect_regressions(history_df, ["baseline", "v2"])
        assert 1 in result["broke"]
        assert 2 not in result["broke"]

    def test_detect_regressions_finds_improved(self):
        history_df = pd.DataFrame({
            "_example_id": [1],
            "baseline": [0.5],
            "v2": [0.8]
        })
        result = detect_regressions(history_df, ["baseline", "v2"])
        assert 1 in result["improved"]


class TestParseClusterJson:
    """Tests for JSON parsing from LLM responses."""

    def test_parse_cluster_json_from_code_block(self):
        response = '''Here is the analysis:
```json
{"clusters": [{"label": "Test", "description": "Test desc", "example_ids": [1, 2]}]}
```
'''
        result = parse_cluster_json(response)
        assert result is not None
        assert "clusters" in result
        assert len(result["clusters"]) == 1

    def test_parse_cluster_json_raw(self):
        response = '{"clusters": [{"label": "Test", "description": "Desc", "example_ids": [1]}]}'
        result = parse_cluster_json(response)
        assert result is not None
        assert result["clusters"][0]["label"] == "Test"

    def test_parse_cluster_json_invalid(self):
        response = "This is not valid JSON at all"
        result = parse_cluster_json(response)
        assert result is None

    def test_parse_cluster_json_partial(self):
        response = '''Some text before
{"clusters": [{"label": "A", "description": "B", "example_ids": [1, 2, 3]}]}
Some text after'''
        result = parse_cluster_json(response)
        assert result is not None

    def test_parse_cluster_json_empty_response(self):
        result = parse_cluster_json("")
        assert result is None
