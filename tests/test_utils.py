"""
Unit tests for utils.py
"""

import os

import pandas as pd
import pytest

from utils import (
    calculate_score_averages,
    ensure_dir,
    extract_score_columns,
    format_user_prompt,
    load_project_metadata,
    load_prompt_file,
    render_jinja_template,
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
