"""
Reusable utilities for the prompt optimizer app.
"""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import litellm
import pandas as pd
from jinja2 import Template
from sklearn.model_selection import train_test_split
from tenacity import retry, stop_after_attempt, wait_exponential


# =============================================================================
# LLM UTILITIES
# =============================================================================


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """
    Call an LLM using LiteLLM with retry logic.

    Args:
        system_prompt: System message content
        user_prompt: User message content
        model: LiteLLM model string (e.g., "openai/gpt-4o-mini")
        temperature: Sampling temperature (default 0.0 for determinism)

    Returns:
        The assistant's response text

    Example:
        >>> response = call_llm(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="What is 2+2?",
        ...     model="openai/gpt-4o-mini"
        ... )
    """
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_single_prompt(prompt: str, model: str, temperature: float = 0.7) -> str:
    """
    Call an LLM with a single prompt (no system/user split).
    Used for optimizer and analyzer prompts.

    Args:
        prompt: The full prompt text
        model: LiteLLM model string
        temperature: Sampling temperature

    Returns:
        The assistant's response text
    """
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


class EvaluationError(Exception):
    """Raised when evaluation fails and should stop."""

    pass


def call_llm_parallel(
    tasks: list[dict],
    model: str,
    max_workers: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Execute multiple LLM calls in parallel with retry.
    Raises EvaluationError if any task fails after retries.

    Args:
        tasks: List of dicts with 'system_prompt', 'user_prompt', and 'row_data'
        model: LiteLLM model string
        max_workers: Maximum concurrent requests
        on_progress: Optional callback(completed, total) for progress updates

    Returns:
        List of dicts with original row_data plus 'llm_response' key

    Raises:
        EvaluationError: If any LLM call fails after retries

    Example:
        >>> tasks = [
        ...     {"system_prompt": "...", "user_prompt": "...", "row_data": {"id": 1}},
        ...     {"system_prompt": "...", "user_prompt": "...", "row_data": {"id": 2}},
        ... ]
        >>> results = call_llm_parallel(tasks, "openai/gpt-4o-mini")
    """
    results: list[dict] = []
    completed = 0
    error_occurred: Exception | None = None

    def process_task(task: dict) -> dict:
        response = call_llm(task["system_prompt"], task["user_prompt"], model)
        return {**task["row_data"], "llm_response": response}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_occurred = e
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            completed += 1
            if on_progress:
                on_progress(completed, len(tasks))

    if error_occurred:
        raise EvaluationError(
            f"Evaluation failed after retries: {error_occurred}"
        ) from error_occurred

    return results


# =============================================================================
# TEMPLATE UTILITIES
# =============================================================================


def render_jinja_template(template_str: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template with the given variables.

    Args:
        template_str: Jinja2 template string
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template string

    Example:
        >>> template = "Hello {{ name }}! You have {{ count }} messages."
        >>> render_jinja_template(template, name="Alice", count=5)
        'Hello Alice! You have 5 messages.'
    """
    template = Template(template_str)
    return template.render(**kwargs)


def format_user_prompt(template: str, row: dict) -> str:
    """
    Format a user prompt template with values from a row.
    Uses Python format strings (not Jinja2).

    Args:
        template: Format string with {column_name} placeholders
        row: Dictionary of column values

    Returns:
        Formatted prompt string

    Example:
        >>> template = "Question: {question}\\nAnswer: {answer}"
        >>> row = {"question": "What is AI?", "answer": "Artificial Intelligence"}
        >>> format_user_prompt(template, row)
        'Question: What is AI?\\nAnswer: Artificial Intelligence'
    """
    return template.format(**row)


# =============================================================================
# DATASET UTILITIES
# =============================================================================


def split_dataset(
    df: pd.DataFrame,
    split_ratio: str,
    stratify_column: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/dev/test sets.

    Args:
        df: Input DataFrame
        split_ratio: Ratio string like "40/40/20"
        stratify_column: Column to stratify on (optional)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, dev_df, test_df)

    Example:
        >>> train, dev, test = split_dataset(df, "40/40/20", stratify_column="label")
    """
    # Parse ratio
    train_pct, dev_pct, test_pct = map(int, split_ratio.split("/"))
    assert train_pct + dev_pct + test_pct == 100, "Split ratios must sum to 100"

    # Calculate sizes
    n = len(df)
    train_size = max(1, int(n * train_pct / 100))
    dev_size = max(1, int(n * dev_pct / 100))

    # Ensure we don't exceed total size
    if train_size + dev_size >= n:
        train_size = max(1, n // 3)
        dev_size = max(1, n // 3)

    # Prepare stratify array
    stratify = df[stratify_column] if stratify_column else None

    try:
        # First split: train vs (dev + test)
        train_df, temp_df = train_test_split(
            df, train_size=train_size, stratify=stratify, random_state=random_state
        )

        # Second split: dev vs test
        remaining = len(temp_df)
        if remaining <= 1:
            # Can't split further, put everything in dev
            dev_df = temp_df
            test_df = pd.DataFrame(columns=df.columns)
        else:
            remaining_stratify = temp_df[stratify_column] if stratify_column else None
            # Calculate relative size for dev within remaining
            relative_dev_size = min(0.99, max(0.01, dev_size / remaining))

            dev_df, test_df = train_test_split(
                temp_df,
                train_size=relative_dev_size,
                stratify=remaining_stratify,
                random_state=random_state,
            )
    except ValueError:
        # Stratification failed (too few samples per class), fall back to random
        train_df, temp_df = train_test_split(
            df, train_size=train_size, random_state=random_state
        )
        remaining = len(temp_df)
        if remaining <= 1:
            # Can't split further, put everything in dev
            dev_df = temp_df
            test_df = pd.DataFrame(columns=df.columns)
        else:
            relative_dev_size = min(0.99, max(0.01, dev_size / remaining))
            dev_df, test_df = train_test_split(
                temp_df, train_size=relative_dev_size, random_state=random_state
            )

    return (
        train_df.reset_index(drop=True),
        dev_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_project_metadata(project_path: str) -> dict:
    """Load project metadata.json."""
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


def save_project_metadata(project_path: str, metadata: dict) -> None:
    """Save project metadata.json."""
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_run_metadata(run_path: str) -> dict:
    """Load run metadata.json."""
    metadata_path = os.path.join(run_path, "metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


def save_run_metadata(run_path: str, metadata: dict) -> None:
    """Save run metadata.json."""
    metadata_path = os.path.join(run_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def list_projects(projects_dir: str = "./projects") -> list[str]:
    """List all project names."""
    if not os.path.exists(projects_dir):
        return []
    return [
        d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))
    ]


def list_runs(project_path: str) -> list[str]:
    """List all run names in a project."""
    return [
        d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))
    ]


def load_prompt_file(file_path: str) -> str:
    """Load a prompt from a text file."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def save_prompt_file(file_path: str, content: str) -> None:
    """Save a prompt to a text file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# SCORE UTILITIES
# =============================================================================


def calculate_score_averages(df: pd.DataFrame, score_columns: list[str]) -> dict[str, float]:
    """
    Calculate average scores for specified columns.

    Args:
        df: DataFrame with score columns
        score_columns: List of column names to average (excluding *_reason columns)

    Returns:
        Dictionary of column_name -> average_value
    """
    averages = {}
    for col in score_columns:
        if col in df.columns and not col.endswith("_reason"):
            averages[col] = df[col].mean()
    return averages


def extract_score_columns(df: pd.DataFrame) -> list[str]:
    """
    Extract score column names (numeric columns that have a paired *_reason column).
    """
    score_cols = []
    for col in df.columns:
        if not col.endswith("_reason") and pd.api.types.is_numeric_dtype(df[col]):
            # Check if there's a corresponding reason column
            if f"{col}_reason" in df.columns:
                score_cols.append(col)
    return score_cols


# =============================================================================
# PATH UTILITIES
# =============================================================================


def get_project_path(project_name: str, projects_dir: str = "./projects") -> str:
    """Get the full path to a project directory."""
    return os.path.join(projects_dir, project_name)


def get_run_path(project_name: str, run_name: str, projects_dir: str = "./projects") -> str:
    """Get the full path to a run directory."""
    return os.path.join(projects_dir, project_name, run_name)


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
