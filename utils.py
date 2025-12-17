"""
Reusable utilities for the prompt optimizer app.
"""

import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import litellm
import numpy as np
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


def parse_cluster_json(response: str) -> dict | None:
    """
    Extract and parse JSON from LLM response.

    Handles various formats:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - JSON with surrounding text

    Args:
        response: Raw LLM response

    Returns:
        Parsed dict or None if parsing fails
    """
    if not response or not response.strip():
        return None

    # Try to find JSON in markdown code block (greedy match for nested braces)
    code_block_match = re.search(r'```(?:json)?\s*(\{.+\})\s*```', response, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object containing "clusters" by finding balanced braces
    # Look for the outermost { } that contains "clusters"
    start_idx = response.find('{"clusters"')
    if start_idx == -1:
        start_idx = response.find('{ "clusters"')
    if start_idx != -1:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(response[start_idx:]):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start_idx:start_idx + i + 1])
                    except json.JSONDecodeError:
                        break

    # Try the whole response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None


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
    # Add example IDs before splitting so they're preserved
    df = add_example_ids(df)

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
# STATISTICAL UTILITIES
# =============================================================================


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        scores: List of score values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Example:
        >>> scores = [0.7, 0.8, 0.75, 0.85, 0.9]
        >>> lower, upper = bootstrap_ci(scores)
        >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    scores = np.array(scores)
    if len(scores) == 0:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (float(scores[0]), float(scores[0]))

    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return (float(lower), float(upper))


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000
) -> dict:
    """
    Test if scores_b is significantly different from scores_a using paired bootstrap.

    This is appropriate when comparing the same examples across two runs.

    Args:
        scores_a: Scores from run A (e.g., baseline)
        scores_b: Scores from run B (e.g., new version)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dict with keys:
        - observed_diff: Mean difference (B - A)
        - ci_lower: Lower bound of 95% CI for difference
        - ci_upper: Upper bound of 95% CI for difference
        - significant: True if CI doesn't include zero

    Example:
        >>> baseline = [0.7, 0.6, 0.8, 0.75]
        >>> new_version = [0.8, 0.7, 0.85, 0.8]
        >>> result = paired_bootstrap_test(baseline, new_version)
        >>> print(f"Diff: {result['observed_diff']:.3f}, Significant: {result['significant']}")
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    if len(scores_a) == 0:
        return {
            "observed_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "significant": False
        }

    diffs = scores_b - scores_a
    observed_diff = np.mean(diffs)

    boot_diffs = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(diffs), size=len(diffs), replace=True)
        boot_diffs.append(np.mean(diffs[sample_idx]))

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    significant = ci_lower > 0 or ci_upper < 0

    return {
        "observed_diff": float(observed_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(significant)
    }


def sample_size_guidance(n: int) -> str:
    """
    Return guidance string based on sample size.

    Args:
        n: Number of examples

    Returns:
        Human-readable guidance about statistical power
    """
    if n < 20:
        return "Too few examples for reliable significance testing"
    elif n < 50:
        return "Can detect large effects (>0.15 difference)"
    elif n < 100:
        return "Can detect medium effects (>0.08 difference)"
    else:
        return "Can detect small effects (>0.05 difference)"


def format_score_with_ci(
    scores: list[float],
    ci: float = 0.95
) -> str:
    """
    Format a score with its confidence interval for display.

    Args:
        scores: List of score values
        ci: Confidence level

    Returns:
        Formatted string like "0.78 +/- 0.05"
    """
    if len(scores) == 0:
        return "N/A"

    mean = np.mean(scores)
    if len(scores) == 1:
        return f"{mean:.2f}"

    lower, upper = bootstrap_ci(scores, ci=ci)
    half_width = (upper - lower) / 2
    return f"{mean:.2f} +/- {half_width:.2f}"


# =============================================================================
# EXAMPLE TRACKING UTILITIES
# =============================================================================


def add_example_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add _example_id column if not present.

    Uses existing 'id' column if present, otherwise generates sequential IDs.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with _example_id column added

    Example:
        >>> df = pd.DataFrame({"question": ["Q1", "Q2"]})
        >>> df = add_example_ids(df)
        >>> assert "_example_id" in df.columns
    """
    df = df.copy()
    if "_example_id" not in df.columns:
        if "id" in df.columns:
            df["_example_id"] = df["id"]
        else:
            df["_example_id"] = range(len(df))
    return df


def get_run_lineage(project_path: str, run_name: str) -> list[str]:
    """
    Get ordered list of ancestor runs using parent_run metadata.

    Args:
        project_path: Path to project directory
        run_name: Name of the run to get lineage for

    Returns:
        List of run names from oldest ancestor to current, e.g., ["baseline", "v2", "v3"]
    """
    lineage = []
    current = run_name

    while current:
        lineage.append(current)
        run_path = os.path.join(project_path, current)
        try:
            meta = load_run_metadata(run_path)
            current = meta.get("parent_run")
        except (FileNotFoundError, json.JSONDecodeError):
            break

    return list(reversed(lineage))


def load_example_history(
    project_path: str,
    run_names: list[str],
    split: str,
    score_column: str
) -> pd.DataFrame:
    """
    Load score history for examples across multiple runs.

    Args:
        project_path: Path to project directory
        run_names: List of run names to include
        split: Dataset split ("train", "dev", or "test")
        score_column: Name of score column to track

    Returns:
        DataFrame with columns: _example_id, run1_score, run2_score, ...
    """
    history_data = {}

    for run_name in run_names:
        eval_path = os.path.join(project_path, run_name, f"eval-{split}.csv")
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            if "_example_id" in df.columns and score_column in df.columns:
                for _, row in df.iterrows():
                    example_id = row["_example_id"]
                    if example_id not in history_data:
                        history_data[example_id] = {"_example_id": example_id}
                    history_data[example_id][run_name] = row[score_column]

    return pd.DataFrame(list(history_data.values()))


def detect_regressions(
    history_df: pd.DataFrame,
    run_names: list[str],
    pass_threshold: float = 0.7,
    fail_threshold: float = 0.5
) -> dict:
    """
    Identify examples that regressed, improved, or oscillated.

    Args:
        history_df: DataFrame from load_example_history
        run_names: Ordered list of run names (oldest to newest)
        pass_threshold: Score above which example is considered "passing"
        fail_threshold: Score below which example is considered "failing"

    Returns:
        Dict with keys:
        - improved: List of example IDs that improved
        - regressed: List of example IDs that regressed
        - oscillating: List of example IDs that went up then down (or vice versa)
        - broke: List of example IDs that passed in first run but fail in latest
    """
    if len(run_names) < 2:
        return {"improved": [], "regressed": [], "oscillating": [], "broke": []}

    improved = []
    regressed = []
    oscillating = []
    broke = []

    for _, row in history_df.iterrows():
        example_id = row["_example_id"]
        scores = [row.get(run) for run in run_names if run in row and pd.notna(row.get(run))]

        if len(scores) < 2:
            continue

        first_score = scores[0]
        last_score = scores[-1]

        # Check for broke (passed initially, fails now)
        if first_score >= pass_threshold and last_score < fail_threshold:
            broke.append(example_id)

        # Check for improvement vs regression
        if last_score > first_score + 0.05:
            improved.append(example_id)
        elif last_score < first_score - 0.05:
            regressed.append(example_id)

        # Check for oscillation (changed direction at least once)
        if len(scores) >= 3:
            directions = []
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1] + 0.02:
                    directions.append("up")
                elif scores[i] < scores[i-1] - 0.02:
                    directions.append("down")

            if len(directions) >= 2 and len(set(directions)) > 1:
                oscillating.append(example_id)

    return {
        "improved": improved,
        "regressed": regressed,
        "oscillating": oscillating,
        "broke": broke
    }


def get_trend_label(scores: list[float], run_names: list[str]) -> str:
    """
    Get a human-readable trend label for an example's score history.

    Args:
        scores: List of scores (aligned with run_names)
        run_names: List of run names

    Returns:
        Trend label string
    """
    if len(scores) < 2:
        return ""

    first = scores[0]
    last = scores[-1]

    if len(scores) >= 3:
        # Check for oscillation
        went_up = any(scores[i] > scores[i-1] + 0.05 for i in range(1, len(scores)))
        went_down = any(scores[i] < scores[i-1] - 0.05 for i in range(1, len(scores)))
        if went_up and went_down:
            return "Oscillating"

    if last > first + 0.05:
        return "Improving"
    elif last < first - 0.05:
        if first >= 0.7 and last < 0.5:
            return f"Broke in {run_names[-1]}"
        return f"Regressed from {run_names[0]}"
    else:
        return "Stable"


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
