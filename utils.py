"""
Reusable utilities for the prompt optimizer app.
"""

import json
import os
from datetime import datetime
from typing import Any, Literal

import litellm
import numpy as np
import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress verbose LiteLLM logging (e.g., "Provider List" messages)
litellm.suppress_debug_info = True


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =============================================================================


class Cluster(BaseModel):
    """A single failure cluster."""

    label: str = Field(..., description="Short descriptive label (3-5 words)")
    description: str = Field(..., description="One sentence describing the pattern")
    example_ids: list[int | str] = Field(
        ..., description="IDs of examples in this cluster"
    )


class ClusterResponse(BaseModel):
    """Response from failure clustering."""

    clusters: list[Cluster]


# =============================================================================
# METADATA MODELS
# =============================================================================


class ProjectMetadata(BaseModel):
    """Project configuration and settings."""

    project_name: str
    dataset_name: str
    split_ratio: str
    optimizer_model: str
    stratify_column: str | None = None
    prompt_to_optimize: Literal["system", "user"] = "system"
    created_at: datetime
    dataset_source: str | None = None
    system_prompt_source: str | None = None
    user_prompt_source: str | None = None
    grader_prompt_source: str | None = None


class RunMetadata(BaseModel):
    """Run configuration and results."""

    run_name: str
    created_at: datetime
    parent_run: str | None = None
    eval_completed: bool = False
    scores: dict[str, dict[str, float]] | None = None
    analysis_text: str | None = None
    selected_examples: list[int] | None = None
    clustering_results: list[dict] | None = None  # List of cluster dicts with label, description, example_ids


# =============================================================================
# LLM UTILITIES
# =============================================================================


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    model_params: dict | None = None,
) -> str:
    """
    Call an LLM using LiteLLM with retry logic.

    Args:
        system_prompt: System message content
        user_prompt: User message content
        model: LiteLLM model string (e.g., "openai/gpt-4o-mini")
        model_params: LLM parameters (e.g., temperature, reasoning_effort)

    Returns:
        The assistant's response text

    Example:
        >>> response = call_llm(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="What is 2+2?",
        ...     model="openai/gpt-4o-mini",
        ...     model_params={"temperature": 0.0}
        ... )
    """
    if model_params is None:
        model_params = {}
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **model_params,
    )
    return response.choices[0].message.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_single_prompt(
    prompt: str,
    model: str,
    model_params: dict | None = None,
) -> str:
    """
    Call an LLM with a single prompt (no system/user split).
    Used for optimizer and analyzer prompts.

    Args:
        prompt: The full prompt text
        model: LiteLLM model string
        model_params: LLM parameters (e.g., temperature, reasoning_effort)

    Returns:
        The assistant's response text
    """
    if model_params is None:
        model_params = {}
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **model_params,
    )
    return response.choices[0].message.content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_structured(
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    system_prompt: str | None = None,
    model_params: dict | None = None,
) -> BaseModel:
    """
    Call an LLM with structured output using response_format.

    Args:
        prompt: The user prompt text
        model: LiteLLM model string
        response_model: Pydantic model class for the expected response
        system_prompt: Optional system message for context
        model_params: LLM parameters (e.g., temperature, reasoning_effort)

    Returns:
        An instance of the response_model with the parsed response

    Example:
        >>> result = call_llm_structured(
        ...     prompt="Evaluate this answer...",
        ...     model="openai/gpt-4o-mini",
        ...     response_model=EvalResponse,
        ...     model_params={"temperature": 0.0}
        ... )
        >>> print(result.score, result.reasoning)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if model_params is None:
        model_params = {}
    response = litellm.completion(
        model=model,
        messages=messages,
        response_format=response_model,
        **model_params,
    )
    return response_model.model_validate_json(response.choices[0].message.content)


class EvaluationError(Exception):
    """Raised when evaluation fails and should stop."""

    pass


# =============================================================================
# TEMPLATE UTILITIES
# =============================================================================


def validate_jinja_template(template_str: str) -> tuple[bool, str | None]:
    """
    Validate Jinja2 template syntax.

    Args:
        template_str: The template string to validate

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.

    Example:
        >>> is_valid, error = validate_jinja_template("Hello {{ name }}!")
        >>> assert is_valid
        >>> is_valid, error = validate_jinja_template("Hello {{ name }")
        >>> assert not is_valid
    """
    from jinja2 import TemplateSyntaxError

    try:
        Template(template_str)
        return (True, None)
    except TemplateSyntaxError as e:
        return (False, f"Template syntax error at line {e.lineno}: {e.message}")


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
    Uses regex substitution to safely handle values containing curly braces.

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
    import re

    def replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in row:
            return str(row[key])
        # Raise KeyError for missing keys (consistent with str.format behavior)
        raise KeyError(key)

    return re.sub(r"\{(\w+)\}", replacer, template)


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
        split_ratio: Ratio string like "40/40/20" (must sum to 100)
        stratify_column: Column to stratify on (optional)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, dev_df, test_df)

    Raises:
        ValueError: If split_ratio is invalid or dataset is empty

    Example:
        >>> train, dev, test = split_dataset(df, "40/40/20", stratify_column="label")
    """
    # Validate input DataFrame
    if len(df) == 0:
        raise ValueError("Cannot split empty DataFrame")

    # Add example IDs before splitting so they're preserved
    df = add_example_ids(df)

    # Validate and parse ratio
    if "/" not in split_ratio:
        raise ValueError(f"Invalid split_ratio format: '{split_ratio}'. Expected format like '40/40/20'")

    parts = split_ratio.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid split_ratio: '{split_ratio}'. Expected 3 values separated by '/'")

    try:
        train_pct, dev_pct, test_pct = map(int, parts)
    except ValueError as e:
        raise ValueError(f"Split ratios must be integers: {e}")

    if train_pct < 0 or dev_pct < 0 or test_pct < 0:
        raise ValueError(f"Split ratios must be non-negative: {split_ratio}")

    if train_pct + dev_pct + test_pct != 100:
        raise ValueError(f"Split ratios must sum to 100, got {train_pct + dev_pct + test_pct}")

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


def load_project_metadata(project_path: str) -> ProjectMetadata:
    """
    Load project metadata.json as Pydantic model.

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValidationError: If metadata is invalid
    """
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)
    return ProjectMetadata.model_validate(data)


def save_project_metadata(project_path: str, metadata: ProjectMetadata) -> None:
    """Save project metadata.json from Pydantic model."""
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def load_run_metadata(run_path: str) -> RunMetadata:
    """
    Load run metadata.json as Pydantic model.

    If metadata.json doesn't exist, creates a default one with run_name
    inferred from the directory name and created_at set to now.

    Raises:
        ValidationError: If metadata is invalid
    """
    metadata_path = os.path.join(run_path, "metadata.json")
    if not os.path.exists(metadata_path):
        # Create default metadata
        run_name = os.path.basename(run_path)
        metadata = RunMetadata(run_name=run_name, created_at=datetime.now())
        save_run_metadata(run_path, metadata)
        return metadata

    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)
    return RunMetadata.model_validate(data)


def save_run_metadata(run_path: str, metadata: RunMetadata) -> None:
    """Save run metadata.json from Pydantic model."""
    metadata_path = os.path.join(run_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def list_projects(projects_dir: str = "./projects") -> list[str]:
    """List all project names."""
    if not os.path.exists(projects_dir):
        return []
    return [
        d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))
    ]


def list_runs(project_path: str) -> list[str]:
    """List all run names in a project. Returns empty list if path doesn't exist."""
    if not os.path.exists(project_path):
        return []
    return [
        d for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))
    ]


def load_prompt_file(file_path: str) -> str:
    """Load a prompt from a text file."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def load_template_with_fallback(project_path: str, template_name: str) -> tuple[str, str]:
    """
    Load a template file, checking project directory first then falling back to default.

    Args:
        project_path: Path to project directory
        template_name: Template filename (e.g., "error-analysis-prompt.jinja2")

    Returns:
        Tuple of (template_content, path_used)

    Example:
        >>> template, path = load_template_with_fallback("./projects/my-project", "grader.jinja2")
    """
    project_template = os.path.join(project_path, template_name)
    if os.path.exists(project_template):
        return load_prompt_file(project_template), project_template
    return load_prompt_file(template_name), template_name


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
    ci: float = 0.95,
    random_state: int | None = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        scores: List of score values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 95%)
        random_state: Random seed for reproducibility (default 42)

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

    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return (float(lower), float(upper))


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    random_state: int | None = 42,
) -> dict:
    """
    Test if scores_b is significantly different from scores_a using paired bootstrap.

    This is appropriate when comparing the same examples across two runs.

    Args:
        scores_a: Scores from run A (e.g., baseline)
        scores_b: Scores from run B (e.g., new version)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility (default 42)

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

    rng = np.random.default_rng(random_state)
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(len(diffs), size=len(diffs), replace=True)
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
            current = meta.parent_run
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

    # Thresholds for detecting changes (use math.isclose-style tolerance)
    IMPROVEMENT_THRESHOLD = 0.05
    OSCILLATION_THRESHOLD = 0.02

    for _, row in history_df.iterrows():
        example_id = row["_example_id"]
        # Filter out NaN values while preserving order
        scores = [row.get(run) for run in run_names if run in row and pd.notna(row.get(run))]

        if len(scores) < 2:
            continue

        first_score = float(scores[0])
        last_score = float(scores[-1])

        # Check for broke (passed initially, fails now) - mutually exclusive with improved/regressed
        is_broke = first_score >= pass_threshold and last_score < fail_threshold
        if is_broke:
            broke.append(example_id)
            # Don't add to regressed since "broke" is more specific
            continue

        # Check for improvement vs regression (mutually exclusive)
        if last_score > first_score + IMPROVEMENT_THRESHOLD:
            improved.append(example_id)
        elif last_score < first_score - IMPROVEMENT_THRESHOLD:
            regressed.append(example_id)

        # Check for oscillation (changed direction at least once)
        # This can overlap with improved/regressed since it's a different metric
        if len(scores) >= 3:
            directions = []
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1] + OSCILLATION_THRESHOLD:
                    directions.append("up")
                elif scores[i] < scores[i-1] - OSCILLATION_THRESHOLD:
                    directions.append("down")

            if len(directions) >= 2 and len(set(directions)) > 1:
                oscillating.append(example_id)

    return {
        "improved": improved,
        "regressed": regressed,
        "oscillating": oscillating,
        "broke": broke
    }


def get_trend_label(
    scores: list[float],
    run_names: list[str],
    improvement_threshold: float = 0.05,
    pass_threshold: float = 0.7,
    fail_threshold: float = 0.5,
) -> str:
    """
    Get a human-readable trend label for an example's score history.

    Args:
        scores: List of scores (aligned with run_names)
        run_names: List of run names
        improvement_threshold: Minimum difference to count as improvement/regression
        pass_threshold: Score above which example is "passing"
        fail_threshold: Score below which example is "failing"

    Returns:
        Trend label string
    """
    if len(scores) < 2:
        return ""

    first = float(scores[0])
    last = float(scores[-1])

    if len(scores) >= 3:
        # Check for oscillation
        went_up = any(scores[i] > scores[i-1] + improvement_threshold for i in range(1, len(scores)))
        went_down = any(scores[i] < scores[i-1] - improvement_threshold for i in range(1, len(scores)))
        if went_up and went_down:
            return "Oscillating"

    if last > first + improvement_threshold:
        return "Improving"
    elif last < first - improvement_threshold:
        if first >= pass_threshold and last < fail_threshold:
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
