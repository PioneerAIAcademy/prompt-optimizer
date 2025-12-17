# Prompt Optimizer - Three Improvements Implementation Plan

## Executive Summary

This document provides complete implementation instructions for three foundational improvements to the prompt optimizer system. These can be implemented by a fresh Claude instance with no prior context.

**The Three Improvements:**
1. **Statistical Guardrails** — Add bootstrap confidence intervals and significance testing (~150 lines)
2. **Per-Example Regression Tracking** — Track individual examples across runs with diff view (~185 lines)
3. **Guided Example Selection** — LLM-based clustering of failures with coverage metrics (~265 lines)

**Total: ~600 lines of new code**

**Implementation Order:** Statistical → Regression → Clustering (each builds on the previous)

---

## CRITICAL: Before You Start

### 1. Update requirements.txt
Add `numpy` to `requirements.txt`:
```
numpy>=1.24.0
```

### 2. Backwards Compatibility
Existing projects created before these changes won't have `_example_id` columns in their eval CSVs. The code handles this gracefully:
- Diff view and clustering show "Example tracking not available" message
- Users can re-run evaluation to get the new columns

### 3. Finding Code Locations
This plan references line numbers as approximations. **Always search for the specific code patterns** rather than relying on line numbers, as the file may have been modified.

### 4. Test-Driven Development (IMPORTANT)
Each Part ends with a **"VERIFY BEFORE PROCEEDING"** section. You MUST:
1. Write the tests FIRST (they're provided in each Part)
2. Implement the code
3. Run the tests to verify correctness
4. Only proceed to the next Part after ALL tests pass

**Do NOT skip testing.** The E2E tests in Part 3 make real API calls to verify the full integration works.

---

## Part 0: Existing Codebase Context

### Project Purpose
This is a Streamlit app for iteratively optimizing LLM prompts using human feedback. Users:
1. Create a project with a CSV dataset
2. Evaluate prompts against train/dev/test splits
3. Select low-scoring examples
4. Generate improved prompts based on those examples
5. Repeat until satisfied

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `app.py` | Streamlit UI with 3 tabs: Create Project, Evaluate, Optimize | ~600 |
| `config.py` | User-customizable functions: `stratify()`, `eval()`, `score()`, `optimize()`, `analyze()` | ~260 |
| `utils.py` | Utilities: LLM calls, templates, dataset splitting, file I/O, score calculations | ~400 |

### Current Data Flow
1. **Create Project**: Upload CSV → `split_dataset()` creates train/dev/test → save baseline prompts
2. **Evaluate**: For each row, call `config.eval()` → `config.score()` → save to `eval-{split}.csv`
3. **Optimize**: Select examples → optionally call `config.analyze()` → call `config.optimize()` → new run

### Current Limitations Being Addressed
- **No statistical rigor**: Raw averages shown with no confidence intervals or significance testing
- **No per-example tracking**: Can't see which specific examples improved/regressed
- **No selection guidance**: Users pick examples arbitrarily from a grid

---

## Part 1: Statistical Guardrails

### Goal
Prevent users from believing score changes that are random noise.

### 1.1 Add Statistical Functions to `utils.py`

Add these imports at the top of `utils.py`:
```python
import numpy as np
```

Add these functions after the existing score utilities section (around line 375):

```python
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
        return (scores[0], scores[0])

    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return (lower, upper)


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
        "significant": significant
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
```

### 1.2 Modify `optimize_tab()` in `app.py`

The runs table is built starting around line 322. Modify it to show confidence intervals.

**Find this code block** (around line 322-344):
```python
# Build runs table data
runs = list_runs(project_path)
runs_data = []
for run in runs:
    run_path = get_run_path(project_name, run, PROJECTS_DIR)
    try:
        run_meta = load_run_metadata(run_path)
        row = {"run_name": run, "eval_completed": run_meta.get("eval_completed", False)}

        # Add score columns
        if "scores" in run_meta:
            for split in ["train", "dev", "test"]:
                if split in run_meta["scores"]:
                    for score_name, value in run_meta["scores"][split].items():
                        row[f"{split}_{score_name}"] = (
                            round(value, 3) if value else None
                        )

        runs_data.append(row)
    except Exception:
        runs_data.append({"run_name": run, "eval_completed": False})
```

**Replace with:**

**PERFORMANCE NOTE:** This loads eval CSVs for each run to compute CIs. With many runs (>10), this could be slow. Consider caching or lazy loading if performance becomes an issue.

```python
# Build runs table data with confidence intervals
runs = list_runs(project_path)
runs_data = []

# We need to load eval data to compute CIs
for run in runs:
    run_path = get_run_path(project_name, run, PROJECTS_DIR)
    try:
        run_meta = load_run_metadata(run_path)
        row = {"run_name": run, "eval_completed": run_meta.get("eval_completed", False)}

        # Load eval data and compute scores with CIs
        if run_meta.get("eval_completed", False):
            for split in ["train", "dev", "test"]:
                eval_path = os.path.join(run_path, f"eval-{split}.csv")
                if os.path.exists(eval_path):
                    eval_df = pd.read_csv(eval_path)
                    score_cols = extract_score_columns(eval_df)
                    for score_col in score_cols:
                        scores = eval_df[score_col].dropna().tolist()
                        if scores:
                            row[f"{split}_{score_col}"] = format_score_with_ci(scores)
                        else:
                            row[f"{split}_{score_col}"] = "N/A"

        runs_data.append(row)
    except Exception:
        runs_data.append({"run_name": run, "eval_completed": False})
```

**Modify the imports at the top of `app.py`.**

Find the existing import block that looks like:
```python
from utils import (
    EvaluationError,
    calculate_score_averages,
    ensure_dir,
    extract_score_columns,
    ...
)
```

Add these new imports to that block:
```python
from utils import (
    EvaluationError,
    calculate_score_averages,
    ensure_dir,
    extract_score_columns,
    get_project_path,
    get_run_path,
    list_projects,
    list_runs,
    load_project_metadata,
    load_prompt_file,
    load_run_metadata,
    save_project_metadata,
    save_prompt_file,
    save_run_metadata,
    split_dataset,
    # NEW: Statistical utilities
    format_score_with_ci,
    sample_size_guidance,
    paired_bootstrap_test,
)
```

### 1.3 Add Sample Size Warning

**Location:** After the runs table AgGrid, INSIDE the block where a run is selected.

Search for the AgGrid component that displays runs, and add this code in the section where `selected_rows` is processed. The variables `selected_run` and `run_path` may already be defined in this scope - if so, use the existing variables rather than redefining them.

Add:

```python
# Sample size warning
if selected_rows is not None and len(selected_rows) > 0:
    if isinstance(selected_rows, pd.DataFrame):
        selected_run = selected_rows.iloc[0]["run_name"]
    else:
        selected_run = selected_rows[0]["run_name"]

    run_path = get_run_path(project_name, selected_run, PROJECTS_DIR)
    test_eval_path = os.path.join(run_path, "eval-test.csv")
    if os.path.exists(test_eval_path):
        test_df = pd.read_csv(test_eval_path)
        n_test = len(test_df)
        guidance = sample_size_guidance(n_test)
        if n_test < 50:
            st.warning(f"Test set has {n_test} examples. {guidance}")
        else:
            st.info(f"Test set has {n_test} examples. {guidance}")
```

### 1.4 Add Significance Testing Between Runs

**Location:** This code goes in the section where two runs are being compared. Search for:
```python
compare_run = st.selectbox(
    "Compare with",
```

Add the significance testing AFTER the comparison DataFrame is loaded (after `compare_df = pd.read_csv(...)`).

**IMPORTANT:** This code assumes `eval_df` (the selected run's evaluation data) is already loaded. Verify this is true at the insertion point.

Add:

```python
if compare_run != "None":
    compare_path = get_run_path(project_name, compare_run, PROJECTS_DIR)
    compare_eval_path = os.path.join(compare_path, "eval-train.csv")
    if os.path.exists(compare_eval_path):
        compare_df = pd.read_csv(compare_eval_path)

        # Add comparison columns
        score_cols = extract_score_columns(eval_df)
        for col in score_cols:
            if col in compare_df.columns:
                eval_df[f"{col}_compare"] = compare_df[col]
                eval_df[f"{col}_diff"] = eval_df[col] - compare_df[col]

        # Show significance test results
        st.subheader("Statistical Comparison")
        for col in score_cols:
            if col in compare_df.columns:
                scores_selected = eval_df[col].dropna().tolist()
                scores_compare = compare_df[col].dropna().tolist()

                # Align by index for paired test
                min_len = min(len(scores_selected), len(scores_compare))
                if min_len > 0:
                    result = paired_bootstrap_test(
                        scores_compare[:min_len],
                        scores_selected[:min_len]
                    )
                    sig_marker = "*" if result["significant"] else "ns"
                    st.write(
                        f"**{col}**: {result['observed_diff']:+.3f} "
                        f"(95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]) "
                        f"**{sig_marker}**"
                    )
```

### 1.5 Add Tests for Statistical Functions

Add to `tests/test_utils.py`. First, ensure these imports are at the top of the test file:

```python
import pytest
import numpy as np
import pandas as pd
from utils import (
    # ... existing imports ...
    bootstrap_ci,
    paired_bootstrap_test,
    sample_size_guidance,
    format_score_with_ci,
)
```

Then add this test class:

```python
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
```

### 1.6 VERIFY BEFORE PROCEEDING

**IMPORTANT: Run these tests before moving to Part 2.**

```bash
# Run Part 1 unit tests
pytest tests/test_utils.py::TestStatisticalUtilities -v

# Expected: All tests pass
# If any fail, fix the implementation before continuing
```

**Quick manual verification:**
1. Start the app: `streamlit run app.py`
2. Go to Optimize tab, select a project with completed evaluation
3. Verify: Runs table shows scores with `+/-` format (e.g., "0.75 +/- 0.08")
4. Verify: Sample size guidance appears below the table

If tests pass and manual verification works, proceed to Part 2.

---

## Part 2: Per-Example Regression Tracking

### Goal
Let users see which specific examples improved, regressed, or oscillated across runs.

### 2.1 Add Example ID Functions to `utils.py`

Add after the statistical utilities section:

```python
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

    first_run = run_names[0]
    last_run = run_names[-1]

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
```

### 2.2 Modify `split_dataset()` to Add Example IDs

In `utils.py`, modify the `split_dataset()` function. Find the function (around line 202) and add this at the beginning:

```python
def split_dataset(
    df: pd.DataFrame,
    split_ratio: str,
    stratify_column: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/dev/test sets.
    ...
    """
    # Add example IDs before splitting so they're preserved
    df = add_example_ids(df)

    # Parse ratio
    train_pct, dev_pct, test_pct = map(int, split_ratio.split("/"))
    # ... rest of function unchanged
```

### 2.3 Add Example Diff View UI in `app.py`

**Location:** Inside `optimize_tab()`, after the run selection logic but BEFORE the "Error Analysis (Optional)" section.

Search for this pattern to find the right location:
```python
# Check if evaluation exists
eval_train_path = os.path.join(run_path, "eval-train.csv")
if not os.path.exists(eval_train_path):
```

Add the diff view code AFTER the `eval_df = pd.read_csv(eval_train_path)` line and BEFORE the "Error Analysis" section.

**Also add these imports** to the existing utils import block in `app.py`:
```python
from utils import (
    # ... existing imports from Part 1 ...
    # NEW: Example tracking utilities
    get_run_lineage,
    load_example_history,
    detect_regressions,
    get_trend_label,
)
```

```python
# Example Performance Diff View
if selected_rows is not None and len(selected_rows) > 0:
    if isinstance(selected_rows, pd.DataFrame):
        selected_run = selected_rows.iloc[0]["run_name"]
    else:
        selected_run = selected_rows[0]["run_name"]

    run_path = get_run_path(project_name, selected_run, PROJECTS_DIR)
    eval_train_path = os.path.join(run_path, "eval-train.csv")

    if os.path.exists(eval_train_path):
        # Get run lineage
        lineage = get_run_lineage(project_path, selected_run)

        if len(lineage) >= 2:
            st.subheader("Example Performance Across Runs")

            # Get primary score column
            eval_df = pd.read_csv(eval_train_path)
            score_cols = extract_score_columns(eval_df)

            if score_cols and "_example_id" in eval_df.columns:
                primary_score = score_cols[0]

                # Load history
                history_df = load_example_history(
                    project_path, lineage, "train", primary_score
                )

                if len(history_df) > 0:
                    # Detect regressions
                    regressions = detect_regressions(history_df, lineage)

                    # Show warnings
                    if regressions["broke"]:
                        st.warning(
                            f"{len(regressions['broke'])} examples that passed in "
                            f"{lineage[0]} now fail. Consider including them in optimization."
                        )

                    if regressions["oscillating"]:
                        st.info(
                            f"{len(regressions['oscillating'])} examples are oscillating "
                            f"(improved then regressed or vice versa)."
                        )

                    # Build display DataFrame
                    display_df = history_df.copy()

                    # Add trend column
                    def compute_trend(row):
                        scores = [row.get(run) for run in lineage if run in row and pd.notna(row.get(run))]
                        return get_trend_label(scores, lineage)

                    display_df["Trend"] = display_df.apply(compute_trend, axis=1)

                    # Reorder columns
                    cols = ["_example_id"] + lineage + ["Trend"]
                    display_df = display_df[[c for c in cols if c in display_df.columns]]

                    # Round scores for display
                    for run in lineage:
                        if run in display_df.columns:
                            display_df[run] = display_df[run].round(2)

                    # Show table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=300
                    )
            else:
                if "_example_id" not in eval_df.columns:
                    st.info(
                        "Example tracking not available. "
                        "Re-run evaluation to enable per-example tracking."
                    )
```

### 2.4 Update Imports in `app.py`

Add to imports:
```python
from utils import (
    # ... existing imports ...
    get_run_lineage,
    load_example_history,
    detect_regressions,
    get_trend_label,
)
```

### 2.5 Add Tests for Regression Tracking

Add these imports to `tests/test_utils.py` (extend the existing import block):

```python
from utils import (
    # ... existing imports ...
    add_example_ids,
    get_trend_label,
    detect_regressions,
    # Note: get_run_lineage and load_example_history require file system,
    # so they're tested in integration tests or manually
)
```

Then add this test class:

```python
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
```

### 2.6 VERIFY BEFORE PROCEEDING

**IMPORTANT: Run these tests before moving to Part 3.**

```bash
# Run Part 2 unit tests
pytest tests/test_utils.py::TestExampleTracking -v

# Expected: All tests pass
# If any fail, fix the implementation before continuing
```

**Quick manual verification:**
1. Create a new test project with a small CSV (5-10 rows)
2. Run evaluation on baseline
3. Create v2 run (optimize from baseline)
4. Re-run evaluation on v2
5. Go to Optimize tab, select v2
6. Verify: "Example Performance Across Runs" section appears
7. Verify: Table shows `_example_id | baseline | v2 | Trend` columns
8. Verify: Trend labels appear (Improving, Stable, etc.)

**Critical check:** Verify `_example_id` column exists in the eval CSV files:
```bash
head -1 projects/*/baseline/eval-train.csv | grep "_example_id"
```

If tests pass and `_example_id` is present, proceed to Part 3.

---

## Part 3: Guided Example Selection with Clustering

### Goal
Help users select diverse, representative failure examples using LLM-based clustering.

### 3.1 Create Clustering Template

Create new file `clustering-prompt.jinja2`:

```jinja2
You are an expert at analyzing LLM evaluation failures to identify patterns.

## Task
Analyze these failure examples and group them into 2-{{ max_clusters }} clusters based on the type of failure or error pattern.

For each cluster, provide:
1. A short label (3-5 words) describing the failure type
2. A one-sentence description of the common pattern
3. List of example IDs that belong to this cluster

## Failure Examples

{% for ex in failures %}
### Example ID: {{ ex._example_id }}
{% for key, value in ex.items() %}
{% if key not in ['llm_response', '_example_id'] and not key.endswith('_reason') and not key.startswith('extracted_') %}
**{{ key }}**: {{ value | string | truncate(200) }}
{% endif %}
{% endfor %}

**LLM Response**: {{ ex.llm_response | default("No response") | truncate(400) }}

**Score**: {{ ex[score_column] | default("N/A") }}
{% if ex[score_column ~ '_reason'] is defined %}
**Reason**: {{ ex[score_column ~ '_reason'] | truncate(150) }}
{% endif %}

---
{% endfor %}

## Output Format
Respond with valid JSON only, no other text:

```json
{
  "clusters": [
    {
      "label": "Short descriptive label",
      "description": "One sentence describing the common failure pattern",
      "example_ids": [1, 2, 3]
    }
  ]
}
```
```

### 3.2 Add Clustering Function to `config.py`

Add this function after the `analyze()` function:

```python
def cluster_failures(
    rows: list[dict],
    clustering_prompt_template: str,
    score_column: str,
    model: str,
    max_clusters: int = 5
) -> dict:
    """
    Cluster failure examples by pattern using LLM.

    Args:
        rows: List of row dictionaries (failure examples)
        clustering_prompt_template: Jinja2 template for clustering prompt
        score_column: Name of the score column being analyzed
        model: LiteLLM model string
        max_clusters: Maximum number of clusters to request

    Returns:
        Dict with keys:
        - clusters: List of cluster dicts with label, description, example_ids
        - raw_response: The raw LLM response (for fallback display)
        - success: Whether parsing succeeded
    """
    from utils import render_jinja_template, call_llm_single_prompt, parse_cluster_json

    # Render the prompt
    formatted_prompt = render_jinja_template(
        clustering_prompt_template,
        failures=rows,
        score_column=score_column,
        max_clusters=max_clusters
    )

    # Call the LLM
    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.3)

    # Try to parse the response
    parsed = parse_cluster_json(response)

    if parsed and "clusters" in parsed:
        return {
            "clusters": parsed["clusters"],
            "raw_response": response,
            "success": True
        }
    else:
        return {
            "clusters": [],
            "raw_response": response,
            "success": False
        }
```

### 3.3 Add JSON Parsing Helper to `utils.py`

**First**, add `re` to the imports at the top of `utils.py` (if not already present):
```python
import re
```

**Then** add this function to the template utilities section (after `format_user_prompt`):

```python
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
```

### 3.4 Add Clustering UI to `app.py`

**Location:** Inside `optimize_tab()`, after the "Error Analysis (Optional)" section.

Search for this pattern to find the right location:
```python
# Editable analysis text
analysis_text = st.text_area(
    "Analysis (editable)",
```

Add the clustering section AFTER that text_area block.

**Note:** The following code assumes these variables are already defined earlier in `optimize_tab()`:
- `project_name`, `project_path`, `project_meta` - defined when project is selected
- `selected_run`, `run_path` - defined when a run is selected
- `eval_df` - loaded from `eval-train.csv`

```python
# Failure Clustering Section
st.subheader("Failure Clustering")

col1, col2 = st.columns([1, 3])
with col1:
    cluster_threshold = st.number_input(
        "Cluster threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Cluster rows with scores below this threshold"
    )

# Session state for clustering results
cluster_key = f"clusters_{project_name}_{selected_run}"

with col2:
    if st.button("Cluster Failures"):
        # Filter rows for clustering
        score_cols = extract_score_columns(eval_df)
        if not score_cols:
            st.error("No score columns found. Run evaluation first.")
        elif score_cols:
            primary_score = score_cols[0]
            mask = eval_df[primary_score] < cluster_threshold
            failure_rows = eval_df[mask].to_dict("records")

            if not failure_rows:
                st.warning("No rows below threshold to cluster")
            elif "_example_id" not in eval_df.columns:
                st.error("Example IDs not found. Re-run evaluation to enable clustering.")
            else:
                # Load clustering template
                project_clustering_path = os.path.join(
                    project_path, "clustering-prompt.jinja2"
                )
                if os.path.exists(project_clustering_path):
                    clustering_template = load_prompt_file(project_clustering_path)
                else:
                    clustering_template = load_prompt_file("clustering-prompt.jinja2")

                with st.spinner(f"Clustering {len(failure_rows)} failures..."):
                    try:
                        result = config.cluster_failures(
                            failure_rows,
                            clustering_template,
                            primary_score,
                            project_meta["optimizer_model"]
                        )
                        st.session_state[cluster_key] = result
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")

# Display clustering results
if cluster_key in st.session_state:
    cluster_result = st.session_state[cluster_key]

    if cluster_result["success"]:
        st.markdown(f"**Found {len(cluster_result['clusters'])} clusters:**")

        for i, cluster in enumerate(cluster_result["clusters"]):
            with st.container():
                st.markdown(f"**Cluster {i+1}: {cluster.get('label', 'Unnamed')}** "
                           f"({len(cluster.get('example_ids', []))} examples)")
                st.markdown(f"_{cluster.get('description', 'No description')}_")
                st.markdown(f"IDs: {', '.join(map(str, cluster.get('example_ids', [])))}")
                st.divider()

        # Coverage tracking
        # NOTE: This relies on the existing AgGrid example selection logic in optimize_tab().
        # Look for where selected_rows from the example grid is processed.
        # You'll need to extract _example_id from selected rows and store in session state.
        # If this mechanism doesn't exist yet, you can skip coverage tracking for now.

        # Example of how to populate selected_example_ids (add this where AgGrid selection is handled):
        # if grid_response and grid_response.selected_rows is not None:
        #     selected_ids = [row["_example_id"] for row in grid_response.selected_rows if "_example_id" in row]
        #     st.session_state["selected_example_ids"] = selected_ids

        selected_ids = set(st.session_state.get("selected_example_ids", []))

        if selected_ids:
            st.markdown("**Selection Coverage:**")
            covered = 0
            for cluster in cluster_result["clusters"]:
                cluster_ids = set(cluster.get("example_ids", []))
                has_selection = bool(cluster_ids & selected_ids)
                if has_selection:
                    covered += 1
                    st.markdown(f"- Cluster '{cluster.get('label', '?')}': ✓ Selected")
                else:
                    st.markdown(f"- Cluster '{cluster.get('label', '?')}': ✗ Not covered")

            total = len(cluster_result["clusters"])
            if covered < total:
                st.warning(f"Coverage: {covered}/{total} clusters. "
                          f"Consider selecting from uncovered clusters.")

        # Auto-select button (shows suggestion - user must manually select in grid)
        # NOTE: This is a simplified implementation. A more advanced version would
        # programmatically update the AgGrid selection, but that requires more complex
        # Streamlit/AgGrid integration. For now, it shows suggested IDs.
        if st.button("Auto-select diverse set"):
            diverse_ids = []
            for cluster in cluster_result["clusters"]:
                ids = cluster.get("example_ids", [])
                if ids:
                    diverse_ids.append(ids[0])  # Take first from each cluster
            st.info(f"Suggested IDs for diverse selection: {diverse_ids}")
            st.caption("Select these IDs manually in the grid below.")

    else:
        # Fallback: show raw response
        st.warning("Clustering couldn't parse structured output. "
                  "You can still select examples manually below.")
        with st.expander("Show raw analysis"):
            st.code(cluster_result["raw_response"])
```

### 3.5 Add Tests for Clustering

**Add to `tests/test_utils.py`** (since `parse_cluster_json` is in `utils.py`):

First, add the import:
```python
from utils import parse_cluster_json
```

Then add this test class:

```python
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
```

**Add E2E test to `tests/test_e2e.py`** (requires API key):

```python
import os
import pytest
from utils import parse_cluster_json
from config import cluster_failures


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires OPENAI_API_KEY or ANTHROPIC_API_KEY"
)
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
        # Use a cheap model for testing
        model = os.getenv("TEST_MODEL", "gpt-4o-mini")

        result = cluster_failures(
            rows=sample_failures,
            clustering_prompt_template=clustering_template,
            score_column="accuracy",
            model=model,
            max_clusters=3
        )

        # Check structure
        assert "clusters" in result
        assert "raw_response" in result
        assert "success" in result

        # If successful, verify cluster structure
        if result["success"]:
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

    def test_cluster_failures_handles_empty_input(self, clustering_template):
        """Test behavior with empty input."""
        model = os.getenv("TEST_MODEL", "gpt-4o-mini")

        result = cluster_failures(
            rows=[],
            clustering_prompt_template=clustering_template,
            score_column="accuracy",
            model=model
        )

        # Should handle gracefully (either empty clusters or failure)
        assert "raw_response" in result
```

### 3.6 VERIFY BEFORE PROCEEDING

**IMPORTANT: Run ALL tests (unit + E2E) to verify Part 3.**

```bash
# Run Part 3 unit tests (JSON parsing)
pytest tests/test_utils.py::TestParseClusterJson -v

# Run Part 3 E2E tests (requires API key)
# Set your API key first: export OPENAI_API_KEY=your-key
pytest tests/test_e2e.py::TestClusterFailuresE2E -v -s

# Expected: All tests pass
# The E2E test makes real API calls - verify you see actual cluster output
```

**Quick manual verification:**
1. Start the app: `streamlit run app.py`
2. Go to Optimize tab, select a project with completed evaluation
3. Set "Cluster threshold" to a value that captures some failures (e.g., 0.7)
4. Click "Cluster Failures" button
5. Verify: Spinner shows "Clustering X failures..."
6. Verify: Clusters appear with labels, descriptions, and IDs
7. Try selecting examples in the grid, verify coverage tracking updates

**Fallback test:**
1. Temporarily break the JSON template (e.g., ask for invalid format)
2. Click "Cluster Failures"
3. Verify: Warning appears with "Show raw analysis" expander (not a crash)

If all tests pass and clustering works in the UI, proceed to Part 4 (final checklist).

---

## Part 4: Final Checklist

### Files Modified
- [ ] `utils.py` - Add statistical functions, example tracking, JSON parsing
- [ ] `app.py` - Add CI display, diff view, clustering UI
- [ ] `config.py` - Add `cluster_failures()` function
- [ ] `clustering-prompt.jinja2` - New file

### New Imports Required

**In `utils.py`** (add to existing imports at top of file):
```python
import re  # For parse_cluster_json
import numpy as np  # For statistical functions
```

**In `app.py`** (complete import block from utils):
```python
from utils import (
    # Existing
    EvaluationError,
    calculate_score_averages,
    ensure_dir,
    extract_score_columns,
    get_project_path,
    get_run_path,
    list_projects,
    list_runs,
    load_project_metadata,
    load_prompt_file,
    load_run_metadata,
    save_project_metadata,
    save_prompt_file,
    save_run_metadata,
    split_dataset,
    # NEW: Statistical utilities (Part 1)
    format_score_with_ci,
    sample_size_guidance,
    paired_bootstrap_test,
    # NEW: Example tracking (Part 2)
    get_run_lineage,
    load_example_history,
    detect_regressions,
    get_trend_label,
    # Note: parse_cluster_json is used by config.py, not app.py
)
```

**In `app.py`** (add config import if not present):
```python
import config
```

**In `tests/test_utils.py`** (add to existing imports):
```python
import pytest
import numpy as np
from utils import (
    # ... existing ...
    bootstrap_ci,
    paired_bootstrap_test,
    sample_size_guidance,
    format_score_with_ci,
    add_example_ids,
    get_trend_label,
    detect_regressions,
    parse_cluster_json,
)
```

### Tests to Run (FINAL VERIFICATION)

**You should have already run tests after each Part. This is the final verification.**

```bash
# 1. Run ALL unit tests for the new features
pytest tests/test_utils.py::TestStatisticalUtilities -v
pytest tests/test_utils.py::TestExampleTracking -v
pytest tests/test_utils.py::TestParseClusterJson -v

# 2. Run E2E tests (requires API key)
export OPENAI_API_KEY=your-key  # or ANTHROPIC_API_KEY
pytest tests/test_e2e.py::TestClusterFailuresE2E -v -s

# 3. Run the full existing test suite to check for regressions
pytest tests/ -v

# ALL TESTS MUST PASS before considering implementation complete
```

**If any test fails:**
1. Read the error message carefully
2. Fix the implementation (not the test, unless the test is wrong)
3. Re-run the failing test
4. Once fixed, re-run ALL tests to ensure no regressions

### Manual Testing Steps (with Success Criteria)

**Setup:**
1. Create a new project with a dataset (use `tests/fixtures/sample_dataset.csv` or similar)
2. Run evaluation on baseline

**Test Statistical Guardrails:**
3. Go to Optimize tab → **SUCCESS:** Runs table shows scores like "0.75 +/- 0.08" (not just "0.75")
4. **SUCCESS:** Below the runs table, see sample size warning: "Test set has X examples. Can detect..."

**Test Regression Tracking:**
5. Create a v2 run (optimize from baseline), then re-evaluate
6. Select v2 in runs table → **SUCCESS:** "Example Performance Across Runs" section appears
7. **SUCCESS:** Table shows columns: `_example_id | baseline | v2 | Trend`
8. **SUCCESS:** Trend column shows labels like "Improving", "Regressed", "Stable"
9. If any examples broke, **SUCCESS:** Warning message appears above the table

**Test Significance Testing:**
10. Select "Compare with" dropdown → choose baseline
11. **SUCCESS:** "Statistical Comparison" section shows: `accuracy: +0.XX (95% CI: [...]) *` or `ns`

**Test Clustering:**
12. Click "Cluster Failures" button → **SUCCESS:** Spinner shows "Clustering X failures..."
13. **SUCCESS:** Clusters display with labels, descriptions, and example IDs
14. Select some examples in the grid → **SUCCESS:** "Selection Coverage" shows which clusters are covered
15. If clustering fails to parse → **SUCCESS:** Warning shows with expandable raw output (not a crash)

**Edge Cases:**
16. Try on a project with no failures (all scores > threshold) → **SUCCESS:** "No rows below threshold" message
17. Try on an old project without `_example_id` → **SUCCESS:** "Example tracking not available" message

---

## Dependency Graph

```
Part 1: Statistical Guardrails
  └── No dependencies, implement first
  └── Adds: bootstrap_ci, paired_bootstrap_test, sample_size_guidance, format_score_with_ci

Part 2: Per-Example Regression Tracking
  └── Depends on: Nothing from Part 1 (can be done in parallel)
  └── Adds: add_example_ids, get_run_lineage, load_example_history, detect_regressions, get_trend_label
  └── CRITICAL: Must modify split_dataset() to call add_example_ids()
  └── This enables Part 3's clustering to use _example_id

Part 3: Guided Example Selection
  └── Depends on: Part 2's _example_id column (requires re-running evaluation)
  └── Adds: parse_cluster_json, cluster_failures, clustering-prompt.jinja2
  └── Uses Part 2's _example_id for cluster membership
```

**Recommended implementation order:** 1 → 2 → 3 (but 1 and 2 can be parallelized)

---

## Appendix: Complete Function Signatures

### utils.py Additions
```python
def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]
def paired_bootstrap_test(scores_a: list[float], scores_b: list[float], n_bootstrap: int = 1000) -> dict
def sample_size_guidance(n: int) -> str
def format_score_with_ci(scores: list[float], ci: float = 0.95) -> str
def add_example_ids(df: pd.DataFrame) -> pd.DataFrame
def get_run_lineage(project_path: str, run_name: str) -> list[str]
def load_example_history(project_path: str, run_names: list[str], split: str, score_column: str) -> pd.DataFrame
def detect_regressions(history_df: pd.DataFrame, run_names: list[str], pass_threshold: float = 0.7, fail_threshold: float = 0.5) -> dict
def get_trend_label(scores: list[float], run_names: list[str]) -> str
def parse_cluster_json(response: str) -> dict | None
```

### config.py Additions
```python
def cluster_failures(rows: list[dict], clustering_prompt_template: str, score_column: str, model: str, max_clusters: int = 5) -> dict
```
