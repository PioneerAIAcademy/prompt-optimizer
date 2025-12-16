# Prompt Optimizer Streamlit App - Implementation Plan

## Overview

A Streamlit application for iteratively optimizing LLM prompts using human-in-the-loop feedback. Users create projects with datasets, evaluate prompts against those datasets, analyze error patterns, and generate optimized prompts based on selected examples.

## Key Design Decisions

1. **LiteLLM** for LLM API calls (supports OpenAI, Anthropic, etc.)
2. **Format strings** (`{column_name}`) for system/user prompts being optimized
3. **Jinja2 templates** for internal prompts (optimizer prompt, error analysis prompt)
4. **Paired keys** for scores: `{"accuracy": 0.8, "accuracy_reason": "..."}`
5. **2-run comparison limit** in the UI
6. **Parallel evaluation with retry** for performance
7. **AgGrid** for interactive data tables
8. **`./projects`** directory for all project data

---

## File Structure

```
prompt-optimizer/
├── requirements.txt
├── app.py                          # Streamlit app (thin orchestration layer)
├── config.py                       # User-customizable functions
├── utils.py                        # Reusable utilities
├── prompt-optimizer-prompt.txt     # Default optimizer prompt (Jinja2)
├── error-analysis-prompt.txt       # Default error analysis prompt (Jinja2)
├── tests/
│   ├── __init__.py
│   ├── test_utils.py               # Unit tests for utils.py
│   ├── test_config.py              # Unit tests for config.py
│   ├── test_e2e.py                 # E2E tests with real API calls
│   └── fixtures/
│       └── sample_dataset.csv      # Synthetic test dataset
└── projects/                       # Created at runtime
    └── {project-name}/
        ├── grader_prompt.txt       # Optional LLM-as-judge prompt
        ├── error-analysis-prompt.txt  # Project-specific (optional, overrides default)
        ├── {dataset}.csv           # Original uploaded dataset
        ├── {dataset}-train.csv
        ├── {dataset}-dev.csv
        ├── {dataset}-test.csv
        ├── metadata.json
        └── {run-name}/
            ├── system_prompt.txt
            ├── user_prompt.txt
            ├── eval-train.csv
            ├── eval-dev.csv
            ├── eval-test.csv
            └── metadata.json
```

---

## Dependencies

### requirements.txt

```
streamlit>=1.28.0
streamlit-aggrid>=0.3.4
pandas>=2.0.0
litellm>=1.0.0
jinja2>=3.1.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
tenacity>=8.2.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## Data Schemas

### Project metadata.json

```json
{
  "project_name": "my-project",
  "dataset_name": "qa-dataset",
  "split_ratio": "40/40/20",
  "eval_model": "openai/gpt-4o-mini",
  "optimizer_model": "openai/gpt-4o",
  "created_at": "2024-12-15T10:30:00Z",
  "stratify_column": "category"
}
```

### Run metadata.json

```json
{
  "run_name": "baseline",
  "created_at": "2024-12-15T10:30:00Z",
  "parent_run": null,
  "scores": {
    "train": {"accuracy": 0.75, "relevance": 0.82},
    "dev": {"accuracy": 0.72, "relevance": 0.80},
    "test": {"accuracy": 0.70, "relevance": 0.78}
  },
  "eval_completed": true,
  "analysis_text": "Common issues: ...",
  "selected_examples": [0, 3, 7, 12]
}
```

---

## Implementation Details

### 1. utils.py

Contains reusable utilities. All functions should be pure or have minimal side effects for easy testing.

```python
"""
Reusable utilities for the prompt optimizer app.
"""

import os
import json
import re
from datetime import datetime
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.model_selection import train_test_split
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_exponential
import litellm


# =============================================================================
# LLM UTILITIES
# =============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0
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
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
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
        temperature=temperature
    )
    return response.choices[0].message.content


def call_llm_parallel(
    tasks: list[dict],
    model: str,
    max_workers: int = 5,
    on_progress: Optional[callable] = None
) -> list[dict]:
    """
    Execute multiple LLM calls in parallel.

    Args:
        tasks: List of dicts with 'system_prompt', 'user_prompt', and 'row_data'
        model: LiteLLM model string
        max_workers: Maximum concurrent requests
        on_progress: Optional callback(completed, total) for progress updates

    Returns:
        List of dicts with original row_data plus 'llm_response' key

    Example:
        >>> tasks = [
        ...     {"system_prompt": "...", "user_prompt": "...", "row_data": {"id": 1}},
        ...     {"system_prompt": "...", "user_prompt": "...", "row_data": {"id": 2}},
        ... ]
        >>> results = call_llm_parallel(tasks, "openai/gpt-4o-mini")
    """
    results = []
    completed = 0

    def process_task(task):
        response = call_llm(
            task["system_prompt"],
            task["user_prompt"],
            model
        )
        return {**task["row_data"], "llm_response": response}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Include error in result
                task = futures[future]
                results.append({**task["row_data"], "llm_response": None, "error": str(e)})

            completed += 1
            if on_progress:
                on_progress(completed, len(tasks))

    return results


# =============================================================================
# TEMPLATE UTILITIES
# =============================================================================

def render_jinja_template(template_str: str, **kwargs) -> str:
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
    stratify_column: Optional[str] = None,
    random_state: int = 42
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
    train_pct, dev_pct, test_pct = map(int, split_ratio.split('/'))
    assert train_pct + dev_pct + test_pct == 100, "Split ratios must sum to 100"

    # Calculate sizes
    n = len(df)
    train_size = int(n * train_pct / 100)
    dev_size = int(n * dev_pct / 100)

    # Prepare stratify array
    stratify = df[stratify_column] if stratify_column else None

    try:
        # First split: train vs (dev + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=stratify,
            random_state=random_state
        )

        # Second split: dev vs test
        remaining_stratify = temp_df[stratify_column] if stratify_column else None
        # Calculate relative size for dev within remaining
        relative_dev_size = dev_size / (n - train_size) if (n - train_size) > 0 else 0.5

        dev_df, test_df = train_test_split(
            temp_df,
            train_size=relative_dev_size,
            stratify=remaining_stratify,
            random_state=random_state
        )
    except ValueError:
        # Stratification failed (too few samples per class), fall back to random
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state
        )
        relative_dev_size = dev_size / (n - train_size) if (n - train_size) > 0 else 0.5
        dev_df, test_df = train_test_split(
            temp_df,
            train_size=relative_dev_size,
            random_state=random_state
        )

    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_project_metadata(project_path: str) -> dict:
    """Load project metadata.json."""
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_project_metadata(project_path: str, metadata: dict) -> None:
    """Save project metadata.json."""
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_run_metadata(run_path: str) -> dict:
    """Load run metadata.json."""
    metadata_path = os.path.join(run_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_run_metadata(run_path: str, metadata: dict) -> None:
    """Save run metadata.json."""
    metadata_path = os.path.join(run_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def list_projects(projects_dir: str = "./projects") -> list[str]:
    """List all project names."""
    if not os.path.exists(projects_dir):
        return []
    return [d for d in os.listdir(projects_dir)
            if os.path.isdir(os.path.join(projects_dir, d))]


def list_runs(project_path: str) -> list[str]:
    """List all run names in a project."""
    return [d for d in os.listdir(project_path)
            if os.path.isdir(os.path.join(project_path, d))]


def load_prompt_file(file_path: str) -> str:
    """Load a prompt from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_prompt_file(file_path: str, content: str) -> None:
    """Save a prompt to a text file."""
    with open(file_path, 'w', encoding='utf-8') as f:
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
        if col in df.columns and col.endswith('_reason') is False:
            averages[col] = df[col].mean()
    return averages


def extract_score_columns(df: pd.DataFrame) -> list[str]:
    """
    Extract score column names (numeric columns that aren't *_reason).
    """
    score_cols = []
    for col in df.columns:
        if not col.endswith('_reason') and pd.api.types.is_numeric_dtype(df[col]):
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
```

---

### 2. config.py

Contains user-customizable functions. Provide working default implementations.

```python
"""
User-customizable functions for the prompt optimizer.

Modify these functions to adapt the optimizer to your specific use case.
The default implementations work for a Q&A grading task.
"""

from typing import Optional
import re
from utils import (
    call_llm,
    call_llm_single_prompt,
    format_user_prompt,
    render_jinja_template
)


def stratify(df) -> Optional[str]:
    """
    Return the column name to use for stratification, or None for random split.

    Modify this to stratify on a column relevant to your dataset.
    For example, stratify on 'category', 'difficulty', or 'label'.

    Args:
        df: The dataset DataFrame

    Returns:
        Column name to stratify on, or None for random split

    Example - stratify on a 'difficulty' column:
        >>> def stratify(df):
        ...     if 'difficulty' in df.columns:
        ...         return 'difficulty'
        ...     return None
    """
    # Default: look for common stratification columns
    for col in ['category', 'label', 'difficulty', 'score', 'rating']:
        if col in df.columns:
            return col
    return None


def eval(
    row: dict,
    system_prompt: str,
    user_prompt_template: str,
    model: str
) -> dict:
    """
    Evaluate a single row by calling the LLM.

    This function:
    1. Formats the user prompt with values from the row
    2. Calls the LLM with the system and user prompts
    3. Extracts structured outputs from the response

    Args:
        row: Dictionary of column values from the dataset
        system_prompt: The system prompt
        user_prompt_template: User prompt template with {column} placeholders
        model: LiteLLM model string

    Returns:
        Dictionary of extracted outputs to add to the row.
        Must include 'llm_response' key with the raw response.

    Example return:
        {
            "llm_response": "The answer is correct because...",
            "extracted_score": 4.5,
            "extracted_reasoning": "The response accurately..."
        }
    """
    # Format the user prompt with row values
    user_prompt = format_user_prompt(user_prompt_template, row)

    # Call the LLM
    response = call_llm(system_prompt, user_prompt, model)

    # Extract structured outputs from response
    # Default implementation: extract score and reasoning
    outputs = {
        "llm_response": response,
    }

    # Try to extract a numeric score (pattern: "Score: X" or "**Score:** X")
    score_match = re.search(r'\*?\*?Score:?\*?\*?\s*(\d+\.?\d*)', response, re.IGNORECASE)
    if score_match:
        outputs["extracted_score"] = float(score_match.group(1))

    # Try to extract reasoning/justification
    reasoning_match = re.search(
        r'(?:Reasoning|Justification|Explanation):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        outputs["extracted_reasoning"] = reasoning_match.group(1).strip()

    return outputs


def score(
    row: dict,
    grader_prompt: Optional[str],
    model: str
) -> dict:
    """
    Score an evaluated row, returning scores with reasons.

    This function can use:
    - LLM-as-a-judge (using grader_prompt)
    - Simple heuristics (comparing to expected output)
    - External APIs

    Args:
        row: Dictionary containing original data plus eval outputs
             (e.g., 'llm_response', 'extracted_score')
        grader_prompt: Optional Jinja2 template for LLM-as-judge grading
        model: LiteLLM model string for grading

    Returns:
        Dictionary of scores with paired reason keys.
        Format: {"score_name": value, "score_name_reason": "explanation"}

    Example return:
        {
            "accuracy": 0.85,
            "accuracy_reason": "Score matches expected within 0.5 points",
            "relevance": 0.90,
            "relevance_reason": "Response addresses the question directly"
        }
    """
    scores = {}

    # If we have an extracted score and expected score, compute accuracy
    if "extracted_score" in row and "expected_score" in row:
        extracted = row["extracted_score"]
        expected = row["expected_score"]

        # Calculate accuracy as 1 - normalized_difference
        # Assuming scores are on a 1-5 scale
        diff = abs(extracted - expected)
        accuracy = max(0.0, 1.0 - (diff / 4.0))

        scores["accuracy"] = round(accuracy, 3)
        scores["accuracy_reason"] = (
            f"Extracted score: {extracted}, Expected: {expected}, "
            f"Difference: {diff:.2f}"
        )

    # If we have a grader prompt, use LLM-as-judge for additional scoring
    if grader_prompt and "llm_response" in row:
        # Render the grader prompt with row data
        formatted_grader = render_jinja_template(grader_prompt, row=row)

        # Call the grading LLM
        grading_response = call_llm_single_prompt(formatted_grader, model, temperature=0.0)

        # Extract relevance score from grading response
        relevance_match = re.search(
            r'(?:Relevance|Quality).*?(\d+\.?\d*)\s*/\s*(\d+)',
            grading_response,
            re.IGNORECASE
        )
        if relevance_match:
            score_val = float(relevance_match.group(1))
            max_val = float(relevance_match.group(2))
            scores["relevance"] = round(score_val / max_val, 3)
            scores["relevance_reason"] = grading_response[:500]  # Truncate for storage

    # Default score if nothing else computed
    if not scores:
        scores["quality"] = 0.5
        scores["quality_reason"] = "No scoring criteria matched"

    return scores


def optimize(
    optimizer_prompt_template: str,
    system_prompt: str,
    user_prompt_template: str,
    examples: list[dict],
    analysis: Optional[str],
    model: str
) -> str:
    """
    Generate an optimized system prompt based on examples and analysis.

    Args:
        optimizer_prompt_template: Jinja2 template for the optimizer
        system_prompt: Current system prompt to optimize
        user_prompt_template: Current user prompt template (for context)
        examples: List of row dictionaries (selected examples with scores)
        analysis: Optional error analysis text
        model: LiteLLM model string for optimization

    Returns:
        Optimized system prompt string
    """
    # Render the optimizer prompt
    formatted_prompt = render_jinja_template(
        optimizer_prompt_template,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        examples=examples,
        analysis=analysis
    )

    # Call the optimizer LLM
    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.7)

    # Extract the optimized prompt from the response
    # Look for content between <optimized_prompt> tags or return full response
    prompt_match = re.search(
        r'<optimized_prompt>(.*?)</optimized_prompt>',
        response,
        re.DOTALL
    )
    if prompt_match:
        return prompt_match.group(1).strip()

    # Alternative: look for content after "Optimized Prompt:" header
    header_match = re.search(
        r'(?:Optimized Prompt|New Prompt|Improved Prompt):\s*\n(.*)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if header_match:
        return header_match.group(1).strip()

    # Return full response if no markers found
    return response.strip()


def analyze(
    rows: list[dict],
    analysis_prompt_template: str,
    model: str
) -> str:
    """
    Analyze rows to identify common error patterns.

    Args:
        rows: List of row dictionaries from eval data
        analysis_prompt_template: Jinja2 template for analysis
        model: LiteLLM model string

    Returns:
        Analysis text describing common error patterns
    """
    # Render the analysis prompt with the rows
    formatted_prompt = render_jinja_template(
        analysis_prompt_template,
        rows=rows
    )

    # Call the LLM
    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.3)

    return response
```

---

### 3. prompt-optimizer-prompt.txt

Default Jinja2 template for prompt optimization.

```
You are an expert prompt engineer. Your task is to improve a system prompt based on evaluation examples.

## Current System Prompt
```
{{ system_prompt }}
```

## User Prompt Template (for context)
```
{{ user_prompt_template }}
```

{% if analysis %}
## Error Analysis
The following analysis summarizes common issues found across the evaluation dataset:

{{ analysis }}
{% endif %}

## Selected Examples
Below are examples from the evaluation. Each shows the input, the LLM's response, and the scores received.

{% for example in examples %}
### Example {{ loop.index }}
**Input Data:**
{% for key, value in example.items() %}
{% if key not in ['llm_response', 'extracted_score', 'extracted_reasoning'] and not key.endswith('_reason') %}
- {{ key }}: {{ value }}
{% endif %}
{% endfor %}

**LLM Response:**
{{ example.llm_response | default("No response") }}

**Scores:**
{% for key, value in example.items() %}
{% if not key.endswith('_reason') and key not in ['llm_response', 'extracted_score', 'extracted_reasoning'] %}
{% if example[key ~ '_reason'] is defined %}
- {{ key }}: {{ value }} ({{ example[key ~ '_reason'] }})
{% endif %}
{% endif %}
{% endfor %}

---
{% endfor %}

## Your Task
Based on the examples above{% if analysis %} and the error analysis{% endif %}, create an improved version of the system prompt that will:
1. Address the specific failure patterns observed
2. Provide clearer instructions where the model struggled
3. Maintain the original intent and format requirements

Output your improved prompt between <optimized_prompt> and </optimized_prompt> tags.

<optimized_prompt>
[Your improved system prompt here]
</optimized_prompt>
```

---

### 4. error-analysis-prompt.txt

Default Jinja2 template for error analysis.

```
You are an expert at analyzing LLM evaluation results to identify patterns and common failure modes.

## Task
Analyze the following evaluation results and identify:
1. Common error patterns or failure modes
2. Types of inputs where the model struggles
3. Systematic issues in the model's responses
4. Specific improvements that could address these issues

## Evaluation Results
{% for row in rows %}
### Example {{ loop.index }}
**Input:**
{% for key, value in row.items() %}
{% if key not in ['llm_response', 'error'] and not key.endswith('_reason') and not key.startswith('extracted_') %}
- {{ key }}: {{ value | truncate(200) }}
{% endif %}
{% endfor %}

**LLM Response:**
{{ row.llm_response | default("No response") | truncate(500) }}

**Scores:**
{% for key, value in row.items() %}
{% if not key.endswith('_reason') and key not in ['llm_response'] and value is number %}
- {{ key }}: {{ value }}{% if row[key ~ '_reason'] is defined %} - {{ row[key ~ '_reason'] | truncate(100) }}{% endif %}
{% endif %}
{% endfor %}

---
{% endfor %}

## Your Analysis
Provide a concise analysis (3-5 paragraphs) of the patterns you observe. Focus on actionable insights that could guide prompt improvement.
```

---

### 5. app.py

Streamlit application. Keep this as thin as possible - delegate to utils.py and config.py.

```python
"""
Prompt Optimizer Streamlit App

A human-in-the-loop tool for iteratively optimizing LLM prompts.
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

import config
from utils import (
    split_dataset,
    list_projects,
    list_runs,
    get_project_path,
    get_run_path,
    load_project_metadata,
    save_project_metadata,
    load_run_metadata,
    save_run_metadata,
    load_prompt_file,
    save_prompt_file,
    ensure_dir,
    calculate_score_averages,
    extract_score_columns,
    call_llm_parallel,
    format_user_prompt
)

PROJECTS_DIR = "./projects"

st.set_page_config(page_title="Prompt Optimizer", layout="wide")
st.title("Prompt Optimizer")


def create_project_tab():
    """Create New Project tab."""
    st.header("Create New Project")

    with st.form("create_project_form"):
        # Project name
        project_name = st.text_input("Project Name", placeholder="my-qa-project")

        # Dataset upload
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

        # Split ratio
        split_ratio = st.selectbox(
            "Split Ratio (Train/Dev/Test)",
            ["40/40/20", "33/33/34", "50/25/25", "60/20/20"],
            index=0
        )

        # Model configuration
        col1, col2 = st.columns(2)
        with col1:
            eval_model = st.text_input(
                "Evaluation Model",
                value="openai/gpt-4o-mini",
                help="LiteLLM model string for evaluation"
            )
        with col2:
            optimizer_model = st.text_input(
                "Optimizer Model",
                value="openai/gpt-4o",
                help="LiteLLM model string for optimization"
            )

        # Prompts
        st.subheader("Baseline Prompts")
        system_prompt = st.text_area(
            "System Prompt",
            height=200,
            placeholder="You are a helpful assistant that..."
        )
        user_prompt = st.text_area(
            "User Prompt Template",
            height=100,
            placeholder="Question: {question}\nContext: {context}",
            help="Use {column_name} placeholders for dataset columns"
        )

        # Optional grader prompt
        st.subheader("Optional: Grading Configuration")
        grader_prompt = st.text_area(
            "Grader Prompt (Jinja2 template, optional)",
            height=150,
            placeholder="Rate the following response...\n{{ row.llm_response }}",
            help="Leave empty to use heuristic scoring only"
        )

        submitted = st.form_submit_button("Create Project")

        if submitted:
            if not project_name:
                st.error("Please enter a project name")
                return
            if uploaded_file is None:
                st.error("Please upload a dataset")
                return
            if not system_prompt or not user_prompt:
                st.error("Please enter both system and user prompts")
                return

            # Create project
            project_path = get_project_path(project_name, PROJECTS_DIR)
            if os.path.exists(project_path):
                st.error(f"Project '{project_name}' already exists")
                return

            ensure_dir(project_path)

            # Load and split dataset
            df = pd.read_csv(uploaded_file)
            dataset_name = os.path.splitext(uploaded_file.name)[0]

            # Get stratify column from config
            stratify_col = config.stratify(df)

            # Split dataset
            train_df, dev_df, test_df = split_dataset(
                df, split_ratio, stratify_column=stratify_col
            )

            # Save datasets
            df.to_csv(os.path.join(project_path, f"{dataset_name}.csv"), index=False)
            train_df.to_csv(os.path.join(project_path, f"{dataset_name}-train.csv"), index=False)
            dev_df.to_csv(os.path.join(project_path, f"{dataset_name}-dev.csv"), index=False)
            test_df.to_csv(os.path.join(project_path, f"{dataset_name}-test.csv"), index=False)

            # Save grader prompt if provided
            if grader_prompt.strip():
                save_prompt_file(os.path.join(project_path, "grader_prompt.txt"), grader_prompt)

            # Save project metadata
            metadata = {
                "project_name": project_name,
                "dataset_name": dataset_name,
                "split_ratio": split_ratio,
                "eval_model": eval_model,
                "optimizer_model": optimizer_model,
                "stratify_column": stratify_col,
                "created_at": datetime.now().isoformat()
            }
            save_project_metadata(project_path, metadata)

            # Create baseline run
            baseline_path = get_run_path(project_name, "baseline", PROJECTS_DIR)
            ensure_dir(baseline_path)

            save_prompt_file(os.path.join(baseline_path, "system_prompt.txt"), system_prompt)
            save_prompt_file(os.path.join(baseline_path, "user_prompt.txt"), user_prompt)

            run_metadata = {
                "run_name": "baseline",
                "created_at": datetime.now().isoformat(),
                "parent_run": None,
                "eval_completed": False
            }
            save_run_metadata(baseline_path, run_metadata)

            st.success(f"Project '{project_name}' created successfully!")
            st.info(f"Dataset split: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test")


def eval_tab():
    """Evaluate tab."""
    st.header("Evaluate")

    # Project selection
    projects = list_projects(PROJECTS_DIR)
    if not projects:
        st.warning("No projects found. Create a project first.")
        return

    project_name = st.selectbox("Select Project", projects)
    project_path = get_project_path(project_name, PROJECTS_DIR)
    project_meta = load_project_metadata(project_path)

    # Run selection
    runs = list_runs(project_path)
    run_name = st.selectbox("Select Run", runs)
    run_path = get_run_path(project_name, run_name, PROJECTS_DIR)

    # Display current prompts
    system_prompt = load_prompt_file(os.path.join(run_path, "system_prompt.txt"))
    user_prompt = load_prompt_file(os.path.join(run_path, "user_prompt.txt"))

    with st.expander("View Prompts"):
        st.subheader("System Prompt")
        st.code(system_prompt)
        st.subheader("User Prompt Template")
        st.code(user_prompt)

    # Evaluate button
    if st.button("Evaluate", type="primary"):
        dataset_name = project_meta["dataset_name"]
        eval_model = project_meta["eval_model"]

        # Load grader prompt if exists
        grader_path = os.path.join(project_path, "grader_prompt.txt")
        grader_prompt = load_prompt_file(grader_path) if os.path.exists(grader_path) else None

        # Process each split
        for split in ["train", "dev", "test"]:
            st.subheader(f"Evaluating {split} split...")

            # Load data
            data_path = os.path.join(project_path, f"{dataset_name}-{split}.csv")
            df = pd.read_csv(data_path)

            # Prepare tasks for parallel execution
            tasks = []
            for _, row in df.iterrows():
                tasks.append({
                    "system_prompt": system_prompt,
                    "user_prompt": format_user_prompt(user_prompt, row.to_dict()),
                    "row_data": row.to_dict()
                })

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(completed, total):
                progress_bar.progress(completed / total)
                status_text.text(f"Processing {completed}/{total} rows...")

            # Run evaluation (calls config.eval internally via parallel helper)
            # Note: For proper integration, modify call_llm_parallel or use a custom loop
            results = []
            for i, task in enumerate(tasks):
                row_dict = task["row_data"]

                # Call config.eval
                eval_outputs = config.eval(
                    row_dict,
                    system_prompt,
                    user_prompt,
                    eval_model
                )

                # Merge eval outputs into row
                row_dict.update(eval_outputs)

                # Call config.score
                score_outputs = config.score(row_dict, grader_prompt, eval_model)
                row_dict.update(score_outputs)

                results.append(row_dict)
                update_progress(i + 1, len(tasks))

            # Save results
            results_df = pd.DataFrame(results)
            eval_path = os.path.join(run_path, f"eval-{split}.csv")
            results_df.to_csv(eval_path, index=False)

            st.success(f"Saved {split} evaluation to {eval_path}")

        # Update run metadata with scores
        run_meta = load_run_metadata(run_path)
        run_meta["eval_completed"] = True
        run_meta["scores"] = {}

        for split in ["train", "dev", "test"]:
            eval_path = os.path.join(run_path, f"eval-{split}.csv")
            df = pd.read_csv(eval_path)
            score_cols = extract_score_columns(df)
            run_meta["scores"][split] = calculate_score_averages(df, score_cols)

        save_run_metadata(run_path, run_meta)
        st.success("Evaluation complete!")


def optimize_tab():
    """Optimize tab with run comparison and example selection."""
    st.header("Optimize")

    # Project selection
    projects = list_projects(PROJECTS_DIR)
    if not projects:
        st.warning("No projects found. Create a project first.")
        return

    project_name = st.selectbox("Select Project", projects, key="opt_project")
    project_path = get_project_path(project_name, PROJECTS_DIR)
    project_meta = load_project_metadata(project_path)

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
                            row[f"{split}_{score_name}"] = round(value, 3) if value else None

            runs_data.append(row)
        except:
            runs_data.append({"run_name": run, "eval_completed": False})

    runs_df = pd.DataFrame(runs_data)

    # Runs table with AgGrid
    st.subheader("Runs")
    gb = GridOptionsBuilder.from_dataframe(runs_df)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("run_name", pinned="left")
    grid_options = gb.build()

    runs_grid = AgGrid(
        runs_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=200
    )

    selected_runs = runs_grid["selected_rows"]

    if selected_runs is not None and len(selected_runs) > 0:
        selected_run = selected_runs[0]["run_name"]
        run_path = get_run_path(project_name, selected_run, PROJECTS_DIR)

        # Check if evaluation exists
        eval_train_path = os.path.join(run_path, "eval-train.csv")
        if not os.path.exists(eval_train_path):
            st.warning(f"Run '{selected_run}' has not been evaluated yet. Go to the Eval tab first.")
            return

        # Load eval data
        eval_df = pd.read_csv(eval_train_path)

        # Compare run selection (optional)
        st.subheader("Compare with another run (optional)")
        other_runs = [r for r in runs if r != selected_run]
        compare_run = st.selectbox("Compare with", ["None"] + other_runs)

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

        # Train dataset table with AgGrid
        st.subheader(f"Training Data - {selected_run}")

        # Analysis section
        st.subheader("Error Analysis (Optional)")

        col1, col2 = st.columns([1, 4])
        with col1:
            score_threshold = st.number_input(
                "Score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Analyze rows with scores below this threshold"
            )
            analyze_all = st.checkbox("Analyze all rows", value=False)

        # Load or get analysis
        analysis_text = st.session_state.get(f"analysis_{project_name}_{selected_run}", "")

        with col2:
            if st.button("Analyze"):
                # Filter rows for analysis
                score_cols = extract_score_columns(eval_df)
                if analyze_all or not score_cols:
                    analysis_rows = eval_df.to_dict('records')
                else:
                    # Filter by first score column
                    primary_score = score_cols[0]
                    mask = eval_df[primary_score] < score_threshold
                    analysis_rows = eval_df[mask].to_dict('records')

                if not analysis_rows:
                    st.warning("No rows match the filter criteria")
                else:
                    # Load analysis prompt
                    project_analysis_path = os.path.join(project_path, "error-analysis-prompt.txt")
                    if os.path.exists(project_analysis_path):
                        analysis_template = load_prompt_file(project_analysis_path)
                    else:
                        analysis_template = load_prompt_file("error-analysis-prompt.txt")

                    with st.spinner(f"Analyzing {len(analysis_rows)} rows..."):
                        analysis_text = config.analyze(
                            analysis_rows,
                            analysis_template,
                            project_meta["optimizer_model"]
                        )

                    st.session_state[f"analysis_{project_name}_{selected_run}"] = analysis_text

        # Editable analysis text
        analysis_text = st.text_area(
            "Analysis (editable)",
            value=analysis_text,
            height=200,
            key=f"analysis_edit_{project_name}_{selected_run}"
        )

        # Data grid for example selection
        st.subheader("Select Examples for Optimization")

        gb2 = GridOptionsBuilder.from_dataframe(eval_df)
        gb2.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb2.configure_default_column(sortable=True, filterable=True, resizable=True)

        # Enable column-specific configs
        for col in eval_df.columns:
            if len(str(eval_df[col].iloc[0] if len(eval_df) > 0 else "")) > 50:
                gb2.configure_column(col, wrapText=True, autoHeight=True, maxWidth=300)

        grid_options2 = gb2.build()

        data_grid = AgGrid(
            eval_df,
            gridOptions=grid_options2,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=False,
            height=400,
            allow_unsafe_jscode=True
        )

        selected_examples = data_grid["selected_rows"]

        # Row detail view
        if selected_examples is not None and len(selected_examples) == 1:
            with st.expander("View Full Row Details"):
                for key, value in selected_examples[0].items():
                    st.markdown(f"**{key}:**")
                    st.text(str(value))
                    st.divider()

        # Optimization
        st.subheader("Generate Optimized Prompt")

        target_run_name = st.text_input(
            "New Run Name",
            value=f"{selected_run}-v2",
            help="Name for the new run with optimized prompt"
        )

        if st.button("Optimize", type="primary"):
            if selected_examples is None or len(selected_examples) == 0:
                st.error("Please select at least one example")
                return

            if not target_run_name:
                st.error("Please enter a run name")
                return

            # Check if run already exists
            target_path = get_run_path(project_name, target_run_name, PROJECTS_DIR)
            if os.path.exists(target_path):
                st.error(f"Run '{target_run_name}' already exists")
                return

            # Load current prompts
            system_prompt = load_prompt_file(os.path.join(run_path, "system_prompt.txt"))
            user_prompt = load_prompt_file(os.path.join(run_path, "user_prompt.txt"))

            # Load optimizer prompt
            optimizer_template = load_prompt_file("prompt-optimizer-prompt.txt")

            # Convert selected examples to list of dicts
            examples = selected_examples if isinstance(selected_examples, list) else selected_examples.to_dict('records')

            with st.spinner("Generating optimized prompt..."):
                optimized_prompt = config.optimize(
                    optimizer_template,
                    system_prompt,
                    user_prompt,
                    examples,
                    analysis_text if analysis_text.strip() else None,
                    project_meta["optimizer_model"]
                )

            # Create new run
            ensure_dir(target_path)
            save_prompt_file(os.path.join(target_path, "system_prompt.txt"), optimized_prompt)
            save_prompt_file(os.path.join(target_path, "user_prompt.txt"), user_prompt)

            run_meta = {
                "run_name": target_run_name,
                "created_at": datetime.now().isoformat(),
                "parent_run": selected_run,
                "eval_completed": False,
                "analysis_text": analysis_text,
                "selected_examples": [i for i, row in enumerate(eval_df.to_dict('records'))
                                      if row in examples]
            }
            save_run_metadata(target_path, run_meta)

            st.success(f"Created new run '{target_run_name}' with optimized prompt!")

            with st.expander("View Optimized Prompt"):
                st.code(optimized_prompt)


# Main app
tab1, tab2, tab3 = st.tabs(["Create Project", "Evaluate", "Optimize"])

with tab1:
    create_project_tab()

with tab2:
    eval_tab()

with tab3:
    optimize_tab()
```

---

### 6. Tests

#### tests/fixtures/sample_dataset.csv

```csv
question,context,answer,expected_score,category
What is photosynthesis?,Plants convert sunlight to energy,Photosynthesis is the process by which plants convert light energy into chemical energy,4.5,biology
What is 2+2?,Basic arithmetic,The answer is 4,5.0,math
Explain gravity,Physics concept,"Gravity is a force that attracts objects toward each other. On Earth, it pulls objects toward the center of the planet.",4.0,physics
What is the capital of France?,Geography question,Paris is the capital of France,5.0,geography
Describe machine learning,AI concept,"Machine learning is a type of AI where computers learn from data without being explicitly programmed.",4.0,technology
What causes rain?,Weather phenomenon,Rain is caused by water vapor condensing in clouds and falling as precipitation,4.5,weather
Who wrote Romeo and Juliet?,Literature question,William Shakespeare wrote Romeo and Juliet,5.0,literature
What is DNA?,Biology concept,DNA is the molecule that carries genetic information in living organisms,4.0,biology
Explain the water cycle,Earth science,The water cycle involves evaporation from oceans and precipitation as rain,3.5,weather
What is an atom?,Chemistry concept,An atom is the smallest unit of matter that retains the properties of an element,4.0,chemistry
```

#### tests/test_utils.py

```python
"""
Unit tests for utils.py
"""

import pytest
import pandas as pd
import os
import tempfile
import json

from utils import (
    format_user_prompt,
    render_jinja_template,
    split_dataset,
    calculate_score_averages,
    extract_score_columns,
    load_prompt_file,
    save_prompt_file,
    load_project_metadata,
    save_project_metadata,
    ensure_dir
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
        return pd.DataFrame({
            "text": [f"text_{i}" for i in range(100)],
            "category": ["A"] * 50 + ["B"] * 50,
            "score": [1, 2, 3, 4, 5] * 20
        })

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
        train, dev, test = split_dataset(
            sample_df, "40/40/20", stratify_column="category"
        )
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
        df = pd.DataFrame({
            "accuracy": [0.8, 0.9, 1.0],
            "accuracy_reason": ["good", "great", "perfect"],
            "relevance": [0.7, 0.8, 0.9]
        })
        averages = calculate_score_averages(df, ["accuracy", "relevance"])
        assert abs(averages["accuracy"] - 0.9) < 0.01
        assert abs(averages["relevance"] - 0.8) < 0.01

    def test_extract_score_columns(self):
        df = pd.DataFrame({
            "text": ["a", "b"],
            "accuracy": [0.8, 0.9],
            "accuracy_reason": ["good", "great"],
            "count": [1, 2]
        })
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
        metadata = {
            "project_name": "test",
            "created_at": "2024-01-01T00:00:00"
        }

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
```

#### tests/test_config.py

```python
"""
Unit tests for config.py

These tests use mocking to avoid actual LLM calls.
"""

import pytest
from unittest.mock import patch, MagicMock

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

    @patch('config.call_llm')
    def test_basic_eval(self, mock_llm):
        mock_llm.return_value = "Analysis complete.\n**Score:** 4.5\nReasoning: The answer is good."

        row = {"question": "What is AI?", "answer": "Artificial Intelligence"}
        result = config.eval(
            row,
            system_prompt="You are a grader.",
            user_prompt_template="Question: {question}\nAnswer: {answer}",
            model="test-model"
        )

        assert "llm_response" in result
        assert result["extracted_score"] == 4.5
        assert "extracted_reasoning" in result

    @patch('config.call_llm')
    def test_missing_score_in_response(self, mock_llm):
        mock_llm.return_value = "This response has no score."

        row = {"question": "Test"}
        result = config.eval(
            row,
            system_prompt="Test",
            user_prompt_template="{question}",
            model="test-model"
        )

        assert "llm_response" in result
        assert "extracted_score" not in result


class TestScore:
    """Tests for score function."""

    def test_accuracy_calculation(self):
        row = {
            "extracted_score": 4.0,
            "expected_score": 4.5,
            "llm_response": "Some response"
        }

        result = config.score(row, grader_prompt=None, model="test-model")

        assert "accuracy" in result
        assert "accuracy_reason" in result
        assert 0.8 < result["accuracy"] < 1.0  # Difference of 0.5 on 1-5 scale

    def test_perfect_score(self):
        row = {
            "extracted_score": 5.0,
            "expected_score": 5.0,
            "llm_response": "Response"
        }

        result = config.score(row, grader_prompt=None, model="test-model")

        assert result["accuracy"] == 1.0

    @patch('config.call_llm_single_prompt')
    def test_with_grader_prompt(self, mock_llm):
        mock_llm.return_value = "Quality: 8/10. Good response."

        row = {
            "llm_response": "Test response",
            "question": "Test question"
        }
        grader_prompt = "Rate this: {{ row.llm_response }}"

        result = config.score(row, grader_prompt=grader_prompt, model="test-model")

        # Should have called the LLM
        mock_llm.assert_called_once()


class TestOptimize:
    """Tests for optimize function."""

    @patch('config.call_llm_single_prompt')
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
            model="test-model"
        )

        assert result == "You are an improved assistant."

    @patch('config.call_llm_single_prompt')
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
            model="test-model"
        )

        assert "better assistant" in result


class TestAnalyze:
    """Tests for analyze function."""

    @patch('config.call_llm_single_prompt')
    def test_basic_analysis(self, mock_llm):
        mock_llm.return_value = "Common issues found: 1. Vague responses 2. Missing details"

        rows = [
            {"question": "Q1", "accuracy": 0.5, "accuracy_reason": "poor"},
            {"question": "Q2", "accuracy": 0.6, "accuracy_reason": "okay"}
        ]

        result = config.analyze(
            rows,
            analysis_prompt_template="Analyze: {% for row in rows %}{{ row.question }}{% endfor %}",
            model="test-model"
        )

        assert "Common issues" in result
        mock_llm.assert_called_once()
```

#### tests/test_e2e.py

```python
"""
End-to-end tests with real API calls.

Run with: pytest tests/test_e2e.py -v -s

Requires:
- OPENAI_API_KEY environment variable set
- Network access to OpenAI API
"""

import pytest
import os
import pandas as pd

# Skip all tests in this file if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class TestLLMCalls:
    """E2E tests for LLM utility functions."""

    def test_call_llm_basic(self):
        from utils import call_llm

        response = call_llm(
            system_prompt="You are a helpful assistant. Respond in exactly one word.",
            user_prompt="What is 2+2?",
            model="openai/gpt-4o-mini"
        )

        assert response is not None
        assert len(response) > 0
        assert "4" in response.lower() or "four" in response.lower()

    def test_call_llm_single_prompt(self):
        from utils import call_llm_single_prompt

        response = call_llm_single_prompt(
            prompt="What is the capital of France? Answer in one word.",
            model="openai/gpt-4o-mini"
        )

        assert "paris" in response.lower()


class TestConfigFunctions:
    """E2E tests for config.py functions."""

    def test_eval_function(self):
        import config

        row = {
            "question": "What is photosynthesis?",
            "context": "Biology topic about plants",
            "answer": "Photosynthesis is how plants make food from sunlight."
        }

        system_prompt = """You are a grading assistant. Evaluate the answer and provide:
1. A score from 1-5
2. Brief reasoning

Format your response as:
**Score:** [number]
Reasoning: [your reasoning]"""

        user_prompt_template = "Question: {question}\nContext: {context}\nStudent Answer: {answer}"

        result = config.eval(
            row,
            system_prompt,
            user_prompt_template,
            model="openai/gpt-4o-mini"
        )

        assert "llm_response" in result
        assert len(result["llm_response"]) > 0
        # Score extraction is best-effort, may or may not succeed

    def test_score_function_without_grader(self):
        import config

        row = {
            "extracted_score": 4.0,
            "expected_score": 4.5,
            "llm_response": "Some response text"
        }

        result = config.score(row, grader_prompt=None, model="openai/gpt-4o-mini")

        assert "accuracy" in result
        assert "accuracy_reason" in result
        assert 0 <= result["accuracy"] <= 1

    def test_analyze_function(self):
        import config
        from utils import load_prompt_file

        rows = [
            {
                "question": "What is AI?",
                "llm_response": "AI is artificial intelligence.",
                "accuracy": 0.6,
                "accuracy_reason": "Too brief"
            },
            {
                "question": "Explain gravity",
                "llm_response": "Gravity pulls things down.",
                "accuracy": 0.5,
                "accuracy_reason": "Incomplete explanation"
            }
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
            model="openai/gpt-4o-mini"
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
            model="openai/gpt-4o-mini"
        )

        assert result is not None
        assert len(result) > 20


class TestFullWorkflow:
    """E2E test of the complete workflow."""

    def test_complete_workflow(self, tmp_path):
        """Test creating a project, evaluating, and optimizing."""
        import config
        from utils import (
            split_dataset,
            save_prompt_file,
            load_prompt_file,
            save_project_metadata,
            ensure_dir,
            calculate_score_averages,
            extract_score_columns
        )

        # 1. Create a small dataset
        df = pd.DataFrame({
            "question": [
                "What is AI?",
                "Explain gravity",
                "What is DNA?"
            ],
            "expected_score": [4.0, 4.5, 4.0],
            "category": ["tech", "physics", "biology"]
        })

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
                model="openai/gpt-4o-mini"
            )
            row_dict.update(eval_result)

            # Score
            score_result = config.score(
                row_dict,
                grader_prompt=None,
                model="openai/gpt-4o-mini"
            )
            row_dict.update(score_result)

            results.append(row_dict)

        results_df = pd.DataFrame(results)

        # Verify we have results
        assert len(results_df) > 0
        assert "llm_response" in results_df.columns

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
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run unit tests (no API calls)
pytest tests/test_utils.py tests/test_config.py -v

# Run E2E tests (requires API key)
export OPENAI_API_KEY="your-key-here"
pytest tests/test_e2e.py -v -s

# Run all tests
pytest tests/ -v
```

---

## Implementation Order

1. **utils.py** - Start here. All functions are independent and testable.
2. **config.py** - Depends on utils.py. Mock LLM calls for unit tests.
3. **prompt-optimizer-prompt.txt** and **error-analysis-prompt.txt** - Static files.
4. **tests/** - Write tests alongside implementation.
5. **app.py** - Last. Thin orchestration layer using utils and config.

---

## Key Testing Strategy

1. **utils.py**: Pure functions, no mocking needed except for LLM calls
2. **config.py**: Mock `call_llm` and `call_llm_single_prompt` for unit tests
3. **E2E tests**: Real API calls with small dataset, requires API key
4. **app.py**: Manual testing via Streamlit; automated UI testing optional

---

## Environment Setup

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Or for Anthropic
echo "ANTHROPIC_API_KEY=your-key-here" >> .env

# Run the app
streamlit run app.py
```

---

## Notes for Developer

1. **Keep app.py thin**: If logic grows complex, move it to utils.py
2. **All user-customizable code in config.py**: Users should only need to modify this file
3. **Paired score keys**: Always return `{"score": val, "score_reason": "..."}` format
4. **Progress callbacks**: Use `on_progress` pattern for long operations
5. **Error handling**: LLM calls have retry logic; surface errors to UI
6. **AgGrid state**: Selection persists across sorts/filters automatically
