# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run tests (unit tests don't require API keys)
pytest tests/test_utils.py tests/test_config.py -v

# Run E2E tests (requires API keys in .env)
pytest tests/test_e2e.py -v -s

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_utils.py::TestFormatUserPrompt::test_simple_format -v

# Lint
ruff check .
```

## Architecture

This is a Streamlit app for iteratively optimizing LLM prompts using human feedback. The app uses LiteLLM for LLM calls and supports multiple providers (OpenAI, Anthropic, Azure).

### Core Files

- **app.py** - Streamlit UI with four tabs: Create Project, Evaluate, Optimize, Compare
- **config.py** - User-customizable functions that control the evaluation pipeline. Contains 7 functions: `stratify()`, `primary_score()`, `eval()`, `score()`, `optimize()`, `analyze()`, `cluster_failures()`
- **utils.py** - Reusable utilities for LLM calls, templates, dataset splitting, file I/O, statistical functions, and example tracking
- **clustering-prompt.jinja2** - Default template for LLM-based failure clustering

### Sample Data

The `samples/` directory contains a working example (emotion classification):
- `sampled_emotions.csv` - 301 examples with columns: `text`, `emotion` (joy, anger, sadness, surprise)
- `system_prompt.txt` - Emotion classifier prompt that outputs one of 4 emotion labels
- `user_prompt.txt` - Simple template with `{text}` placeholder

This sample demonstrates a straightforward classification task with exact-match scoring.

### Data Flow

1. **Create Project**: Upload CSV dataset → split into train/dev/test → save baseline prompts
2. **Evaluate**: Run prompts against dataset splits → call `config.eval()` per row → call `config.score()` → save results
3. **Optimize**: Select low-scoring examples → optionally run `config.analyze()` → call `config.optimize()` → generate new prompt

### Project File Structure

Projects are stored in `./projects/{project-name}/`:
- `metadata.json` - Project settings (models, split ratio, `prompt_to_optimize`, etc.)
- `{dataset}-train.csv`, `{dataset}-dev.csv`, `{dataset}-test.csv` - Data splits (with `_example_id` column)
- `grader_prompt.txt` - Optional LLM-as-judge prompt
- `{run-name}/` - Run directories containing:
  - `system_prompt.txt`, `user_prompt.txt` - Prompts for this run
  - `eval-train.csv`, `eval-dev.csv`, `eval-test.csv` - Evaluation results (with `_example_id` column)
  - `metadata.json` - Run metadata, scores, and optional `parent_run` for lineage tracking

### Customization Points

The main customization point is `config.py`. Modify these functions to adapt to different use cases:

- `EVAL_MODEL` - Hardcoded model for evaluation (e.g., `"openai/gpt-4o-mini"`)
- `EVAL_MODEL_PARAMS` - Parameters for eval model (e.g., `{"temperature": 0.0}`)
- `stratify(df)` - Returns column name for dataset stratification
- `primary_score(df)` - Returns the column name of the primary score to use for optimization (filtering failures, clustering, etc.)
- `eval(row, system_prompt, user_prompt_template)` - Calls LLM and extracts outputs. Must return dict with `response` key. Includes retry logic for invalid responses.
- `score(row, grader_prompt)` - Computes scores. Returns dict with paired keys (e.g., `accuracy` + `accuracy_reason`)
- `optimize(..., target_prompt)` - Generates improved prompt (system or user based on `target_prompt`). Extracts result from `<optimized_prompt>` tags
- `analyze(rows, template, model)` - Identifies error patterns
- `cluster_failures(rows, template, score_column, model, max_clusters)` - Groups low-scoring examples by failure pattern using LLM

**Note:** The evaluation model is hardcoded in `config.py` (not configurable via UI) because it's specific to your task. The optimizer model is still configurable at project creation time.

### Template System

- User prompts use Python format strings: `{column_name}` placeholders (system prompts do not support placeholders)
- Optimizer and analyzer prompts use Jinja2 templates (`.jinja2` files)
- Per-project template overrides: place files in `projects/{project-name}/`

**Important:** If your user prompt needs to contain literal curly braces (e.g., JSON examples), you must escape them by doubling: `{{` and `}}`. For example:
```
Return your answer as JSON: {{"score": 5, "reason": "explanation"}}
```
Without escaping, `{score}` would be interpreted as a placeholder for a dataset column.

### Optimization Target

Projects specify which prompt to optimize via `prompt_to_optimize` in metadata:
- `"system"` (default) - Optimize the system prompt; user prompt stays constant
- `"user"` - Optimize the user prompt template; system prompt stays constant

### Score Convention

Scores must have paired keys for the UI to recognize them:
- `{score_name}` (numeric value)
- `{score_name}_reason` (explanation string)

Example: `accuracy` + `accuracy_reason`

### Statistical Features

The app includes statistical utilities for data-driven optimization:

- **Bootstrap CI** - `bootstrap_ci()` computes confidence intervals for score estimates
- **Paired Bootstrap Test** - `paired_bootstrap_test()` tests significance of score differences between runs
- **Sample Size Guidance** - `sample_size_guidance()` advises on statistical power based on test set size
- **Format with CI** - `format_score_with_ci()` displays scores as "0.75 +/- 0.08"

### Example Tracking

Each dataset row gets a unique `_example_id` that persists across runs:

- **add_example_ids()** - Assigns IDs during dataset splitting (uses `id` column if present, else sequential)
- **get_run_lineage()** - Traces parent_run chain to find related runs
- **load_example_history()** - Loads scores for examples across multiple runs
- **detect_regressions()** - Finds examples that broke/improved between runs
- **get_trend_label()** - Labels examples as "Improving", "Regressed", "Stable", or "Oscillating"

### Failure Clustering

The `cluster_failures()` function groups low-scoring examples by failure pattern:

- Uses LLM with `clustering-prompt.jinja2` template and structured output (Pydantic)
- Returns 2-5 clusters with label, description, and example_ids
- UI tracks coverage to ensure diverse example selection for optimization

### Pydantic Metadata Models

Project and run configuration use type-safe Pydantic models:

- **ProjectMetadata** - Project settings: `project_name`, `dataset_name`, `split_ratio`, `optimizer_model`, `stratify_column`, `prompt_to_optimize`, `created_at`, `dataset_source`, `system_prompt_source`, `user_prompt_source`, `grader_prompt_source`
- **RunMetadata** - Run settings: `run_name`, `created_at`, `parent_run`, `eval_completed`, `scores`, `analysis_text`, `selected_examples`, `clustering_results`

### Parallel Evaluation

Evaluation uses `ThreadPoolExecutor(max_workers=8)` for parallel LLM calls, providing significant speedup on multi-row datasets.

### Template Validation

The `validate_jinja_template()` function validates Jinja2 syntax before saving grader prompts, preventing runtime errors from malformed templates.
