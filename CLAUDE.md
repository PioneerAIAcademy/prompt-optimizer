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

- **app.py** - Streamlit UI with three tabs: Create Project, Evaluate, Optimize
- **config.py** - User-customizable functions that control the evaluation pipeline. Contains 5 functions: `stratify()`, `eval()`, `score()`, `optimize()`, `analyze()`
- **utils.py** - Reusable utilities for LLM calls, templates, dataset splitting, and file I/O

### Data Flow

1. **Create Project**: Upload CSV dataset → split into train/dev/test → save baseline prompts
2. **Evaluate**: Run prompts against dataset splits → call `config.eval()` per row → call `config.score()` → save results
3. **Optimize**: Select low-scoring examples → optionally run `config.analyze()` → call `config.optimize()` → generate new prompt

### Project File Structure

Projects are stored in `./projects/{project-name}/`:
- `metadata.json` - Project settings (models, split ratio, etc.)
- `{dataset}-train.csv`, `{dataset}-dev.csv`, `{dataset}-test.csv` - Data splits
- `grader_prompt.txt` - Optional LLM-as-judge prompt
- `{run-name}/` - Run directories containing:
  - `system_prompt.txt`, `user_prompt.txt` - Prompts for this run
  - `eval-train.csv`, `eval-dev.csv`, `eval-test.csv` - Evaluation results
  - `metadata.json` - Run metadata and scores

### Customization Points

The main customization point is `config.py`. Modify these functions to adapt to different use cases:

- `stratify(df)` - Returns column name for dataset stratification
- `eval(row, system_prompt, user_prompt_template, model)` - Calls LLM and extracts outputs. Must return dict with `llm_response` key
- `score(row, grader_prompt, model)` - Computes scores. Returns dict with paired keys (e.g., `accuracy` + `accuracy_reason`)
- `optimize(...)` - Generates improved prompts. Extracts result from `<optimized_prompt>` tags
- `analyze(rows, template, model)` - Identifies error patterns

### Template System

- User prompts use Python format strings: `{column_name}` placeholders
- Optimizer and analyzer prompts use Jinja2 templates (`.jinja2` files)
- Per-project template overrides: place files in `projects/{project-name}/`

### Score Convention

Scores must have paired keys for the UI to recognize them:
- `{score_name}` (numeric value)
- `{score_name}_reason` (explanation string)

Example: `accuracy` + `accuracy_reason`
