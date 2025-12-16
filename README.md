# Prompt Optimizer

A Streamlit application for iteratively optimizing LLM prompts using human-in-the-loop feedback. Create projects with datasets, evaluate prompts against those datasets, analyze error patterns, and generate optimized prompts based on selected examples.

## Quick Start

### 1. Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure API Keys

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

### Workflow Overview

```
1. Create Project    Upload dataset, define baseline prompts
        |
        v
2. Evaluate          Run prompts against train/dev/test splits
        |
        v
3. Analyze           Identify error patterns in low-scoring examples
        |
        v
4. Optimize          Select examples, generate improved prompts
        |
        v
5. Repeat            Evaluate new prompts, iterate until satisfied
```

### The Three Tabs

#### Tab 1: Create Project

1. **Project Name**: Unique identifier for your project
2. **Dataset**: Upload a CSV file with your evaluation data
3. **Split Ratio**: How to divide data into train/dev/test (default: 40/40/20)
4. **Models**:
   - Evaluation Model: Used for running your prompts (default: `openai/responses/gpt-5-mini`)
   - Optimizer Model: Used for generating improved prompts (default: `anthropic/claude-opus-4-5-20251101`)
5. **Prompts**:
   - System Prompt: Instructions for the LLM
   - User Prompt Template: Template with `{column_name}` placeholders
6. **Grader Prompt (Optional)**: Jinja2 template for LLM-as-judge scoring

#### Tab 2: Evaluate

1. Select a project and run
2. Click "Evaluate" to run prompts against all splits
3. View progress as each row is processed
4. Results are saved as CSV files with scores

#### Tab 3: Optimize

1. Select a project and view the runs table
2. Click on a run to see its training data
3. (Optional) Compare with another run to see score differences
4. (Optional) Click "Analyze" to generate error analysis
5. Select examples that demonstrate failure patterns
6. Enter a new run name and click "Optimize"
7. Review the generated prompt and go to Evaluate tab

## Data Format

### Dataset CSV

Your dataset should be a CSV with columns that match your prompt template placeholders:

```csv
question,context,answer,expected_score,category
"What is AI?","Technology basics","AI is artificial intelligence",4.5,technology
"Explain gravity","Physics","Gravity is a force",4.0,physics
```

### Prompt Templates

**System Prompt** - Plain text instructions:
```
You are a grading assistant. Evaluate answers on a scale of 1-5.

Format your response as:
**Score:** [number]
Reasoning: [your explanation]
```

**User Prompt Template** - Uses Python format strings with `{column_name}`:
```
Question: {question}
Context: {context}
Student Answer: {answer}

Please evaluate this answer.
```

### Score Output Format

The app uses paired keys for scores:
- `accuracy` (numeric value)
- `accuracy_reason` (explanation string)

This pairing is required for scores to be recognized and averaged.

## Customizing for Your Project

The main customization point is `config.py`. This file contains four functions you can modify:

### 1. `stratify(df)` - Dataset Splitting

Controls how the dataset is split into train/dev/test.

```python
def stratify(df) -> str | None:
    """Return column name to stratify on, or None for random split."""
    # Default: looks for common columns
    for col in ["category", "label", "difficulty"]:
        if col in df.columns:
            return col
    return None
```

**Customization Example** - Stratify on difficulty:
```python
def stratify(df) -> str | None:
    if "difficulty" in df.columns:
        return "difficulty"
    return None
```

### 2. `eval(row, system_prompt, user_prompt_template, model)` - Running Prompts

Controls how each row is evaluated.

```python
def eval(row, system_prompt, user_prompt_template, model) -> dict:
    """
    Evaluate a single row.

    Must return dict with 'llm_response' key.
    Can include extracted values like 'extracted_score'.
    """
    user_prompt = format_user_prompt(user_prompt_template, row)
    response = call_llm(system_prompt, user_prompt, model)

    outputs = {"llm_response": response}

    # Extract score from response
    score_match = re.search(r"Score:\s*(\d+\.?\d*)", response)
    if score_match:
        outputs["extracted_score"] = float(score_match.group(1))

    return outputs
```

**Customization Example** - Extract JSON output:
```python
def eval(row, system_prompt, user_prompt_template, model) -> dict:
    user_prompt = format_user_prompt(user_prompt_template, row)
    response = call_llm(system_prompt, user_prompt, model)

    outputs = {"llm_response": response}

    # Parse JSON from response
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            outputs["extracted_answer"] = parsed.get("answer")
            outputs["extracted_confidence"] = parsed.get("confidence")
    except json.JSONDecodeError:
        pass

    return outputs
```

### 3. `score(row, grader_prompt, model)` - Computing Scores

Controls how rows are scored after evaluation.

```python
def score(row, grader_prompt, model) -> dict:
    """
    Score an evaluated row.

    Must return dict with paired keys: {score_name: value, score_name_reason: explanation}
    """
    scores = {}

    # Heuristic scoring
    if "extracted_score" in row and "expected_score" in row:
        diff = abs(row["extracted_score"] - row["expected_score"])
        accuracy = max(0, 1 - diff/4)
        scores["accuracy"] = round(accuracy, 3)
        scores["accuracy_reason"] = f"Diff: {diff:.2f}"

    # LLM-as-judge scoring (if grader_prompt provided)
    if grader_prompt:
        # ... call LLM to grade
        pass

    return scores
```

**Customization Example** - Exact match scoring:
```python
def score(row, grader_prompt, model) -> dict:
    scores = {}

    if "extracted_answer" in row and "expected_answer" in row:
        is_correct = row["extracted_answer"].lower().strip() == row["expected_answer"].lower().strip()
        scores["exact_match"] = 1.0 if is_correct else 0.0
        scores["exact_match_reason"] = "Match" if is_correct else f"Expected: {row['expected_answer']}"

    return scores
```

### 4. `optimize(...)` - Generating New Prompts

Controls how optimized prompts are generated.

```python
def optimize(optimizer_prompt_template, system_prompt, user_prompt_template,
             examples, analysis, model) -> str:
    """Generate an optimized system prompt."""
    formatted_prompt = render_jinja_template(
        optimizer_prompt_template,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        examples=examples,
        analysis=analysis,
    )

    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.7)

    # Extract prompt from <optimized_prompt> tags
    match = re.search(r"<optimized_prompt>(.*?)</optimized_prompt>", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()
```

### 5. `analyze(rows, analysis_prompt_template, model)` - Error Analysis

Controls how error patterns are identified.

```python
def analyze(rows, analysis_prompt_template, model) -> str:
    """Analyze rows to identify error patterns."""
    formatted_prompt = render_jinja_template(analysis_prompt_template, rows=rows)
    return call_llm_single_prompt(formatted_prompt, model, temperature=0.3)
```

## Customizing Prompt Templates

### Optimizer Prompt (`prompt-optimizer-prompt.jinja2`)

This Jinja2 template generates the prompt for optimizing system prompts. Available variables:
- `{{ system_prompt }}` - Current system prompt
- `{{ user_prompt_template }}` - User prompt template
- `{{ examples }}` - List of selected example rows
- `{{ analysis }}` - Error analysis text (optional)

### Error Analysis Prompt (`error-analysis-prompt.jinja2`)

This Jinja2 template generates the prompt for error analysis. Available variables:
- `{{ rows }}` - List of row dictionaries to analyze

### Project-Specific Templates

You can override the default templates per-project by placing files in the project directory:
- `projects/{project-name}/error-analysis-prompt.jinja2`
- `projects/{project-name}/grader_prompt.txt`

## Project Structure

```
prompt-optimizer/
├── app.py                          # Streamlit application
├── config.py                       # User-customizable functions
├── utils.py                        # Reusable utilities
├── requirements.txt                # Python dependencies
├── prompt-optimizer-prompt.jinja2  # Default optimizer prompt
├── error-analysis-prompt.jinja2    # Default analysis prompt
├── tests/                          # Test files
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_e2e.py
│   └── fixtures/
│       └── sample_dataset.csv
└── projects/                       # Created at runtime
    └── {project-name}/
        ├── metadata.json           # Project settings
        ├── grader_prompt.txt       # Optional LLM-as-judge prompt
        ├── {dataset}.csv           # Original dataset
        ├── {dataset}-train.csv     # Training split
        ├── {dataset}-dev.csv       # Development split
        ├── {dataset}-test.csv      # Test split
        └── {run-name}/
            ├── metadata.json       # Run settings and scores
            ├── system_prompt.txt   # System prompt for this run
            ├── user_prompt.txt     # User prompt template
            ├── eval-train.csv      # Evaluation results
            ├── eval-dev.csv
            └── eval-test.csv
```

## Using Different LLM Providers

The app uses [LiteLLM](https://docs.litellm.ai/) which supports many providers:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
```

Model name formats:
- OpenAI: `openai/gpt-4o-mini`, `openai/gpt-4o`
- Anthropic: `anthropic/claude-opus-4-5-20251101`, `anthropic/claude-sonnet-4-20250514`
- Azure: `azure/your-deployment-name`

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run unit tests (no API key needed)
pytest tests/test_utils.py tests/test_config.py -v

# Run E2E tests (requires API keys in .env)
pytest tests/test_e2e.py -v -s

# Run all tests
pytest tests/ -v

# Run linter
ruff check .
```

## Troubleshooting

### "No projects found"
Create a project in the "Create Project" tab first.

### "Run has not been evaluated yet"
Go to the Evaluate tab and run evaluation before trying to optimize.

### API Key Errors
- Check your `.env` file has the correct keys
- Verify the model names match your provider
- Check your API key has sufficient credits/quota

### Empty Score Columns
Scores must have paired keys (`accuracy` + `accuracy_reason`). Check your `score()` function in `config.py`.

### Slow Evaluation
- Reduce dataset size for initial testing
- Use a faster model like `gpt-4o-mini` for evaluation
- The app processes rows sequentially; very large datasets will take time

## Tips for Effective Prompt Optimization

1. **Start Simple**: Begin with a clear, minimal prompt
2. **Use Representative Examples**: Select examples that show diverse failure modes
3. **Analyze Before Optimizing**: The error analysis helps identify patterns
4. **Iterate Incrementally**: Make small changes and re-evaluate
5. **Track Your Runs**: The runs table shows score progression
6. **Compare Runs**: Use the comparison feature to see what improved
7. **Save Good Examples**: Note which examples led to improvements
