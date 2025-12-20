# Prompt Optimizer

A tool that helps you systematically improve your LLM prompts using real data and feedback.

## What This Tool Does

When you write prompts for LLMs, you often find they work well for some inputs but fail for others. **Prompt Optimizer** helps you:

1. **Test your prompt** against many examples at once
2. **See where it fails** and understand why
3. **Generate improved prompts** based on those failures
4. **Track your progress** as you iterate

Instead of manually testing prompts one at a time, you upload a dataset, run your prompt against all examples, and use the failures to systematically improve.

## The Core Idea

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE OPTIMIZATION LOOP                           │
│                                                                     │
│   1. Write a prompt                                                 │
│          ↓                                                          │
│   2. Run it on your test dataset                                    │
│          ↓                                                          │
│   3. See which examples it got wrong                                │
│          ↓                                                          │
│   4. Feed those failures to an AI to generate a better prompt       │
│          ↓                                                          │
│   5. Repeat until you're happy with the results                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
# Clone and set up
git clone <your-repo>
cd prompt-optimizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
cp .env.example .env
# Edit .env with your OpenAI and/or Anthropic API keys
```

### 2. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Sample Data: Emotion Classification

The `samples/` directory contains a complete working example you can use immediately:

| File | Description |
|------|-------------|
| `sampled_emotions.csv` | 301 examples with columns: `text`, `emotion` (joy, anger, sadness, surprise) |
| `system_prompt.txt` | Emotion classifier prompt that outputs one of 4 emotion labels |
| `user_prompt.txt` | Simple template with `{text}` placeholder |

### Dataset Columns

| Column | Description |
|--------|-------------|
| `text` | The input text to classify |
| `emotion` | The expected emotion label (joy, anger, sadness, or surprise) |

### Using the Sample Data

1. Go to **Create Project** tab
2. Enter `./samples/sampled_emotions.csv` as the Dataset File path
3. Enter `./samples/system_prompt.txt` as the System Prompt File path
4. Enter `./samples/user_prompt.txt` as the User Prompt Template File path
5. Create project and start evaluating

This example demonstrates a straightforward classification task with exact-match scoring. The system prompt instructs the LLM to output exactly one of 4 emotion labels.

---

## Tutorial: Your First Optimization

Let's walk through a complete example. You'll create a grading assistant that evaluates student answers.

### Step 1: Prepare Your Dataset

Create a CSV file with examples you want to test. Each row is one test case:

```csv
question,context,answer,expected_score
"What is photosynthesis?","Biology basics","Plants make food from sunlight",4.5
"What is 2+2?","Math","Four",5.0
"Explain gravity","Physics","Things fall down",3.0
"What is the capital of France?","Geography","Paris",5.0
```

**Key columns:**
- **Input columns** (`question`, `context`, `answer`): Data your prompt will use
- **Expected output** (`expected_score`): What a good response should produce

### Step 2: Create a Project

1. Go to the **Create Project** tab
2. Enter a project name: `grading-assistant`
3. Enter the path to your CSV file (e.g., `./data/grading-examples.csv`)
4. Set the **Evaluation Model** (the model that runs your prompt): `openai/responses/gpt-5-mini`
5. Set the **Optimizer Model** (the model that improves your prompt): `anthropic/claude-opus-4-5-20251101`

Now create your prompt files:

**system_prompt.txt:**
```
You are a grading assistant. Evaluate the student's answer on a scale of 1-5.

Format your response exactly as:
**Score:** [number]
Reasoning: [your explanation]
```

**user_prompt.txt:**
```
Question: {question}
Context: {context}
Student Answer: {answer}

Please evaluate this answer.
```

6. Enter the paths to your prompt files in the form

Notice the `{question}`, `{context}`, and `{answer}` placeholders in the user prompt - these get filled in with values from each row of your CSV.

> **Note:** Only the User Prompt Template supports `{column}` placeholders. The System Prompt is static and passed to the LLM unchanged for every row.

> **Tip:** If your user prompt needs literal curly braces (e.g., JSON examples), escape them by doubling: `{{` and `}}`. For example: `Return JSON: {{"score": 5}}`

Finally, select your **Optimization Target** - whether you want to iteratively improve the System Prompt or the User Prompt Template. The other prompt will remain constant across all optimization runs.

Click **Create Project**.

### Step 3: Evaluate Your Baseline

1. Go to the **Evaluate** tab
2. Select your project and the `baseline` run
3. Click **Evaluate**

The app will run your prompt against every row in your dataset. When done, you'll see results with scores.

### Step 4: Find the Failures

1. Go to the **Optimize** tab
2. Select your project and click on the `baseline` run

You'll see a table with all your examples and their scores. Look for:
- **Low scores**: Examples where your prompt performed poorly
- **Patterns**: Are certain types of questions consistently failing?

### Step 5: Analyze and Improve

1. Click **Analyze** to have the AI identify common error patterns
2. Click **Cluster Failures** to group failures by type
3. Select 1-2 representative examples from each cluster (use the checkboxes)
4. Enter a name for your new run: `v2`
5. Click **Optimize**

The tool will generate an improved prompt based on the failures you selected.

### Step 6: Test the Improved Prompt

1. Go back to the **Evaluate** tab
2. Select your new run (`v2`)
3. Click **Evaluate**
4. Compare the scores to your baseline

### Step 7: Iterate

Keep repeating steps 4-6 until you're satisfied:
- Select new failures from the latest run
- Generate another improved prompt
- Evaluate and compare

---

## Understanding the UI

### Tab 1: Create Project

This is where you set up a new optimization project:

| Setting | What it does |
|---------|-------------|
| **Project Name** | Unique identifier for your project |
| **Dataset File** | Path to your CSV file with test examples |
| **Split Ratio** | How to divide data (40% train / 40% dev / 20% test is default) |
| **Evaluation Model** | The LLM that runs your prompt (use a fast/cheap model like `gpt-5-mini`) |
| **Optimizer Model** | The LLM that generates improved prompts (use a smart model like `claude-opus-4-5`) |
| **System Prompt File** | Path to instructions for the LLM (static, no placeholders) |
| **User Prompt Template File** | Path to template with `{column}` placeholders for your data |
| **Optimization Target** | Which prompt to improve: System Prompt or User Prompt Template |

### Tab 2: Evaluate

Run your prompt against the dataset:
- Select a project and run
- Click **Evaluate** to process all examples
- Results are saved and can be viewed in the Optimize tab

### Tab 3: Optimize

Analyze results and generate improvements:

| Feature | What it does |
|---------|-------------|
| **Runs table** | Shows all your runs with scores (with confidence intervals) |
| **Compare runs** | Statistical comparison between two runs |
| **Analyze** | AI identifies patterns in failures |
| **Cluster Failures** | Groups similar failures together |
| **Example selection** | Pick specific failures to learn from |
| **Optimize** | Generate an improved prompt |

### Tab 4: Compare

Compare prompts between any two runs:

| Feature | What it does |
|---------|-------------|
| **Run selection** | Choose two runs to compare (Run A as base, Run B as compare) |
| **View mode** | Toggle between Diff view (unified diff format) and Side-by-side view |
| **System prompt diff** | Shows changes in system prompt between runs |
| **User prompt diff** | Shows changes in user prompt template between runs |
| **Summary** | Indicates which prompts changed between the selected runs |

---

## How Scoring Works

For the app to track scores, you need to follow a naming convention:

```
score_name     →  the numeric score (e.g., 0.85)
score_name_reason  →  explanation of the score (e.g., "Answer was correct but incomplete")
```

For example, the default setup looks for:
- `accuracy` and `accuracy_reason`
- `relevance` and `relevance_reason`

The app compares extracted values from the LLM's response against expected values in your dataset. You can customize this in `config.py`.

---

## Example: Grading Prompt Evolution

Here's how a prompt might evolve over iterations:

**Baseline (accuracy: 0.65):**
```
You are a grading assistant. Rate the answer from 1-5.
```

**v2 after seeing failures (accuracy: 0.78):**
```
You are a grading assistant. Rate the student's answer from 1-5.

Scoring rubric:
- 5: Complete and accurate
- 4: Mostly correct with minor issues
- 3: Partially correct
- 2: Shows some understanding but has major errors
- 1: Incorrect or irrelevant

Always consider the context provided when evaluating.
```

**v3 after more iteration (accuracy: 0.89):**
```
You are a grading assistant. Rate the student's answer from 1-5.

Scoring rubric:
- 5: Complete, accurate, and well-explained
- 4: Correct answer with adequate explanation
- 3: Partially correct or correct but poorly explained
- 2: Shows understanding but has significant errors
- 1: Incorrect, irrelevant, or no real attempt

Guidelines:
- Compare the answer to the context provided
- Partial credit for showing correct reasoning even if the final answer is wrong
- Deduct points for inaccurate statements, even if the core answer is correct

Format your response exactly as:
**Score:** [number]
Reasoning: [your explanation]
```

---

## Tips for Effective Optimization

### Selecting Good Examples

When choosing failures to learn from:

1. **Cluster first** - Always run "Cluster Failures" before selecting examples. This reveals the distinct failure patterns in your data.
2. **1-2 examples per cluster** - Select 1-2 representative examples from each cluster. If clustering identifies 4 failure modes, select 4-8 examples total.
3. **Pick clear failures** - Choose unambiguous failures over edge cases. The optimizer learns better from clear patterns.
4. **Diversity over volume** - Examples covering different clusters teach more than many examples of the same failure. The goal is coverage, not quantity.

### Writing Good Initial Prompts

1. **Be specific about format** - Tell the LLM exactly how to structure its output
2. **Include examples** - Show what good responses look like
3. **Anticipate failure modes** - Add instructions for common pitfalls

### Interpreting Results

- **Look at confidence intervals**: A score of `0.75 +/- 0.08` means the true score is likely between 0.67 and 0.83
- **Check for regressions**: Sometimes fixing one problem breaks something else
- **Use the compare feature**: Statistical significance testing tells you if improvements are real

---

## Customizing for Your Use Case

The main customization point is `config.py`. This file has functions you can modify:

### Change How Scores Are Extracted

Edit the `score()` function in `config.py`:

```python
def score(row, grader_prompt, model) -> dict:
    # Example: exact match scoring
    scores = {}
    if "extracted_answer" in row and "expected_answer" in row:
        is_correct = row["extracted_answer"].lower() == row["expected_answer"].lower()
        scores["exact_match"] = 1.0 if is_correct else 0.0
        scores["exact_match_reason"] = "Match" if is_correct else "No match"
    return scores
```

### Change Which Score Is Used for Optimization

Edit the `primary_score()` function to specify which score column drives the optimization:

```python
def primary_score(df) -> str | None:
    # Return the score column used for filtering failures and clustering
    return "accuracy"  # or "relevance", "exact_match", etc.
```

This column is used for filtering low-scoring examples, selecting calibration examples, and computing statistics.

### Change How Responses Are Parsed

Edit the `eval()` function:

```python
def eval(row, system_prompt, user_prompt_template, model) -> dict:
    user_prompt = format_user_prompt(user_prompt_template, row)
    response = call_llm(system_prompt, user_prompt, model)

    outputs = {"response": response}

    # Parse JSON from response
    import json
    try:
        parsed = json.loads(response)
        outputs["extracted_answer"] = parsed.get("answer")
    except json.JSONDecodeError:
        pass

    return outputs
```

### Use a Custom Optimizer Prompt

Create `projects/your-project/prompt-optimizer-prompt.jinja2` to override the default optimizer template.

### Change How Dataset Is Split

Edit the `stratify()` function to control stratification:

```python
def stratify(df) -> str | None:
    # Stratify on difficulty to ensure each split has similar distribution
    if 'difficulty' in df.columns:
        return 'difficulty'
    return None  # Random split
```

The default looks for common columns like `category`, `label`, or `difficulty`.

### Change How Prompts Are Optimized

Edit the `optimize()` function to customize how improved prompts are generated:

```python
def optimize(optimizer_prompt_template, system_prompt, user_prompt_template,
             examples, analysis, model) -> str:
    formatted_prompt = render_jinja_template(
        optimizer_prompt_template,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        examples=examples,
        analysis=analysis,
    )

    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.7)

    # Extract from <optimized_prompt> tags, or return full response
    prompt_match = re.search(r"<optimized_prompt>(.*?)</optimized_prompt>",
                             response, re.DOTALL)
    if prompt_match:
        return prompt_match.group(1).strip()
    return response.strip()
```

### Change How Errors Are Analyzed

Edit the `analyze()` function to customize error pattern identification:

```python
def analyze(rows, analysis_prompt_template, model) -> str:
    formatted_prompt = render_jinja_template(analysis_prompt_template, rows=rows)
    return call_llm_single_prompt(formatted_prompt, model, temperature=0.3)
```

You can also override the analysis template by creating `projects/your-project/error-analysis-prompt.jinja2`.

### Change How Failures Are Clustered

Edit the `cluster_failures()` function to customize how low-scoring examples are grouped:

```python
def cluster_failures(rows, clustering_prompt_template, score_column,
                     model, max_clusters=5) -> dict:
    formatted_prompt = render_jinja_template(
        clustering_prompt_template,
        failures=rows,
        score_column=score_column,
        max_clusters=max_clusters
    )

    # Uses structured output with Pydantic ClusterResponse model
    result = call_llm_structured(formatted_prompt, model, ClusterResponse)

    return {
        "clusters": [cluster.model_dump() for cluster in result.clusters],
    }
```

Override the clustering template with `projects/your-project/clustering-prompt.jinja2`.

---

## Project File Structure

```
projects/
└── your-project/
    ├── metadata.json          # Project settings
    ├── grader_prompt.txt      # Optional LLM-as-judge prompt
    ├── dataset.csv            # Your original data
    ├── dataset-train.csv      # Training split (40%)
    ├── dataset-dev.csv        # Development split (40%)
    ├── dataset-test.csv       # Test split (20%)
    └── baseline/              # Your first run
        ├── metadata.json
        ├── system_prompt.txt
        ├── user_prompt.txt
        ├── eval-train.csv     # Results on training data
        ├── eval-dev.csv       # Results on dev data
        └── eval-test.csv      # Results on test data
```

### Creating Custom Runs

You can create your own runs manually by simply creating a new directory with your prompt files:

```
projects/your-project/my-custom-run/
├── system_prompt.txt    # Your system prompt
└── user_prompt.txt      # Your user prompt template
```

That's it. The app will auto-generate `metadata.json` when the run is first loaded. This is useful when you want to:
- Test a hand-crafted prompt variant
- Import prompts from another source
- A/B test specific prompt changes

---

## Supported LLM Providers

This app uses [LiteLLM](https://docs.litellm.ai/) which supports many providers:

| Provider | Model format | API key env var |
|----------|--------------|-----------------|
| OpenAI | `openai/responses/gpt-5-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-opus-4-5-20251101` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-flash-preview-05-20` | `GEMINI_API_KEY` |

Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

---

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run unit tests (no API key needed)
pytest tests/test_utils.py tests/test_config.py -v

# Run E2E tests (requires API keys)
pytest tests/test_e2e.py -v -s

# Run linter
ruff check .
```

---

## Troubleshooting

### "No projects found"
Create a project in the Create Project tab first.

### "Run has not been evaluated yet"
Go to the Evaluate tab and run evaluation before trying to optimize.

### API Key Errors
- Check your `.env` file has the correct keys
- Verify the model names match your provider
- Make sure your API key has credits

### Empty Score Columns
Scores need paired keys (`accuracy` + `accuracy_reason`). Check your `score()` function in `config.py`.

### Slow Evaluation
- Use a faster model like `gpt-5-mini` for evaluation
- Reduce dataset size for initial testing
- Evaluation runs 8 parallel threads by default, but API rate limits may still slow large datasets

---

## Glossary

| Term | Definition |
|------|-----------|
| **Run** | A specific version of your prompt (baseline, v2, v3, etc.) |
| **Split** | Division of data: train (for learning), dev (for tuning), test (for final evaluation) |
| **Evaluation** | Running your prompt against all examples and recording results |
| **Optimization** | Using an AI to generate an improved prompt based on failures |
| **Optimization Target** | Which prompt to improve (system or user); the other stays constant |
| **Confidence Interval** | Range showing uncertainty in a score (e.g., 0.75 +/- 0.05) |
| **Regression** | When a change breaks something that was previously working |
| **Clustering** | Grouping similar failures together to understand patterns |

---

## FAQ

**Q: How many examples do I need in my dataset?**

A: At least 20 for basic testing, 50+ for reliable statistics, 100+ for detecting small improvements.

**Q: Should I use the same model for evaluation and optimization?**

A: Not necessarily. Use a fast/cheap model (like `gpt-5-mini`) for evaluation since you're running it many times. Use a smarter model (like `claude-opus-4-5`) for optimization since it only runs once per iteration.

**Q: How do I know when to stop optimizing?**

A: When either:
- Your scores are high enough for your use case
- Further improvements aren't statistically significant
- You're seeing diminishing returns

**Q: What if my prompt keeps getting worse?**

A: Try:
- Selecting different/better failure examples
- Looking at what changed between versions
- Starting fresh from a different baseline approach

**Q: Can I use this for classification tasks?**

A: Yes. Set up your scoring to compare the extracted classification against expected labels.

**Q: Should I optimize the system prompt or user prompt?**

A: It depends on your use case:
- **Optimize System Prompt** (default): Best for improving instructions, rubrics, and behavioral guidelines. The system prompt is static (no placeholders).
- **Optimize User Prompt Template**: Best when you want to improve how data is presented to the LLM. The user prompt template contains `{column}` placeholders that get filled with dataset values.

Only the user prompt supports placeholders - the system prompt is passed unchanged for every row.
