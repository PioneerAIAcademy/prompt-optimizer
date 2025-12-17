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
3. Upload your CSV file
4. Set the **Evaluation Model** (the model that runs your prompt): `openai/gpt-4o-mini`
5. Set the **Optimizer Model** (the model that improves your prompt): `anthropic/claude-sonnet-4-20250514`

Now write your initial prompt:

**System Prompt:**
```
You are a grading assistant. Evaluate the student's answer on a scale of 1-5.

Format your response exactly as:
**Score:** [number]
Reasoning: [your explanation]
```

**User Prompt Template:**
```
Question: {question}
Context: {context}
Student Answer: {answer}

Please evaluate this answer.
```

Notice the `{question}`, `{context}`, and `{answer}` placeholders - these get filled in with values from each row of your CSV.

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
2. Select 3-5 examples that show different failure modes (use the checkboxes)
3. Click **Cluster Failures** to group failures by type (optional but helpful)
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
| **Dataset** | Your CSV file with test examples |
| **Split Ratio** | How to divide data (40% train / 40% dev / 20% test is default) |
| **Evaluation Model** | The LLM that runs your prompt (use a fast/cheap model like `gpt-4o-mini`) |
| **Optimizer Model** | The LLM that generates improved prompts (use a smart model like `claude-sonnet`) |
| **System Prompt** | Instructions for the LLM |
| **User Prompt Template** | Template with `{column}` placeholders for your data |

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

1. **Pick diverse failures** - Don't select 5 examples of the same problem
2. **Use clustering** - The cluster feature helps ensure coverage of different failure types
3. **Include edge cases** - Examples that are tricky or unusual
4. **Quality over quantity** - 3-5 well-chosen examples beat 10 random ones

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

### Change How Responses Are Parsed

Edit the `eval()` function:

```python
def eval(row, system_prompt, user_prompt_template, model) -> dict:
    user_prompt = format_user_prompt(user_prompt_template, row)
    response = call_llm(system_prompt, user_prompt, model)

    outputs = {"llm_response": response}

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

---

## Supported LLM Providers

This app uses [LiteLLM](https://docs.litellm.ai/) which supports many providers:

| Provider | Model format | API key env var |
|----------|--------------|-----------------|
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Azure | `azure/your-deployment` | `AZURE_API_KEY`, `AZURE_API_BASE` |

Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
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
- Use a faster model like `gpt-4o-mini` for evaluation
- Reduce dataset size for initial testing
- Processing is sequential, so large datasets take time

---

## Glossary

| Term | Definition |
|------|-----------|
| **Run** | A specific version of your prompt (baseline, v2, v3, etc.) |
| **Split** | Division of data: train (for learning), dev (for tuning), test (for final evaluation) |
| **Evaluation** | Running your prompt against all examples and recording results |
| **Optimization** | Using an AI to generate an improved prompt based on failures |
| **Confidence Interval** | Range showing uncertainty in a score (e.g., 0.75 +/- 0.05) |
| **Regression** | When a change breaks something that was previously working |
| **Clustering** | Grouping similar failures together to understand patterns |

---

## FAQ

**Q: How many examples do I need in my dataset?**

A: At least 20 for basic testing, 50+ for reliable statistics, 100+ for detecting small improvements.

**Q: Should I use the same model for evaluation and optimization?**

A: Not necessarily. Use a fast/cheap model (like `gpt-4o-mini`) for evaluation since you're running it many times. Use a smarter model (like `claude-sonnet`) for optimization since it only runs once per iteration.

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
