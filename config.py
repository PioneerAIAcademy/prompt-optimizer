"""
User-customizable functions for the prompt optimizer.

Modify these functions to adapt the optimizer to your specific use case.
The default implementations work for a Q&A grading task.
"""

import re

from utils import (
    call_llm,
    call_llm_single_prompt,
    format_user_prompt,
    render_jinja_template,
)


def stratify(df) -> str | None:
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
    for col in ["category", "label", "difficulty", "score", "rating"]:
        if col in df.columns:
            return col
    return None


def eval(
    row: dict,
    system_prompt: str,
    user_prompt_template: str,
    model: str,
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
    score_match = re.search(r"\*?\*?Score:?\*?\*?\s*(\d+\.?\d*)", response, re.IGNORECASE)
    if score_match:
        outputs["extracted_score"] = float(score_match.group(1))

    # Try to extract reasoning/justification
    reasoning_match = re.search(
        r"(?:Reasoning|Justification|Explanation):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        outputs["extracted_reasoning"] = reasoning_match.group(1).strip()

    return outputs


def score(
    row: dict,
    grader_prompt: str | None,
    model: str,
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
            f"Extracted score: {extracted}, Expected: {expected}, Difference: {diff:.2f}"
        )

    # If we have a grader prompt, use LLM-as-judge for additional scoring
    if grader_prompt and "llm_response" in row:
        # Render the grader prompt with row data
        formatted_grader = render_jinja_template(grader_prompt, row=row)

        # Call the grading LLM
        grading_response = call_llm_single_prompt(formatted_grader, model, temperature=0.0)

        # Extract relevance score from grading response
        relevance_match = re.search(
            r"(?:Relevance|Quality).*?(\d+\.?\d*)\s*/\s*(\d+)",
            grading_response,
            re.IGNORECASE,
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
    analysis: str | None,
    model: str,
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
        analysis=analysis,
    )

    # Call the optimizer LLM
    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.7)

    # Extract the optimized prompt from the response
    # Look for content between <optimized_prompt> tags or return full response
    prompt_match = re.search(
        r"<optimized_prompt>(.*?)</optimized_prompt>",
        response,
        re.DOTALL,
    )
    if prompt_match:
        return prompt_match.group(1).strip()

    # Alternative: look for content after "Optimized Prompt:" header
    header_match = re.search(
        r"(?:Optimized Prompt|New Prompt|Improved Prompt):\s*\n(.*)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if header_match:
        return header_match.group(1).strip()

    # Return full response if no markers found
    return response.strip()


def analyze(
    rows: list[dict],
    analysis_prompt_template: str,
    model: str,
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
    formatted_prompt = render_jinja_template(analysis_prompt_template, rows=rows)

    # Call the LLM
    response = call_llm_single_prompt(formatted_prompt, model, temperature=0.3)

    return response
