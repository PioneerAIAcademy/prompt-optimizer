"""
User-customizable functions for the prompt optimizer.

Modify these functions to adapt the optimizer to your specific use case.
The default implementations work for a Q&A grading task.
"""

import re

from utils import (
    ClusterResponse,
    EvalResponse,
    EvaluationError,
    GraderResponse,
    call_llm_single_prompt,
    call_llm_structured,
    format_user_prompt,
    render_jinja_template,
)

# Score scale configuration (default 1-5 scale, range = 4)
# Adjust these if your scoring system uses a different scale
SCORE_SCALE_RANGE = 4.0  # max_score - min_score (e.g., 5 - 1 = 4)


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
    for col in ["category", "label", "difficulty", "score", "rating", "expected_score"]:
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
    Evaluate a single row by calling the LLM with structured output.

    This function:
    1. Formats the user prompt with values from the row
    2. Calls the LLM with structured output to get response, score, and reasoning

    Args:
        row: Dictionary of column values from the dataset
        system_prompt: The system prompt
        user_prompt_template: User prompt template with {column} placeholders
        model: LiteLLM model string

    Returns:
        Dictionary of outputs to add to the row:
        - response: The main LLM response text
        - score: Numeric score if the prompt asks for one (optional)
        - reasoning: Explanation for the score (optional)

    Example return:
        {
            "response": "The answer is correct because...",
            "score": 4.5,
            "reasoning": "The response accurately..."
        }
    """
    # Format the user prompt with row values
    try:
        user_prompt = format_user_prompt(user_prompt_template, row)
    except KeyError as e:
        raise EvaluationError(f"User prompt template references missing column: {e}")

    # Call the LLM with structured output
    try:
        result = call_llm_structured(
            prompt=user_prompt,
            model=model,
            response_model=EvalResponse,
            system_prompt=system_prompt,
        )
    except Exception as e:
        raise EvaluationError(f"LLM call failed in eval: {e}")

    # Build output dict from structured response
    outputs = {
        "response": result.response,
    }

    if result.score is not None:
        outputs["score"] = result.score

    if result.reasoning is not None:
        outputs["reasoning"] = result.reasoning

    return outputs


def score(
    row: dict,
    grader_prompt: str | None,
    model: str,
) -> dict:
    """
    Score an evaluated row, returning scores with reasons.

    This function can use:
    - LLM-as-a-judge (using grader_prompt) with structured output
    - Simple heuristics (comparing to expected output)
    - External APIs

    Args:
        row: Dictionary containing original data plus eval outputs
             (e.g., 'response', 'score')
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
    if "score" in row and "expected_score" in row:
        extracted = row["score"]
        expected = row["expected_score"]

        # Validate types before arithmetic
        try:
            extracted = float(extracted)
            expected = float(expected)
        except (TypeError, ValueError) as e:
            scores["accuracy"] = 0.0
            scores["accuracy_reason"] = f"Type error in scores: {e}"
        else:
            # Calculate accuracy as 1 - normalized_difference
            # Using configurable SCORE_SCALE_RANGE (default 4.0 for 1-5 scale)
            diff = abs(extracted - expected)
            accuracy = max(0.0, 1.0 - (diff / SCORE_SCALE_RANGE))

            scores["accuracy"] = round(accuracy, 3)
            scores["accuracy_reason"] = (
                f"Extracted score: {extracted}, Expected: {expected}, Difference: {diff:.2f}"
            )

    # If we have a grader prompt, use LLM-as-judge with structured output
    if grader_prompt and "response" in row:
        try:
            # Render the grader prompt with row data
            formatted_grader = render_jinja_template(grader_prompt, row=row)

            # Call the grading LLM with structured output
            result = call_llm_structured(
                prompt=formatted_grader,
                model=model,
                response_model=GraderResponse,
                temperature=0.0,
            )

            # Clamp relevance to valid range
            relevance = max(0.0, min(1.0, result.relevance))
            scores["relevance"] = round(relevance, 3)
            scores["relevance_reason"] = result.relevance_reason

        except Exception as e:
            raise EvaluationError(f"Grading LLM call failed: {e}")

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
    target_prompt: str = "system",
) -> str:
    """
    Generate an optimized prompt based on examples and analysis.

    Args:
        optimizer_prompt_template: Jinja2 template for the optimizer
        system_prompt: Current system prompt
        user_prompt_template: Current user prompt template
        examples: List of row dictionaries (selected examples with scores)
        analysis: Optional error analysis text
        model: LiteLLM model string for optimization
        target_prompt: Which prompt to optimize ("system" or "user")

    Returns:
        Optimized prompt string (system or user prompt depending on target_prompt)
    """
    # Render the optimizer prompt
    try:
        formatted_prompt = render_jinja_template(
            optimizer_prompt_template,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            examples=examples,
            analysis=analysis,
            target_prompt=target_prompt,
        )
    except Exception as e:
        raise EvaluationError(f"Failed to render optimizer template: {e}")

    # Call the optimizer LLM
    try:
        response = call_llm_single_prompt(formatted_prompt, model, temperature=0.7)
    except Exception as e:
        raise EvaluationError(f"Optimizer LLM call failed: {e}")

    # Extract the optimized prompt from the response
    # Look for content between <optimized_prompt> tags (case-insensitive, flexible whitespace)
    prompt_match = re.search(
        r"<\s*optimized_prompt\s*>(.*?)<\s*/\s*optimized_prompt\s*>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if prompt_match:
        return prompt_match.group(1).strip()

    # Alternative: look for content after common header formats
    header_match = re.search(
        r"(?:Optimized\s+Prompt|New\s+Prompt|Improved\s+Prompt|Updated\s+Prompt)\s*:\s*\n(.*)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if header_match:
        return header_match.group(1).strip()

    # If no markers found, log a warning and return the response
    # This allows the user to see what the LLM generated even if format was unexpected
    import logging
    logging.warning(
        "Optimizer response did not contain expected markers. "
        "Returning full response. Consider checking the optimizer prompt template."
    )
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

    Raises:
        EvaluationError: If template rendering or LLM call fails
    """
    if not rows:
        return "No rows provided for analysis."

    # Render the analysis prompt with the rows
    try:
        formatted_prompt = render_jinja_template(analysis_prompt_template, rows=rows)
    except Exception as e:
        raise EvaluationError(f"Failed to render analysis template: {e}")

    # Call the LLM
    try:
        response = call_llm_single_prompt(formatted_prompt, model, temperature=0.3)
    except Exception as e:
        raise EvaluationError(f"Analysis LLM call failed: {e}")

    return response


def cluster_failures(
    rows: list[dict],
    clustering_prompt_template: str,
    score_column: str,
    model: str,
    max_clusters: int = 5
) -> dict:
    """
    Cluster failure examples by pattern using LLM with structured output.

    Args:
        rows: List of row dictionaries (failure examples)
        clustering_prompt_template: Jinja2 template for clustering prompt
        score_column: Name of the score column being analyzed
        model: LiteLLM model string
        max_clusters: Maximum number of clusters to request

    Returns:
        Dict with key:
        - clusters: List of cluster dicts with label, description, example_ids

    Raises:
        EvaluationError: If template rendering or LLM call fails
    """
    if not rows:
        return {"clusters": []}

    # Render the prompt
    try:
        formatted_prompt = render_jinja_template(
            clustering_prompt_template,
            failures=rows,
            score_column=score_column,
            max_clusters=max_clusters
        )
    except Exception as e:
        raise EvaluationError(f"Failed to render clustering template: {e}")

    # Call the LLM with structured output
    try:
        result = call_llm_structured(
            prompt=formatted_prompt,
            model=model,
            response_model=ClusterResponse,
            temperature=0.3,
        )
    except Exception as e:
        raise EvaluationError(f"Clustering LLM call failed: {e}")

    # Convert Pydantic models to dicts for compatibility
    return {
        "clusters": [cluster.model_dump() for cluster in result.clusters],
    }
