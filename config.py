"""
User-customizable functions for the prompt optimizer.

Modify these functions to adapt the optimizer to your specific use case.
The default implementations work for a Q&A grading task.
"""

import logging
import re
import time

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

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Parameters for OpenAI Responses API (reasoning models like GPT-5)
# Customize these or add new parameter sets for different model types
RESPONSES_API_PARAMS = {
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_output_tokens": 65536,
    "num_retries": 5,
}


def get_model_params(model: str, temperature: float = 0.0) -> dict:
    """
    Return LLM parameters based on model type.

    Customize this function to support different model providers.
    Add new conditions for models that require special parameters.

    Args:
        model: LiteLLM model string (e.g., "openai/gpt-4o-mini")
        temperature: Sampling temperature (ignored for reasoning models)

    Returns:
        Dictionary of parameters to pass to litellm.completion()

    Example - add support for a custom model:
        >>> def get_model_params(model, temperature=0.0):
        ...     if "my-custom-model" in model:
        ...         return {"custom_param": "value"}
        ...     # ... rest of function
    """
    # OpenAI Responses API (reasoning models like GPT-5)
    if "/responses/" in model:
        return RESPONSES_API_PARAMS.copy()

    # Default: standard models with temperature
    return {"temperature": temperature}


# =============================================================================
# SCORE CONFIGURATION
# =============================================================================

# Score scale configuration (default 1-5 scale, range = 4)
# Adjust these if your scoring system uses a different scale
SCORE_SCALE_RANGE = 4.0  # max_score - min_score (e.g., 5 - 1 = 4)


def stratify(df) -> str | None:
    """
    Return the column name to use for stratification, or None for random split.

    Customize this for your dataset. The stratification column is also used
    as the ground truth reference for scoring pattern analysis.

    Args:
        df: The dataset DataFrame

    Returns:
        Column name to stratify on, or None for random split
    """
    return "expected_score"


def primary_score(df) -> str | None:
    """
    Return the primary score column for optimization thresholds and filtering.

    Customize this for your dataset. This column is used for:
    - Filtering low-scoring examples for analysis
    - Selecting high-scoring calibration examples
    - Computing score statistics and patterns

    Args:
        df: The evaluation DataFrame

    Returns:
        Score column name, or None if no scores
    """
    return "accuracy"


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

    # Call the LLM with structured output, retrying if score is missing
    max_retries = 3
    result = None
    last_error = None

    for attempt in range(max_retries):
        try:
            result = call_llm_structured(
                prompt=user_prompt,
                model=model,
                response_model=EvalResponse,
                system_prompt=system_prompt,
                model_params=get_model_params(model),
            )
            # Success if we got a score
            if result.score is not None:
                break
            # No score - log and retry
            logging.warning(
                f"Eval attempt {attempt + 1}/{max_retries}: no score returned, retrying..."
            )
        except Exception as e:
            last_error = e
            logging.warning(f"Eval attempt {attempt + 1}/{max_retries} failed: {e}")

        # Exponential backoff before retry (skip on last attempt)
        if attempt < max_retries - 1:
            time.sleep(0.5 * (2**attempt))

    # If all retries failed with exceptions, raise error
    if result is None:
        raise EvaluationError(f"LLM call failed in eval after {max_retries} retries: {last_error}")

    # Build output dict from structured response
    outputs = {
        "response": result.response,
    }

    # If we still don't have a score after retries, mark as error
    if result.score is None:
        outputs["_eval_error"] = "No score returned after 3 retries"
        logging.warning(f"Row failed to return score after {max_retries} retries")
    else:
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
                model_params=get_model_params(model),
            )

            # Clamp relevance to valid range
            relevance = max(0.0, min(1.0, result.relevance))
            scores["relevance"] = round(relevance, 3)
            scores["relevance_reason"] = result.relevance_reason

        except Exception as e:
            raise EvaluationError(f"Grading LLM call failed: {e}")

    # Return scores (may be empty if no scoring criteria matched)
    # Rows with _eval_error from eval() are handled separately in aggregation
    return scores


def optimize(
    optimizer_prompt_template: str,
    system_prompt: str,
    user_prompt_template: str,
    examples: list[dict],
    analysis: str | None,
    model: str,
    target_prompt: str = "system",
    high_scoring_examples: list[dict] | None = None,
    score_stats: dict | None = None,
    scoring_pattern: str | None = None,
) -> str:
    """
    Generate an optimized prompt based on examples and analysis.

    Args:
        optimizer_prompt_template: Jinja2 template for the optimizer
        system_prompt: Current system prompt
        user_prompt_template: Current user prompt template
        examples: List of row dictionaries (selected low-scoring examples)
        analysis: Optional error analysis text
        model: LiteLLM model string for optimization
        target_prompt: Which prompt to optimize ("system" or "user")
        high_scoring_examples: Optional list of high-scoring examples for contrast
        score_stats: Optional dict with per-score statistics (mean, n_total, n_low, threshold)
        scoring_pattern: Optional pre-computed scoring pattern analysis text

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
            high_scoring_examples=high_scoring_examples,
            score_stats=score_stats,
            scoring_pattern=scoring_pattern,
        )
    except Exception as e:
        raise EvaluationError(f"Failed to render optimizer template: {e}")

    # Call the optimizer LLM
    try:
        response = call_llm_single_prompt(
            formatted_prompt, model, model_params=get_model_params(model, temperature=0.7)
        )
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
    total_examples: int | None = None,
    score_summary: str | None = None,
    stratify_column: str | None = None,
) -> str:
    """
    Analyze rows to identify common error patterns.

    Args:
        rows: List of row dictionaries from eval data
        analysis_prompt_template: Jinja2 template for analysis
        model: LiteLLM model string
        total_examples: Optional total number of examples in the dataset
        score_summary: Optional pre-computed score summary text
        stratify_column: Optional column name used for stratification/reference

    Returns:
        Analysis text describing common error patterns

    Raises:
        EvaluationError: If template rendering or LLM call fails
    """
    if not rows:
        return "No rows provided for analysis."

    # Render the analysis prompt with the rows
    try:
        formatted_prompt = render_jinja_template(
            analysis_prompt_template,
            rows=rows,
            total_examples=total_examples,
            score_summary=score_summary,
            stratify_column=stratify_column,
        )
    except Exception as e:
        raise EvaluationError(f"Failed to render analysis template: {e}")

    # Call the LLM
    try:
        response = call_llm_single_prompt(
            formatted_prompt, model, model_params=get_model_params(model, temperature=0.3)
        )
    except Exception as e:
        raise EvaluationError(f"Analysis LLM call failed: {e}")

    return response


def cluster_failures(
    rows: list[dict],
    clustering_prompt_template: str,
    score_column: str,
    model: str,
    max_clusters: int = 5,
    total_examples: int | None = None,
    threshold: float | None = None,
    stratify_column: str | None = None,
) -> dict:
    """
    Cluster failure examples by pattern using LLM with structured output.

    Args:
        rows: List of row dictionaries (failure examples)
        clustering_prompt_template: Jinja2 template for clustering prompt
        score_column: Name of the score column being analyzed
        model: LiteLLM model string
        max_clusters: Maximum number of clusters to request
        total_examples: Optional total number of examples in the dataset
        threshold: Optional score threshold used to filter failures
        stratify_column: Optional column name used for stratification/reference

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
            max_clusters=max_clusters,
            total_examples=total_examples,
            threshold=threshold,
            stratify_column=stratify_column,
        )
    except Exception as e:
        raise EvaluationError(f"Failed to render clustering template: {e}")

    # Call the LLM with structured output
    try:
        result = call_llm_structured(
            prompt=formatted_prompt,
            model=model,
            response_model=ClusterResponse,
            model_params=get_model_params(model, temperature=0.3),
        )
    except Exception as e:
        raise EvaluationError(f"Clustering LLM call failed: {e}")

    # Convert Pydantic models to dicts for compatibility
    return {
        "clusters": [cluster.model_dump() for cluster in result.clusters],
    }
