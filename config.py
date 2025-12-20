"""
User-customizable functions for the prompt optimizer.

Modify these functions to adapt the optimizer to your specific use case.
The default implementations work for an emotion classification task.
"""

import logging
import re
import time

from pydantic import BaseModel, Field

from utils import (
    ClusterResponse,
    EvaluationError,
    call_llm_single_prompt,
    call_llm_structured,
    format_user_prompt,
    render_jinja_template,
)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Evaluation model - hardcoded for this sample task
EVAL_MODEL = "openai/gpt-4o-mini"
EVAL_MODEL_PARAMS = {"temperature": 0.0}  # Deterministic for classification

# Optimizer model params - used by optimize(), analyze(), cluster_failures()
# Empty by default for reasoning models; modify if needed for your model
OPTIMIZE_MODEL_PARAMS = {}

# Valid emotion labels for classification
VALID_EMOTIONS = {"joy", "anger", "sadness", "surprise"}


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =============================================================================


class EmotionResponse(BaseModel):
    """Response for emotion classification."""

    emotion: str = Field(..., description="One of: joy, anger, sadness, surprise")


# =============================================================================
# SCORE CONFIGURATION
# =============================================================================


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
    return "emotion"


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
) -> dict:
    """
    Evaluate a single row by calling the LLM with structured output.

    This function:
    1. Formats the user prompt with values from the row
    2. Calls the LLM with structured output to get the predicted emotion
    3. Retries up to 3 times if the LLM returns an invalid emotion

    Args:
        row: Dictionary of column values from the dataset
        system_prompt: The system prompt
        user_prompt_template: User prompt template with {column} placeholders

    Returns:
        Dictionary of outputs to add to the row:
        - response: The raw LLM response (emotion label)
        - predicted_emotion: Normalized emotion label (lowercase)

    Example return:
        {
            "response": "joy",
            "predicted_emotion": "joy"
        }
    """
    # Format the user prompt with row values
    try:
        user_prompt = format_user_prompt(user_prompt_template, row)
    except KeyError as e:
        raise EvaluationError(f"User prompt template references missing column: {e}")

    # Call the LLM with structured output, retrying if invalid emotion returned
    max_retries = 3
    last_error = None
    predicted = None

    for attempt in range(max_retries):
        try:
            result = call_llm_structured(
                prompt=user_prompt,
                model=EVAL_MODEL,
                response_model=EmotionResponse,
                system_prompt=system_prompt,
                model_params=EVAL_MODEL_PARAMS,
            )

            # Normalize: lowercase, strip whitespace
            predicted = result.emotion.strip().lower()

            # Check if emotion is valid
            if predicted in VALID_EMOTIONS:
                return {
                    "response": result.emotion,
                    "predicted_emotion": predicted,
                }

            # Invalid emotion - log and retry
            logging.warning(
                f"Eval attempt {attempt + 1}/{max_retries}: invalid emotion '{predicted}'. "
                f"Expected one of {VALID_EMOTIONS}. Retrying..."
            )

        except Exception as e:
            last_error = e
            logging.warning(f"Eval attempt {attempt + 1}/{max_retries} failed: {e}")

        # Exponential backoff before retry (skip on last attempt)
        if attempt < max_retries - 1:
            time.sleep(0.5 * (2**attempt))

    # If all retries failed with exceptions, raise error
    if last_error is not None and predicted is None:
        raise EvaluationError(
            f"LLM call failed in eval after {max_retries} retries: {last_error}"
        )

    # Return the last result even if invalid (will score 0.0)
    logging.warning(
        f"Returning invalid emotion '{predicted}' after {max_retries} retries"
    )
    return {
        "response": result.emotion,
        "predicted_emotion": predicted,
    }


def score(
    row: dict,
    grader_prompt: str | None,
) -> dict:
    """
    Score an evaluated row, returning scores with reasons.

    For emotion classification, this uses exact match comparison.

    Args:
        row: Dictionary containing original data plus eval outputs
             (e.g., 'predicted_emotion', 'emotion')
        grader_prompt: Optional Jinja2 template for LLM-as-judge grading (not used for exact match)

    Returns:
        Dictionary of scores with paired reason keys.
        Format: {"score_name": value, "score_name_reason": "explanation"}

    Example return:
        {
            "accuracy": 1.0,
            "accuracy_reason": "Predicted: joy, Expected: joy"
        }
    """
    scores = {}

    # Exact match scoring for emotion classification
    predicted = row.get("predicted_emotion", "").lower()
    expected = row.get("emotion", "").lower()

    match = predicted == expected
    scores["accuracy"] = 1.0 if match else 0.0
    scores["accuracy_reason"] = f"Predicted: {predicted}, Expected: {expected}"

    # --- OPTIONAL: LLM-as-judge grading ---
    # Uncomment and customize for additional quality scoring:
    #
    # if grader_prompt and "response" in row:
    #     class GraderResponse(BaseModel):
    #         relevance: float = Field(..., ge=0.0, le=1.0)
    #         relevance_reason: str
    #
    #     formatted_grader = render_jinja_template(grader_prompt, row=row)
    #     result = call_llm_structured(
    #         prompt=formatted_grader,
    #         model=EVAL_MODEL,
    #         response_model=GraderResponse,
    #         model_params=EVAL_MODEL_PARAMS,
    #     )
    #     scores["relevance"] = result.relevance
    #     scores["relevance_reason"] = result.relevance_reason

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
            formatted_prompt, model, model_params=OPTIMIZE_MODEL_PARAMS
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
            formatted_prompt, model, model_params=OPTIMIZE_MODEL_PARAMS
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
            model_params=OPTIMIZE_MODEL_PARAMS,
        )
    except Exception as e:
        raise EvaluationError(f"Clustering LLM call failed: {e}")

    # Convert Pydantic models to dicts for compatibility
    return {
        "clusters": [cluster.model_dump() for cluster in result.clusters],
    }
