"""
Prompt Optimizer Streamlit App

A human-in-the-loop tool for iteratively optimizing LLM prompts.
"""

import difflib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

import config
from utils import (
    EvaluationError,
    ProjectMetadata,
    RunMetadata,
    calculate_score_averages,
    detect_regressions,
    ensure_dir,
    extract_score_columns,
    format_score_with_ci,
    get_project_path,
    get_run_lineage,
    get_run_path,
    get_trend_label,
    list_projects,
    list_runs,
    load_example_history,
    load_project_metadata,
    load_prompt_file,
    load_run_metadata,
    load_template_with_fallback,
    paired_bootstrap_test,
    sample_size_guidance,
    save_project_metadata,
    save_prompt_file,
    save_run_metadata,
    split_dataset,
    validate_jinja_template,
)

# Load environment variables from .env file
load_dotenv()

PROJECTS_DIR = "./projects"

# Regex for valid project/run names (alphanumeric, hyphens, underscores)
VALID_NAME_PATTERN = r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$'

st.set_page_config(page_title="Prompt Optimizer", layout="wide")
st.title("Prompt Optimizer")


def get_selected_row(selected_rows, key: str = None):
    """
    Safely extract selected row(s) from AgGrid result.

    AgGrid can return DataFrame, list of dicts, or None depending on version.
    Returns a dict for single selection, list of dicts for multiple, or None.
    """
    if selected_rows is None:
        return None

    if isinstance(selected_rows, pd.DataFrame):
        if len(selected_rows) == 0:
            return None
        if key:
            return selected_rows.iloc[0][key]
        return selected_rows.to_dict("records")

    if isinstance(selected_rows, list):
        if len(selected_rows) == 0:
            return None
        if key:
            return selected_rows[0].get(key)
        return selected_rows

    return None


def validate_name(name: str, name_type: str = "name") -> str | None:
    """
    Validate project/run name against path traversal and invalid characters.
    Returns error message if invalid, None if valid.
    """
    import re
    if not name:
        return f"Please enter a {name_type}"
    if not re.match(VALID_NAME_PATTERN, name):
        return (
            f"{name_type.capitalize()} must start with alphanumeric and contain "
            "only letters, numbers, hyphens, and underscores"
        )
    if len(name) > 100:
        return f"{name_type.capitalize()} must be 100 characters or less"
    return None


def display_examples_expander(
    examples: list[dict], title: str = "View examples"
) -> None:
    """
    Display a list of examples in an expander with consistent formatting.

    Args:
        examples: List of example dictionaries
        title: Title for the expander (count will be appended)
    """
    with st.expander(f"{title} ({len(examples)})"):
        for i, ex in enumerate(examples):
            ex_id = ex.get("_example_id", i + 1)
            st.markdown(f"**Example {ex_id}**")
            for key, value in ex.items():
                if key != "_example_id":
                    st.text(f"{key}: {str(value)[:200]}")
            if i < len(examples) - 1:
                st.divider()


def run_evaluation(
    df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    eval_model: str,
    grader_prompt: str | None,
    progress_bar,
    status_text,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Run evaluation on a dataframe with parallel LLM calls.

    Uses ThreadPoolExecutor to process multiple rows concurrently.
    Calls config.eval() and config.score() for each row.
    Raises EvaluationError if any LLM call fails.

    Args:
        df: DataFrame with rows to evaluate
        system_prompt: System prompt for evaluation
        user_prompt: User prompt template
        eval_model: LiteLLM model string
        grader_prompt: Optional grader prompt template
        progress_bar: Streamlit progress bar widget
        status_text: Streamlit text widget for status updates
        max_workers: Maximum concurrent threads (default 8)

    Returns:
        DataFrame with evaluation results
    """
    total = len(df)
    completed = 0
    lock = __import__("threading").Lock()

    def process_row(row_dict: dict) -> dict:
        """Process a single row: eval then score."""
        # Call config.eval (this calls the LLM with retry)
        eval_outputs = config.eval(row_dict, system_prompt, user_prompt, eval_model)
        row_dict.update(eval_outputs)

        # Call config.score
        score_outputs = config.score(row_dict, grader_prompt, eval_model)
        row_dict.update(score_outputs)

        return row_dict

    # Convert DataFrame rows to list of dicts
    rows = [row.to_dict() for _, row in df.iterrows()]

    # Pre-allocate results to preserve original row order
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks, mapping future -> original index
        futures = {executor.submit(process_row, row): i for i, row in enumerate(rows)}

        for future in as_completed(futures):
            try:
                result = future.result()
                idx = futures[future]  # Get original row index
                with lock:
                    results[idx] = result  # Store at correct position
                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text.text(f"Processing {completed}/{total} rows...")
            except Exception as e:
                # Cancel remaining futures on error
                for f in futures:
                    f.cancel()
                raise EvaluationError(f"Evaluation failed: {e}") from e

    return pd.DataFrame(results)


def create_project_tab():
    """Create New Project tab."""
    st.header("Create New Project")

    with st.form("create_project_form"):
        # Project name
        project_name = st.text_input("Project Name", placeholder="my-qa-project")

        # Dataset file path
        dataset_path = st.text_input(
            "Dataset File (CSV)",
            placeholder="./samples/answer-evaluation.csv",
            help="Path to a CSV file containing your dataset",
        )

        # Split ratio
        split_ratio = st.selectbox(
            "Split Ratio (Train/Dev/Test)",
            ["40/40/20", "33/33/34", "50/25/25", "60/20/20"],
            index=0,
        )

        # Model configuration
        col1, col2 = st.columns(2)
        with col1:
            eval_model = st.text_input(
                "Evaluation Model",
                value="openai/responses/gpt-5-mini",
                help="LiteLLM model string for evaluation",
            )
        with col2:
            optimizer_model = st.text_input(
                "Optimizer Model",
                value="anthropic/claude-opus-4-5-20251101",
                help="LiteLLM model string for optimization",
            )

        # Prompt file paths
        st.subheader("Baseline Prompts")
        system_prompt_path = st.text_input(
            "System Prompt File",
            placeholder="./samples/system_prompt.txt",
            help="Path to a text file containing the system prompt",
        )
        user_prompt_path = st.text_input(
            "User Prompt Template File",
            placeholder="./samples/user_prompt.txt",
            help="Text file with {column_name} placeholders for dataset columns",
        )

        # Optional grader prompt
        st.subheader("Optional: Grading Configuration")
        grader_prompt_path = st.text_input(
            "Grader Prompt File (Jinja2 template, optional)",
            placeholder="./samples/grader_prompt.txt",
            help="Jinja2 template file, or leave empty for heuristic scoring only",
        )

        # Optimization target selection
        st.subheader("Optimization Target")
        prompt_to_optimize = st.radio(
            "Which prompt should be optimized?",
            ["system", "user"],
            index=0,
            format_func=lambda x: (
                "System Prompt" if x == "system" else "User Prompt Template"
            ),
            help="Which prompt to iteratively improve. The other stays constant.",
        )

        submitted = st.form_submit_button("Create Project")

        if submitted:
            # Validate project name (prevents path traversal)
            name_error = validate_name(project_name, "project name")
            if name_error:
                st.error(name_error)
                return
            if not dataset_path:
                st.error("Please enter a dataset file path")
                return
            if not system_prompt_path or not user_prompt_path:
                st.error("Please enter both system and user prompt file paths")
                return

            # Validate file existence
            if not os.path.isfile(dataset_path):
                st.error(f"Dataset file not found: {dataset_path}")
                return
            if not os.path.isfile(system_prompt_path):
                st.error(f"System prompt file not found: {system_prompt_path}")
                return
            if not os.path.isfile(user_prompt_path):
                st.error(f"User prompt file not found: {user_prompt_path}")
                return
            if grader_prompt_path and not os.path.isfile(grader_prompt_path):
                st.error(f"Grader prompt file not found: {grader_prompt_path}")
                return

            # Read prompt contents from files
            system_prompt = load_prompt_file(system_prompt_path)
            user_prompt = load_prompt_file(user_prompt_path)
            grader_prompt = (
                load_prompt_file(grader_prompt_path) if grader_prompt_path else ""
            )

            # Create project
            project_path = get_project_path(project_name, PROJECTS_DIR)
            if os.path.exists(project_path):
                st.error(f"Project '{project_name}' already exists")
                return

            ensure_dir(project_path)

            # Load and validate CSV
            try:
                df = pd.read_csv(dataset_path)
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")
                return

            if len(df) == 0:
                st.error("CSV file is empty")
                return

            if len(df) > 100000:
                st.error("CSV file too large (max 100,000 rows)")
                return

            # Validate user prompt template placeholders
            import re
            placeholders = re.findall(r'\{(\w+)\}', user_prompt)
            missing_cols = [p for p in placeholders if p not in df.columns]
            if missing_cols:
                st.error(
                    f"User prompt references columns not in dataset: {missing_cols}"
                )
                return
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

            # Get stratify column from config
            stratify_col = config.stratify(df)

            # Split dataset
            train_df, dev_df, test_df = split_dataset(
                df, split_ratio, stratify_column=stratify_col
            )

            # Save datasets
            df.to_csv(os.path.join(project_path, f"{dataset_name}.csv"), index=False)
            train_df.to_csv(
                os.path.join(project_path, f"{dataset_name}-train.csv"), index=False
            )
            dev_df.to_csv(
                os.path.join(project_path, f"{dataset_name}-dev.csv"), index=False
            )
            test_df.to_csv(
                os.path.join(project_path, f"{dataset_name}-test.csv"), index=False
            )

            # Save grader prompt if provided (with validation)
            if grader_prompt.strip():
                is_valid, error = validate_jinja_template(grader_prompt)
                if not is_valid:
                    st.error(f"Invalid grader prompt template: {error}")
                    return
                save_prompt_file(
                    os.path.join(project_path, "grader_prompt.txt"), grader_prompt
                )

            # Save project metadata
            metadata = ProjectMetadata(
                project_name=project_name,
                dataset_name=dataset_name,
                split_ratio=split_ratio,
                eval_model=eval_model,
                optimizer_model=optimizer_model,
                stratify_column=stratify_col,
                prompt_to_optimize=prompt_to_optimize,
                created_at=datetime.now(),
                dataset_source=dataset_path,
                system_prompt_source=system_prompt_path,
                user_prompt_source=user_prompt_path,
                grader_prompt_source=grader_prompt_path if grader_prompt_path else None,
            )
            save_project_metadata(project_path, metadata)

            # Create baseline run
            baseline_path = get_run_path(project_name, "baseline", PROJECTS_DIR)
            ensure_dir(baseline_path)

            sys_path = os.path.join(baseline_path, "system_prompt.txt")
            usr_path = os.path.join(baseline_path, "user_prompt.txt")
            save_prompt_file(sys_path, system_prompt)
            save_prompt_file(usr_path, user_prompt)

            run_metadata = RunMetadata(
                run_name="baseline",
                created_at=datetime.now(),
                parent_run=None,
                eval_completed=False,
            )
            save_run_metadata(baseline_path, run_metadata)

            st.success(f"Project '{project_name}' created successfully!")
            st.info(
                f"Dataset split: {len(train_df)} train, {len(dev_df)} dev, "
                f"{len(test_df)} test"
            )


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
    run_col1, run_col2 = st.columns([4, 1])
    with run_col1:
        run_name = st.selectbox("Select Run", runs)
    with run_col2:
        st.write("")  # Spacing to align with selectbox
        if st.button("↻ Refresh", help="Refresh run list"):
            st.rerun()
    run_path = get_run_path(project_name, run_name, PROJECTS_DIR)

    # Display current prompts
    try:
        system_prompt = load_prompt_file(os.path.join(run_path, "system_prompt.txt"))
        user_prompt = load_prompt_file(os.path.join(run_path, "user_prompt.txt"))
    except FileNotFoundError as e:
        st.error(f"Missing prompt file for run '{run_name}': {e.filename}")
        return

    with st.expander("View Prompts"):
        st.subheader("System Prompt")
        st.code(system_prompt)
        st.subheader("User Prompt Template")
        st.code(user_prompt)

    # Evaluate button
    if st.button("Evaluate", type="primary"):
        dataset_name = project_meta.dataset_name
        eval_model = project_meta.eval_model

        # Load grader prompt if exists
        grader_path = os.path.join(project_path, "grader_prompt.txt")
        grader_prompt = (
            load_prompt_file(grader_path) if os.path.exists(grader_path) else None
        )

        try:
            # Process each split
            for split in ["train", "dev", "test"]:
                st.subheader(f"Evaluating {split} split...")

                # Load data
                data_path = os.path.join(project_path, f"{dataset_name}-{split}.csv")
                df = pd.read_csv(data_path)

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run evaluation
                results_df = run_evaluation(
                    df,
                    system_prompt,
                    user_prompt,
                    eval_model,
                    grader_prompt,
                    progress_bar,
                    status_text,
                )

                # Save results
                eval_path = os.path.join(run_path, f"eval-{split}.csv")
                results_df.to_csv(eval_path, index=False)

                st.success(f"Saved {split} evaluation to {eval_path}")

            # Update run metadata with scores
            run_meta = load_run_metadata(run_path)
            run_meta.eval_completed = True
            run_meta.scores = {}

            for split in ["train", "dev", "test"]:
                eval_path = os.path.join(run_path, f"eval-{split}.csv")
                df = pd.read_csv(eval_path)
                score_cols = extract_score_columns(df)
                run_meta.scores[split] = calculate_score_averages(df, score_cols)

            save_run_metadata(run_path, run_meta)
            st.success("Evaluation complete!")

        except EvaluationError as e:
            st.error(f"Evaluation failed: {e}")
            st.error("Please check your API keys and model configuration.")


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
    prompt_to_optimize = project_meta.prompt_to_optimize

    # Display optimization target
    target_label = (
        "System Prompt" if prompt_to_optimize == "system" else "User Prompt Template"
    )
    st.info(f"**Optimization Target:** {target_label}")

    # Show template sources
    template_names = [
        "prompt-optimizer-prompt.jinja2",
        "error-analysis-prompt.jinja2",
        "clustering-prompt.jinja2",
    ]
    template_sources = {}
    for tname in template_names:
        _, tpath = load_template_with_fallback(project_path, tname)
        template_sources[tname] = tpath

    with st.expander("Template sources"):
        for tname, tpath in template_sources.items():
            # Show just the filename for display, with path as help
            display_name = tname.replace("-prompt.jinja2", "").replace("-", " ").title()
            st.caption(f"**{display_name}:** `{tpath}`")

    # Build runs table data with confidence intervals
    runs = list_runs(project_path)
    runs_data = []
    runs_raw_scores = []  # Raw numeric scores for parallel coordinates
    for run in runs:
        run_path = get_run_path(project_name, run, PROJECTS_DIR)
        try:
            run_meta = load_run_metadata(run_path)
            row = {"run_name": run, "eval_completed": run_meta.eval_completed}
            raw_row = {"run_name": run, "eval_completed": run_meta.eval_completed}

            # Load eval data and compute scores with CIs
            if run_meta.eval_completed:
                # Compute lineage depth for coloring
                lineage = get_run_lineage(project_path, run)
                raw_row["_lineage_depth"] = len(lineage) - 1

                for split in ["train", "dev", "test"]:
                    eval_path = os.path.join(run_path, f"eval-{split}.csv")
                    if os.path.exists(eval_path):
                        eval_df = pd.read_csv(eval_path)

                        # Count and exclude rows with eval errors
                        error_count = 0
                        if "_eval_error" in eval_df.columns:
                            error_count = eval_df["_eval_error"].notna().sum()
                            eval_df = eval_df[eval_df["_eval_error"].isna()]

                        score_cols = extract_score_columns(eval_df)
                        for score_col in score_cols:
                            scores = eval_df[score_col].dropna().tolist()
                            if scores:
                                formatted = format_score_with_ci(scores)
                                if error_count > 0:
                                    formatted += f" ({error_count} err)"
                                row[f"{split}_{score_col}"] = formatted
                                avg = sum(scores) / len(scores)
                                raw_row[f"{split}_{score_col}"] = avg
                            else:
                                row[f"{split}_{score_col}"] = "N/A"
                                raw_row[f"{split}_{score_col}"] = None

            runs_data.append(row)
            runs_raw_scores.append(raw_row)
        except Exception:
            runs_data.append({"run_name": run, "eval_completed": False})
            runs_raw_scores.append({"run_name": run, "eval_completed": False})

    runs_df = pd.DataFrame(runs_data)

    # Parallel coordinates visualization for comparing runs
    completed_raw = [r for r in runs_raw_scores if r.get("eval_completed", False)]

    if len(completed_raw) >= 2:
        st.subheader("Run Comparison")

        # Split selector
        viz_split = st.radio(
            "Score split",
            ["dev", "test", "train"],
            horizontal=True,
            help="Choose which data split to visualize scores for",
        )

        # Get score columns for this split
        sample_row = completed_raw[0]
        score_cols = [
            k.replace(f"{viz_split}_", "")
            for k in sample_row.keys()
            if k.startswith(f"{viz_split}_") and sample_row[k] is not None
        ]

        if score_cols:
            # Build plot dataframe
            plot_data = []
            for r in completed_raw:
                plot_row = {
                    "run_name": r["run_name"],
                    "_lineage_depth": r.get("_lineage_depth", 0),
                }
                for col in score_cols:
                    key = f"{viz_split}_{col}"
                    plot_row[col] = r.get(key)
                plot_data.append(plot_row)

            plot_df = pd.DataFrame(plot_data)

            # Filter out rows with missing scores
            plot_df = plot_df.dropna(subset=score_cols)

            if len(plot_df) >= 2:
                # Create dimensions for parallel coordinates
                dimensions = []

                # Add run name as first categorical dimension
                run_names = plot_df["run_name"].tolist()
                run_name_indices = list(range(len(run_names)))
                dimensions.append(
                    dict(
                        range=[0, len(run_names) - 1],
                        label="Run",
                        values=run_name_indices,
                        tickvals=run_name_indices,
                        ticktext=run_names,
                    )
                )

                # Add score columns as numeric dimensions
                for col in score_cols:
                    values = plot_df[col].tolist()
                    col_min = min(values) if values else 0
                    col_max = max(values) if values else 1
                    # Add padding to range
                    padding = (col_max - col_min) * 0.1 if col_max > col_min else 0.1
                    col_range = [max(0, col_min - padding), min(1, col_max + padding)]
                    dimensions.append(
                        dict(
                            range=col_range,
                            label=col.replace("_", " ").title(),
                            values=values,
                        )
                    )

                # Color by lineage depth (baseline=0, later iterations higher)
                max_depth = plot_df["_lineage_depth"].max()
                color_values = plot_df["_lineage_depth"].tolist()

                # Bright colorscale for dark backgrounds
                bright_colors = [
                    [0, "#00d4ff"],    # Cyan
                    [0.5, "#ff6b6b"],  # Coral
                    [1, "#ffd93d"],    # Yellow
                ]

                fig = go.Figure(
                    data=go.Parcoords(
                        line=dict(
                            color=color_values,
                            colorscale=bright_colors,
                            showscale=False,
                            cmin=0,
                            cmax=max(max_depth, 1),
                        ),
                        dimensions=dimensions,
                        labelangle=-30,
                        labelside="top",
                        unselected=dict(line=dict(opacity=0.5)),
                    )
                )

                fig.update_layout(
                    margin=dict(l=80, r=80, t=60, b=30),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "Need at least 2 runs with complete scores for comparison chart."
                )
        else:
            st.info(f"No score data available for {viz_split} split.")

    # Runs table with AgGrid
    st.subheader("Runs")
    gb = GridOptionsBuilder.from_dataframe(runs_df)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("run_name", pinned="left")
    grid_options = gb.build()

    runs_grid = AgGrid(
        runs_df,
        gridOptions=grid_options,
        update_on=["selectionChanged"],
        fit_columns_on_grid_load=True,
        height=200,
        key=f"runs_grid_{project_name}",
        width="stretch",
    )

    selected_rows = runs_grid.selected_rows
    grid_selected_run = get_selected_row(selected_rows, key="run_name")

    # Persist selection in session state to survive reruns
    selection_key = f"selected_run_{project_name}"
    if grid_selected_run:
        st.session_state[selection_key] = grid_selected_run
    selected_run = st.session_state.get(selection_key)

    # Validate that selected run still exists
    if selected_run and selected_run not in runs:
        del st.session_state[selection_key]
        selected_run = None

    # Clear all selection state when run changes (including first selection)
    prev_run_key = f"prev_optimize_run_{project_name}"
    prev_run = st.session_state.get(prev_run_key)
    if selected_run != prev_run:
        # Clear selection keys for ALL runs in this project to ensure clean slate
        keys_to_clear = [k for k in st.session_state.keys() if any(
            k.startswith(prefix) for prefix in [
                f"perf_select_{project_name}_",
                f"auto_select_{project_name}_",
                f"calibration_{project_name}_",
                f"clusters_{project_name}_",
                f"analysis_{project_name}_",
            ]
        )]
        for key in keys_to_clear:
            del st.session_state[key]
        st.session_state[prev_run_key] = selected_run

    if selected_run:
        run_path = get_run_path(project_name, selected_run, PROJECTS_DIR)

        # Sample size warning
        test_eval_path = os.path.join(run_path, "eval-test.csv")
        if os.path.exists(test_eval_path):
            test_df = pd.read_csv(test_eval_path)
            n_test = len(test_df)
            guidance = sample_size_guidance(n_test)
            if n_test < 50:
                st.warning(f"Test set has {n_test} examples. {guidance}")
            else:
                st.info(f"Test set has {n_test} examples. {guidance}")

        # Check if evaluation exists
        eval_train_path = os.path.join(run_path, "eval-train.csv")
        if not os.path.exists(eval_train_path):
            st.warning(
                f"Run '{selected_run}' has not been evaluated yet. "
                "Go to the Eval tab first."
            )
            return

        # Load eval data
        eval_df = pd.read_csv(eval_train_path)

        # Primary score column selector
        score_cols = extract_score_columns(eval_df)
        if score_cols:
            # Use config.primary_score() as default, fallback to first column
            default_score = config.primary_score(eval_df)
            if default_score in score_cols:
                default_idx = score_cols.index(default_score)
            else:
                default_idx = 0

            primary_score = st.selectbox(
                "Primary score column",
                score_cols,
                index=default_idx,
                key=f"primary_score_{project_name}",
                help="Score column used for thresholds, filtering, and analysis",
            )
        else:
            primary_score = None

        # Example Performance Diff View
        lineage = get_run_lineage(project_path, selected_run)

        if len(lineage) >= 2:
            st.subheader("Example Performance Across Runs")
            st.caption("Select rows to add them to optimization examples")

            if primary_score and "_example_id" in eval_df.columns:
                # Load history
                history_df = load_example_history(
                    project_path, lineage, "train", primary_score
                )

                if len(history_df) > 0:
                    # Detect regressions
                    regressions = detect_regressions(history_df, lineage)

                    # Show warnings
                    if regressions["broke"]:
                        st.warning(
                            f"{len(regressions['broke'])} examples that passed in "
                            f"{lineage[0]} now fail. Consider including them."
                        )

                    if regressions["oscillating"]:
                        st.info(
                            f"{len(regressions['oscillating'])} examples are "
                            "oscillating (improved then regressed or vice versa)."
                        )

                    # Build display DataFrame
                    display_df = history_df.copy()

                    # Add trend column
                    def compute_trend(row):
                        scores = [
                            row.get(run) for run in lineage
                            if run in row and pd.notna(row.get(run))
                        ]
                        return get_trend_label(scores, lineage)

                    display_df["Trend"] = display_df.apply(compute_trend, axis=1)

                    # Reorder columns
                    cols = ["_example_id"] + lineage + ["Trend"]
                    cols = [c for c in cols if c in display_df.columns]
                    display_df = display_df[cols]

                    # Round scores for display
                    for run in lineage:
                        if run in display_df.columns:
                            display_df[run] = display_df[run].round(2)

                    # Show table with selection capability
                    gb_perf = GridOptionsBuilder.from_dataframe(display_df)
                    gb_perf.configure_selection(
                        selection_mode="multiple", use_checkbox=True
                    )
                    gb_perf.configure_column("_example_id", pinned="left")
                    gb_perf.configure_column("Trend", pinned="right")
                    grid_options_perf = gb_perf.build()

                    perf_grid = AgGrid(
                        display_df,
                        gridOptions=grid_options_perf,
                        fit_columns_on_grid_load=True,
                        height=300,
                        key="perf_grid",
                        width="stretch",
                    )

                    # Store selected examples from performance grid
                    perf_select_key = f"perf_select_{project_name}_{selected_run}"
                    perf_selected = get_selected_row(perf_grid.selected_rows) or []
                    if perf_selected:
                        perf_ids = [
                            row["_example_id"] for row in perf_selected
                            if "_example_id" in row
                        ]
                        if perf_ids:
                            existing = st.session_state.get(perf_select_key, [])
                            new_ids = list(set(existing) | set(perf_ids))
                            st.session_state[perf_select_key] = new_ids
                            st.success(
                                f"Added {len(perf_ids)} examples to optimization"
                            )

                    # Show selected examples from performance tracking
                    perf_selected_ids = st.session_state.get(perf_select_key, [])
                    if perf_selected_ids and "_example_id" in eval_df.columns:
                        mask = eval_df["_example_id"].isin(perf_selected_ids)
                        perf_examples = eval_df[mask].to_dict("records")
                        if perf_examples:
                            ids_str = sorted(perf_selected_ids)
                            st.info(
                                f"{len(perf_examples)} examples selected for "
                                f"optimization (IDs: {ids_str})"
                            )
                            display_examples_expander(
                                perf_examples, "View selected examples"
                            )
                            if st.button("Clear selection", key="clear_perf_selection"):
                                del st.session_state[perf_select_key]
                                st.rerun()
            else:
                if "_example_id" not in eval_df.columns:
                    st.info(
                        "Example tracking not available. "
                        "Re-run evaluation to enable per-example tracking."
                    )

        # Compare run selection (optional)
        st.subheader("Compare with another run (optional)")
        other_runs = [r for r in runs if r != selected_run]
        compare_run = st.selectbox("Compare with", ["None"] + other_runs)

        if compare_run != "None":
            compare_path = get_run_path(project_name, compare_run, PROJECTS_DIR)
            compare_eval_path = os.path.join(compare_path, "eval-train.csv")
            if os.path.exists(compare_eval_path):
                compare_df = pd.read_csv(compare_eval_path)

                # Add comparison columns (aligned by _example_id, not position)
                score_cols = extract_score_columns(eval_df)
                has_ids = (
                    "_example_id" in eval_df.columns
                    and "_example_id" in compare_df.columns
                )
                for col in score_cols:
                    if col in compare_df.columns:
                        if has_ids:
                            # Merge on _example_id for correct alignment
                            compare_map = dict(
                                zip(compare_df["_example_id"], compare_df[col])
                            )
                            eval_df[f"{col}_compare"] = (
                                eval_df["_example_id"].map(compare_map).round(2)
                            )
                            eval_df[f"{col}_diff"] = (
                                eval_df[col] - eval_df[f"{col}_compare"]
                            ).round(2)
                        else:
                            # Fall back to positional (legacy, less accurate)
                            eval_df[f"{col}_compare"] = compare_df[col].round(2)
                            diff = (eval_df[col] - compare_df[col]).round(2)
                            eval_df[f"{col}_diff"] = diff

                # Show significance test results
                st.subheader("Statistical Comparison")
                for col in score_cols:
                    has_ids = (
                        "_example_id" in eval_df.columns
                        and "_example_id" in compare_df.columns
                    )
                    if col in compare_df.columns and has_ids:
                        # Align by _example_id for valid paired test
                        merged = eval_df[["_example_id", col]].merge(
                            compare_df[["_example_id", col]],
                            on="_example_id",
                            suffixes=("_selected", "_compare")
                        )
                        scores_selected = merged[f"{col}_selected"].dropna().tolist()
                        scores_compare = merged[f"{col}_compare"].dropna().tolist()

                        if len(scores_selected) > 0:
                            result = paired_bootstrap_test(
                                scores_compare,
                                scores_selected
                            )
                            sig = "✓ significant" if result["significant"] else "ns"
                            ci_lo = result['ci_lower']
                            ci_hi = result['ci_upper']
                            st.write(
                                f"**{col}**: {result['observed_diff']:+.2f} "
                                f"(95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]) **{sig}**"
                            )

        # Train dataset table with AgGrid
        st.subheader(f"Training Data - {selected_run}")

        # Analysis section
        st.subheader("Error Analysis (Optional)")

        col1, col2 = st.columns([1, 4])
        with col1:
            if primary_score:
                threshold_label = f"{primary_score} threshold"
            else:
                threshold_label = "Score threshold"
            score_threshold = st.number_input(
                threshold_label,
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Analyze rows with scores below this threshold",
            )
            chk_key = f"analyze_all_{project_name}_{selected_run}"
            analyze_all = st.checkbox("Analyze all rows", value=False, key=chk_key)

        # Analysis key for session state
        analysis_key = f"analysis_{project_name}_{selected_run}"

        with col2:
            if st.button("Analyze"):
                # Filter rows for analysis
                if analyze_all or not primary_score:
                    analysis_rows = eval_df.to_dict("records")
                else:
                    # Filter by primary score column
                    mask = eval_df[primary_score] < score_threshold
                    analysis_rows = eval_df[mask].to_dict("records")

                if not analysis_rows:
                    st.warning("No rows match the filter criteria")
                else:
                    # Load analysis prompt
                    analysis_template, analysis_template_path = (
                        load_template_with_fallback(
                            project_path, "error-analysis-prompt.jinja2"
                        )
                    )
                    st.caption(f"Using template: {analysis_template_path}")

                    with st.spinner(f"Analyzing {len(analysis_rows)} rows..."):
                        try:
                            stratify_col = config.stratify(eval_df)
                            score_summary = None
                            if primary_score:
                                mean = eval_df[primary_score].mean()
                                score_summary = f"Mean {primary_score}: {mean:.2f}"
                            result = config.analyze(
                                analysis_rows,
                                analysis_template,
                                project_meta.optimizer_model,
                                total_examples=len(eval_df),
                                score_summary=score_summary,
                                stratify_column=stratify_col,
                            )
                            st.session_state[analysis_key] = result
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

        # Editable analysis text
        # Initialize session state if not present
        if analysis_key not in st.session_state:
            st.session_state[analysis_key] = ""
        analysis_text = st.text_area(
            "Analysis result (editable)",
            height=200,
            key=analysis_key,
        )

        # Failure Clustering Section
        st.subheader("Failure Clustering (Optional)")

        cluster_col1, cluster_col2 = st.columns([1, 3])
        with cluster_col1:
            cluster_threshold = st.number_input(
                "Cluster threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Cluster rows with scores below this threshold"
            )

        # Session state for clustering results
        cluster_key = f"clusters_{project_name}_{selected_run}"

        with cluster_col2:
            if st.button("Cluster Failures"):
                # Filter rows for clustering
                if not primary_score:
                    st.error("No score columns found. Run evaluation first.")
                else:
                    mask = eval_df[primary_score] < cluster_threshold
                    failure_rows = eval_df[mask].to_dict("records")

                    if not failure_rows:
                        st.warning("No rows below threshold to cluster")
                    elif "_example_id" not in eval_df.columns:
                        st.error(
                            "Example IDs not found. Re-run evaluation to enable "
                            "clustering."
                        )
                    else:
                        # Load clustering template
                        clustering_template, clustering_template_path = (
                            load_template_with_fallback(
                                project_path, "clustering-prompt.jinja2"
                            )
                        )
                        st.caption(f"Using template: {clustering_template_path}")

                        with st.spinner(f"Clustering {len(failure_rows)} failures..."):
                            try:
                                stratify_col = config.stratify(eval_df)
                                result = config.cluster_failures(
                                    failure_rows,
                                    clustering_template,
                                    primary_score,
                                    project_meta.optimizer_model,
                                    total_examples=len(eval_df),
                                    threshold=cluster_threshold,
                                    stratify_column=stratify_col,
                                )
                                st.session_state[cluster_key] = result
                            except Exception as e:
                                st.error(f"Clustering failed: {e}")

        # Display clustering results
        if cluster_key in st.session_state:
            cluster_result = st.session_state[cluster_key]

            st.markdown(f"**Found {len(cluster_result['clusters'])} clusters:**")

            for i, cluster in enumerate(cluster_result["clusters"]):
                with st.container():
                    label = cluster.get('label', 'Unnamed')
                    n_examples = len(cluster.get('example_ids', []))
                    st.markdown(f"**Cluster {i+1}: {label}** ({n_examples} examples)")
                    st.markdown(f"_{cluster.get('description', 'No description')}_")
                    ids = ', '.join(map(str, cluster.get('example_ids', [])))
                    st.markdown(f"IDs: {ids}")
                    st.divider()

            # Auto-select button
            auto_select_key = f"auto_select_{project_name}_{selected_run}"
            if st.button("Auto-select diverse set"):
                diverse_ids = []
                for cluster in cluster_result["clusters"]:
                    ids = cluster.get("example_ids", [])
                    if ids:
                        diverse_ids.append(ids[0])  # Take first from each cluster
                # Store in session state for grid pre-selection
                st.session_state[auto_select_key] = diverse_ids
                st.rerun()

            # Show selected examples from clustering (under the button)
            auto_selected_ids = st.session_state.get(auto_select_key, [])
            if auto_selected_ids and "_example_id" in eval_df.columns:
                mask = eval_df["_example_id"].isin(auto_selected_ids)
                diverse_examples = eval_df[mask].to_dict("records")
                ids_str = sorted(auto_selected_ids)
                st.success(
                    f"Auto-selected {len(diverse_examples)} examples (IDs: {ids_str})"
                )
                display_examples_expander(diverse_examples, "View diverse examples")
                if st.button("Clear auto-selection", key="clear_auto_cluster"):
                    del st.session_state[auto_select_key]
                    st.rerun()

        # High-scoring (calibration) examples section
        st.subheader("Calibration Examples (Optional)")
        st.caption("High-scoring examples help the optimizer understand what works.")

        # Compute default based on stratify column
        stratify_col = config.stratify(eval_df)
        if stratify_col and stratify_col in eval_df.columns:
            strata_count = eval_df[stratify_col].nunique()
        else:
            strata_count = 3  # fallback

        calib_col1, calib_col2 = st.columns([1, 1])
        with calib_col1:
            n_calibration = st.number_input(
                "Number of examples",
                min_value=0,
                max_value=max(strata_count, 10),
                value=strata_count,
                help=f"Default is {strata_count} (one per {stratify_col or 'stratum'})",
                key=f"n_calib_{project_name}_{selected_run}",
            )
        with calib_col2:
            high_threshold = st.number_input(
                "Score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Select examples with score >= this threshold",
                key=f"high_thresh_{project_name}_{selected_run}",
            )

        calib_key = f"calibration_{project_name}_{selected_run}"

        if st.button("Auto-select high-scoring examples"):
            if primary_score:
                high_mask = eval_df[primary_score] >= high_threshold
                high_df = eval_df[high_mask]

                if len(high_df) > 0:
                    n_samples = min(n_calibration, len(high_df))

                    if stratify_col and stratify_col in high_df.columns:
                        # Filter out NaN values to avoid NaN weights from groupby
                        valid_high_df = high_df.dropna(subset=[stratify_col])
                        if len(valid_high_df) >= n_samples:
                            # Proportional sampling - weight by stratum frequency
                            grouped = valid_high_df.groupby(stratify_col)[
                                stratify_col
                            ]
                            weights = grouped.transform("count")
                            high_scoring_examples = valid_high_df.sample(
                                n=n_samples, weights=weights, random_state=42
                            ).to_dict("records")
                        else:
                            # Not enough valid rows, fall back to random sampling
                            high_scoring_examples = high_df.sample(
                                n=min(n_samples, len(high_df)), random_state=42
                            ).to_dict("records")
                    else:
                        # No stratify column - random sample
                        high_scoring_examples = high_df.sample(
                            n=n_samples, random_state=42
                        ).to_dict("records")

                    st.session_state[calib_key] = high_scoring_examples
                    st.rerun()
                else:
                    st.warning(f"No examples found with score >= {high_threshold}")
            else:
                st.error("No score columns found")

        # Display selected calibration examples (under the button)
        if calib_key in st.session_state:
            calibration_examples = st.session_state[calib_key]
            if calibration_examples:
                # Extract example IDs
                calib_ids = [
                    ex.get("_example_id", i + 1)
                    for i, ex in enumerate(calibration_examples)
                ]
                st.success(
                    f"Selected {len(calibration_examples)} high-scoring examples "
                    f"(IDs: {sorted(calib_ids)})"
                )
                display_examples_expander(
                    calibration_examples, "View calibration examples"
                )

                if st.button("Clear calibration examples"):
                    del st.session_state[calib_key]
                    st.rerun()

        # Data grid for example selection
        st.subheader("Select Additional Examples for Optimization")

        gb2 = GridOptionsBuilder.from_dataframe(eval_df)
        gb2.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb2.configure_default_column(sortable=True, filterable=True, resizable=True)

        # Truncate long text columns with tooltip on hover
        if len(eval_df) > 0:
            for col in eval_df.columns:
                sample_val = str(eval_df[col].iloc[0])
                if len(sample_val) > 50:
                    cell_style = {
                        "textOverflow": "ellipsis",
                        "overflow": "hidden",
                        "whiteSpace": "nowrap",
                    }
                    gb2.configure_column(
                        col,
                        maxWidth=300,
                        tooltipField=col,
                        cellStyle=cell_style,
                    )

        grid_options2 = gb2.build()
        grid_options2["tooltipShowDelay"] = 200

        data_grid = AgGrid(
            eval_df,
            gridOptions=grid_options2,
            fit_columns_on_grid_load=False,
            height=400,
            allow_unsafe_jscode=True,
            width="stretch",
            key=f"example_grid_{project_name}_{selected_run}",
        )

        manually_selected = get_selected_row(data_grid.selected_rows) or []

        # Show selection summary for manually selected examples only
        if manually_selected:
            st.info(f"{len(manually_selected)} additional examples selected")
            display_examples_expander(manually_selected, "View additional examples")

        # Optimization
        st.subheader("Generate Optimized Prompt")

        target_run_name = st.text_input(
            "New Run Name",
            value=f"{selected_run}-v2",
            help="Name for the new run with optimized prompt",
        )

        if st.button("Optimize", type="primary"):
            # Gather examples from all four sources
            auto_select_key = f"auto_select_{project_name}_{selected_run}"
            perf_select_key = f"perf_select_{project_name}_{selected_run}"
            diverse_ids = st.session_state.get(auto_select_key, [])
            perf_ids = st.session_state.get(perf_select_key, [])
            calibration_examples = st.session_state.get(calib_key, [])

            # Build combined examples list (avoiding duplicates)
            all_example_ids = set(diverse_ids) | set(perf_ids)
            for ex in manually_selected:
                ex_id = ex.get("_example_id")
                if ex_id is not None:
                    all_example_ids.add(ex_id)

            if all_example_ids and "_example_id" in eval_df.columns:
                mask = eval_df["_example_id"].isin(all_example_ids)
                low_scoring_examples = eval_df[mask].to_dict("records")
            else:
                low_scoring_examples = manually_selected

            # Check that we have at least one example from any source
            if not low_scoring_examples and not calibration_examples:
                st.error(
                    "Please select at least one example (performance tracking, "
                    "clustering, calibration, or additional)"
                )
                return

            # Validate run name (prevents path traversal)
            name_error = validate_name(target_run_name, "run name")
            if name_error:
                st.error(name_error)
                return

            # Check if run already exists
            target_path = get_run_path(project_name, target_run_name, PROJECTS_DIR)
            if os.path.exists(target_path):
                st.error(f"Run '{target_run_name}' already exists")
                return

            # Load current prompts
            try:
                sys_path = os.path.join(run_path, "system_prompt.txt")
                usr_path = os.path.join(run_path, "user_prompt.txt")
                system_prompt = load_prompt_file(sys_path)
                user_prompt = load_prompt_file(usr_path)
            except FileNotFoundError as e:
                st.error(f"Missing prompt file: {e}")
                return

            # Load optimizer prompt
            optimizer_template, optimizer_template_path = load_template_with_fallback(
                project_path, "prompt-optimizer-prompt.jinja2"
            )
            st.caption(f"Using template: {optimizer_template_path}")

            # Use low-scoring examples (diverse + manually selected)
            examples = low_scoring_examples

            # Compute additional data for optimization
            stratify_col = config.stratify(eval_df)

            # Get high-scoring examples from session state (user-selected via UI)
            high_scoring_examples = st.session_state.get(calib_key, [])

            # Compute score statistics
            score_stats = {}
            for score_col in score_cols:
                scores = eval_df[score_col].dropna()
                if len(scores) > 0:
                    threshold = 0.7
                    score_stats[score_col] = {
                        "mean": float(scores.mean()),
                        "std": float(scores.std()),
                        "n_total": len(scores),
                        "n_low": int((scores < threshold).sum()),
                        "threshold": threshold,
                    }

            # Compute scoring pattern (using stratify column)
            scoring_pattern = None
            if stratify_col and primary_score and stratify_col in eval_df.columns:
                valid = eval_df[[primary_score, stratify_col]].dropna()
                if len(valid) > 0:
                    avg_score = valid[primary_score].mean()

                    # Check if stratify_col is numeric (for calibration analysis)
                    if pd.api.types.is_numeric_dtype(eval_df[stratify_col]):
                        avg_reference = valid[stratify_col].mean()
                        bias = avg_score - avg_reference
                        direction = "over-scoring" if bias > 0 else "under-scoring"

                        # Per-level breakdown
                        breakdown_lines = []
                        for ref_val in sorted(valid[stratify_col].unique()):
                            subset = valid[valid[stratify_col] == ref_val]
                            avg_given = subset[primary_score].mean()
                            line = (
                                f"- {stratify_col}={ref_val}: "
                                f"avg {primary_score}={avg_given:.2f}"
                            )
                            breakdown_lines.append(line)

                        scoring_pattern = (
                            f"The model is {direction} by {abs(bias):.2f} points "
                            f"on average.\n"
                            f"Average {primary_score}: {avg_score:.2f}, "
                            f"Average {stratify_col}: {avg_reference:.2f}\n\n"
                            f"Per-level breakdown:\n"
                        ) + "\n".join(breakdown_lines)
                    else:
                        # Non-numeric stratify column - show distribution by group
                        breakdown_lines = []
                        for group_val in sorted(valid[stratify_col].unique()):
                            subset = valid[valid[stratify_col] == group_val]
                            avg = subset[primary_score].mean()
                            line = (
                                f"- {stratify_col}={group_val}: "
                                f"avg {primary_score}={avg:.2f} (n={len(subset)})"
                            )
                            breakdown_lines.append(line)

                        scoring_pattern = (
                            f"Score distribution by {stratify_col}:\n"
                        ) + "\n".join(breakdown_lines)

            with st.spinner("Generating optimized prompt..."):
                try:
                    optimized_prompt = config.optimize(
                        optimizer_template,
                        system_prompt,
                        user_prompt,
                        examples,
                        analysis_text if analysis_text.strip() else None,
                        project_meta.optimizer_model,
                        target_prompt=prompt_to_optimize,
                        high_scoring_examples=high_scoring_examples,
                        score_stats=score_stats,
                        scoring_pattern=scoring_pattern,
                    )

                    # Create new run
                    ensure_dir(target_path)
                    sys_out = os.path.join(target_path, "system_prompt.txt")
                    usr_out = os.path.join(target_path, "user_prompt.txt")
                    if prompt_to_optimize == "system":
                        # Optimizing system prompt - save new system, copy user
                        save_prompt_file(sys_out, optimized_prompt)
                        save_prompt_file(usr_out, user_prompt)
                    else:
                        # Optimizing user prompt - copy system, save new user
                        save_prompt_file(sys_out, system_prompt)
                        save_prompt_file(usr_out, optimized_prompt)

                    # Get indices of selected examples by _example_id
                    selected_ids = {
                        ex.get("_example_id") for ex in examples
                        if ex.get("_example_id") is not None
                    }
                    selected_indices = []
                    if selected_ids and "_example_id" in eval_df.columns:
                        for i, example_id in enumerate(eval_df["_example_id"]):
                            if example_id in selected_ids:
                                selected_indices.append(i)

                    # Get clustering results from session state if available
                    cluster_key = f"clusters_{project_name}_{selected_run}"
                    clustering_results = None
                    if cluster_key in st.session_state:
                        cluster_data = st.session_state[cluster_key]
                        if "clusters" in cluster_data:
                            clustering_results = [
                                {
                                    "label": c.get("label"),
                                    "description": c.get("description"),
                                    "example_ids": c.get("example_ids", []),
                                }
                                for c in cluster_data["clusters"]
                            ]

                    run_meta = RunMetadata(
                        run_name=target_run_name,
                        created_at=datetime.now(),
                        parent_run=selected_run,
                        eval_completed=False,
                        analysis_text=analysis_text,
                        selected_examples=selected_indices,
                        clustering_results=clustering_results,
                    )
                    save_run_metadata(target_path, run_meta)

                    if prompt_to_optimize == "system":
                        optimized_label = "system prompt"
                    else:
                        optimized_label = "user prompt template"
                    st.success(
                        f"Created new run '{target_run_name}' with optimized "
                        f"{optimized_label}!"
                    )

                    with st.expander(f"View Optimized {target_label}"):
                        st.code(optimized_prompt)

                    # Show diff between original and optimized
                    original_prompt = (
                        system_prompt
                        if prompt_to_optimize == "system"
                        else user_prompt
                    )
                    diff_lines = difflib.unified_diff(
                        original_prompt.splitlines(keepends=True),
                        optimized_prompt.splitlines(keepends=True),
                        fromfile="Original",
                        tofile="Optimized",
                    )
                    diff_text = "".join(diff_lines)
                    if diff_text:
                        with st.expander("View Diff"):
                            st.code(diff_text, language="diff")

                except Exception as e:
                    st.error(f"Optimization failed: {e}")


# Main app
tab1, tab2, tab3 = st.tabs(["Create Project", "Evaluate", "Optimize"])

with tab1:
    create_project_tab()

with tab2:
    eval_tab()

with tab3:
    optimize_tab()
