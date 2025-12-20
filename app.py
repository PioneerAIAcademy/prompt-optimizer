"""
Prompt Optimizer Streamlit App

A human-in-the-loop tool for iteratively optimizing LLM prompts.
"""

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
        return f"{name_type.capitalize()} must start with alphanumeric and contain only letters, numbers, hyphens, and underscores"
    if len(name) > 100:
        return f"{name_type.capitalize()} must be 100 characters or less"
    return None


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
    results = []
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_row, row): i for i, row in enumerate(rows)}

        for future in as_completed(futures):
            try:
                result = future.result()
                with lock:
                    results.append(result)
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
            help="Path to a text file with {column_name} placeholders for dataset columns",
        )

        # Optional grader prompt
        st.subheader("Optional: Grading Configuration")
        grader_prompt_path = st.text_input(
            "Grader Prompt File (Jinja2 template, optional)",
            placeholder="./samples/grader_prompt.txt",
            help="Path to a Jinja2 template file, or leave empty to use heuristic scoring only",
        )

        # Optimization target selection
        st.subheader("Optimization Target")
        prompt_to_optimize = st.radio(
            "Which prompt should be optimized?",
            ["system", "user"],
            index=0,
            format_func=lambda x: "System Prompt" if x == "system" else "User Prompt Template",
            help="Select which prompt will be iteratively improved. The other remains constant across all runs.",
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
            grader_prompt = load_prompt_file(grader_prompt_path) if grader_prompt_path else ""

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
                st.error(f"User prompt references columns not in dataset: {missing_cols}")
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

            save_prompt_file(os.path.join(baseline_path, "system_prompt.txt"), system_prompt)
            save_prompt_file(os.path.join(baseline_path, "user_prompt.txt"), user_prompt)

            run_metadata = RunMetadata(
                run_name="baseline",
                created_at=datetime.now(),
                parent_run=None,
                eval_completed=False,
            )
            save_run_metadata(baseline_path, run_metadata)

            st.success(f"Project '{project_name}' created successfully!")
            st.info(
                f"Dataset split: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test"
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
        grader_prompt = load_prompt_file(grader_path) if os.path.exists(grader_path) else None

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
    target_label = "System Prompt" if prompt_to_optimize == "system" else "User Prompt Template"
    st.info(f"**Optimization Target:** {target_label}")

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
                                raw_row[f"{split}_{score_col}"] = sum(scores) / len(scores)
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
                    dimensions.append(
                        dict(
                            range=[max(0, col_min - padding), min(1, col_max + padding)],
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
                st.info("Need at least 2 runs with complete scores for comparison chart.")
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
                f"Run '{selected_run}' has not been evaluated yet. Go to the Eval tab first."
            )
            return

        # Load eval data
        eval_df = pd.read_csv(eval_train_path)

        # Example Performance Diff View
        lineage = get_run_lineage(project_path, selected_run)

        if len(lineage) >= 2:
            st.subheader("Example Performance Across Runs")
            st.caption("Select rows to add them to optimization examples")

            # Get primary score column
            score_cols = extract_score_columns(eval_df)

            if score_cols and "_example_id" in eval_df.columns:
                primary_score = score_cols[0]

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
                            f"{lineage[0]} now fail. Consider including them in optimization."
                        )

                    if regressions["oscillating"]:
                        st.info(
                            f"{len(regressions['oscillating'])} examples are oscillating "
                            f"(improved then regressed or vice versa)."
                        )

                    # Build display DataFrame
                    display_df = history_df.copy()

                    # Add trend column
                    def compute_trend(row):
                        scores = [row.get(run) for run in lineage if run in row and pd.notna(row.get(run))]
                        return get_trend_label(scores, lineage)

                    display_df["Trend"] = display_df.apply(compute_trend, axis=1)

                    # Reorder columns
                    cols = ["_example_id"] + lineage + ["Trend"]
                    display_df = display_df[[c for c in cols if c in display_df.columns]]

                    # Round scores for display
                    for run in lineage:
                        if run in display_df.columns:
                            display_df[run] = display_df[run].round(2)

                    # Show table with selection capability
                    gb_perf = GridOptionsBuilder.from_dataframe(display_df)
                    gb_perf.configure_selection(selection_mode="multiple", use_checkbox=True)
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
                    perf_selected = get_selected_row(perf_grid.selected_rows) or []
                    if perf_selected:
                        perf_ids = [row["_example_id"] for row in perf_selected if "_example_id" in row]
                        if perf_ids:
                            auto_select_key = f"auto_select_{project_name}_{selected_run}"
                            existing_ids = set(st.session_state.get(auto_select_key, []))
                            new_ids = list(existing_ids | set(perf_ids))
                            st.session_state[auto_select_key] = new_ids
                            st.success(f"Added {len(perf_ids)} examples to optimization selection")
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

                # Add comparison columns
                score_cols = extract_score_columns(eval_df)
                for col in score_cols:
                    if col in compare_df.columns:
                        eval_df[f"{col}_compare"] = compare_df[col].round(2)
                        eval_df[f"{col}_diff"] = (eval_df[col] - compare_df[col]).round(2)

                # Show significance test results
                st.subheader("Statistical Comparison")
                for col in score_cols:
                    if col in compare_df.columns and "_example_id" in eval_df.columns and "_example_id" in compare_df.columns:
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
                            sig_marker = "✓ significant" if result["significant"] else "ns"
                            st.write(
                                f"**{col}**: {result['observed_diff']:+.2f} "
                                f"(95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]) "
                                f"**{sig_marker}**"
                            )

        # Train dataset table with AgGrid
        st.subheader(f"Training Data - {selected_run}")

        # Analysis section
        st.subheader("Error Analysis (Optional)")

        # Get score columns for filtering
        score_cols = extract_score_columns(eval_df)
        primary_score = score_cols[0] if score_cols else None

        col1, col2 = st.columns([1, 4])
        with col1:
            threshold_label = f"{primary_score} threshold" if primary_score else "Score threshold"
            score_threshold = st.number_input(
                threshold_label,
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help=f"Analyze rows with {primary_score or 'scores'} below this threshold",
            )
            analyze_all = st.checkbox("Analyze all rows", value=False, key=f"analyze_all_{project_name}_{selected_run}")

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
                    project_analysis_path = os.path.join(
                        project_path, "error-analysis-prompt.jinja2"
                    )
                    if os.path.exists(project_analysis_path):
                        analysis_template = load_prompt_file(project_analysis_path)
                    else:
                        analysis_template = load_prompt_file("error-analysis-prompt.jinja2")

                    with st.spinner(f"Analyzing {len(analysis_rows)} rows..."):
                        try:
                            result = config.analyze(
                                analysis_rows,
                                analysis_template,
                                project_meta.optimizer_model,
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
                score_cols = extract_score_columns(eval_df)
                if not score_cols:
                    st.error("No score columns found. Run evaluation first.")
                else:
                    primary_score = score_cols[0]
                    mask = eval_df[primary_score] < cluster_threshold
                    failure_rows = eval_df[mask].to_dict("records")

                    if not failure_rows:
                        st.warning("No rows below threshold to cluster")
                    elif "_example_id" not in eval_df.columns:
                        st.error("Example IDs not found. Re-run evaluation to enable clustering.")
                    else:
                        # Load clustering template
                        project_clustering_path = os.path.join(
                            project_path, "clustering-prompt.jinja2"
                        )
                        if os.path.exists(project_clustering_path):
                            clustering_template = load_prompt_file(project_clustering_path)
                        else:
                            clustering_template = load_prompt_file("clustering-prompt.jinja2")

                        with st.spinner(f"Clustering {len(failure_rows)} failures..."):
                            try:
                                result = config.cluster_failures(
                                    failure_rows,
                                    clustering_template,
                                    primary_score,
                                    project_meta.optimizer_model
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
                    st.markdown(f"**Cluster {i+1}: {cluster.get('label', 'Unnamed')}** "
                               f"({len(cluster.get('example_ids', []))} examples)")
                    st.markdown(f"_{cluster.get('description', 'No description')}_")
                    st.markdown(f"IDs: {', '.join(map(str, cluster.get('example_ids', [])))}")
                    st.divider()

            # Auto-select button
            if st.button("Auto-select diverse set"):
                diverse_ids = []
                for cluster in cluster_result["clusters"]:
                    ids = cluster.get("example_ids", [])
                    if ids:
                        diverse_ids.append(ids[0])  # Take first from each cluster
                # Store in session state for grid pre-selection
                st.session_state[f"auto_select_{project_name}_{selected_run}"] = diverse_ids
                st.rerun()

        # Data grid for example selection
        st.subheader("Select Additional Examples for Optimization")

        # Check for auto-selected rows from clustering
        auto_select_key = f"auto_select_{project_name}_{selected_run}"
        auto_selected_ids = st.session_state.get(auto_select_key, [])

        if auto_selected_ids and "_example_id" in eval_df.columns:
            # Get the auto-selected examples as dicts
            auto_selected_examples = eval_df[eval_df["_example_id"].isin(auto_selected_ids)].to_dict("records")
            st.success(f"Auto-selected {len(auto_selected_examples)} rows from clustering (IDs: {sorted(auto_selected_ids)}); select additional rows below")
            if st.button("Clear auto-selection"):
                del st.session_state[auto_select_key]
                st.rerun()
        else:
            auto_selected_examples = []

        gb2 = GridOptionsBuilder.from_dataframe(eval_df)
        gb2.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb2.configure_default_column(sortable=True, filterable=True, resizable=True)

        # Truncate long text columns with tooltip on hover
        if len(eval_df) > 0:
            for col in eval_df.columns:
                sample_val = str(eval_df[col].iloc[0])
                if len(sample_val) > 50:
                    gb2.configure_column(
                        col,
                        maxWidth=300,
                        tooltipField=col,
                        cellStyle={"textOverflow": "ellipsis", "overflow": "hidden", "whiteSpace": "nowrap"},
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
        )

        manually_selected = get_selected_row(data_grid.selected_rows) or []

        # Merge auto-selected and manually selected, avoiding duplicates
        all_selected_ids = set(auto_selected_ids)
        for ex in manually_selected:
            ex_id = ex.get("_example_id")
            if ex_id is not None:
                all_selected_ids.add(ex_id)

        # Build final selected examples list
        if all_selected_ids and "_example_id" in eval_df.columns:
            selected_examples = eval_df[eval_df["_example_id"].isin(all_selected_ids)].to_dict("records")
        else:
            selected_examples = manually_selected

        # Update session state for coverage tracking
        if selected_examples:
            st.session_state["selected_example_ids"] = list(all_selected_ids)

        # Show selection summary
        if auto_selected_examples and manually_selected:
            manual_ids = [ex.get("_example_id") for ex in manually_selected if ex.get("_example_id") is not None]
            st.info(f"Total: {len(selected_examples)} examples ({len(auto_selected_ids)} auto-selected + {len(set(manual_ids) - set(auto_selected_ids))} manually selected)")
        elif selected_examples:
            st.info(f"{len(selected_examples)} examples selected")

        # Row detail view for selected examples
        if selected_examples:
            with st.expander(f"View selected rows ({len(selected_examples)})"):
                for i, example_dict in enumerate(selected_examples):
                    example_id = example_dict.get("_example_id", i + 1)
                    st.markdown(f"### Example {example_id}")
                    for key, value in example_dict.items():
                        st.markdown(f"**{key}:**")
                        st.text(str(value))
                    if i < len(selected_examples) - 1:
                        st.divider()

        # Optimization
        st.subheader("Generate Optimized Prompt")

        target_run_name = st.text_input(
            "New Run Name",
            value=f"{selected_run}-v2",
            help="Name for the new run with optimized prompt",
        )

        if st.button("Optimize", type="primary"):
            if not selected_examples:
                st.error("Please select at least one example")
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
                system_prompt = load_prompt_file(os.path.join(run_path, "system_prompt.txt"))
                user_prompt = load_prompt_file(os.path.join(run_path, "user_prompt.txt"))
            except FileNotFoundError as e:
                st.error(f"Missing prompt file: {e}")
                return

            # Load optimizer prompt
            optimizer_template = load_prompt_file("prompt-optimizer-prompt.jinja2")

            # Use selected examples directly (already list of dicts)
            examples = selected_examples

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
                    )

                    # Create new run
                    ensure_dir(target_path)
                    if prompt_to_optimize == "system":
                        # Optimizing system prompt - save new system, copy user
                        save_prompt_file(
                            os.path.join(target_path, "system_prompt.txt"), optimized_prompt
                        )
                        save_prompt_file(
                            os.path.join(target_path, "user_prompt.txt"), user_prompt
                        )
                    else:
                        # Optimizing user prompt - copy system, save new user
                        save_prompt_file(
                            os.path.join(target_path, "system_prompt.txt"), system_prompt
                        )
                        save_prompt_file(
                            os.path.join(target_path, "user_prompt.txt"), optimized_prompt
                        )

                    # Get indices of selected examples by _example_id
                    selected_ids = {ex.get("_example_id") for ex in examples if ex.get("_example_id") is not None}
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

                    optimized_label = "system prompt" if prompt_to_optimize == "system" else "user prompt template"
                    st.success(f"Created new run '{target_run_name}' with optimized {optimized_label}!")

                    with st.expander(f"View Optimized {target_label}"):
                        st.code(optimized_prompt)

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
