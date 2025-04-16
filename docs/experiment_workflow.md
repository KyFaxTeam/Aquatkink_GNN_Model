# Experiment Workflow

This document outlines the expected workflow for running experiments, training models, evaluating them, and tracking results using the provided Python scripts.

## 1. Configuration (`src/config.py`)

- Before starting, review `src/config.py` to set base hyperparameters, model architecture choices, and other static settings.
- Note that the `EXPERIMENT_ID` variable in `config.py` acts as a *base* or *default* but is **not** the primary ID used for tracking individual runs.

## 2. Running a Training Experiment (`src/train.py`)

- Execute the `train_pipeline()` function (e.g., by running `python src/train.py`).
- **Experiment ID Generation**:
    - The pipeline starts by calling `increment_experiment_id()` from `src/experiment_utils.py`.
    - This reads the counter in `experiments/experiment_id.txt`, increments it, and writes the new value back.
    - This **new, incremented ID** (e.g., `2`, `3`, etc.) becomes the unique identifier for *this specific run*.
- **Data Loading & Splitting**: Loads data and uses pre-defined train/validation splits from `experiments/splits/`.
- **Model Training**:
    - The model trains for `NUM_EPOCHS`.
    - **Checkpoint Saving**: During training, if the validation AUC-PR improves, the current model state is saved using `paths.checkpoint_path(experiment_id=...)`. This saves the checkpoint to `experiments/<new_experiment_id>_best_model.pt` (e.g., `experiments/2_best_model.pt`).
- **Loading Best Model**: After training (or early stopping), the best saved checkpoint is loaded using `paths.checkpoint_path(experiment_id=...)`, ensuring it loads the checkpoint corresponding to the *current run's* ID.

## 3. Evaluating the Model (`src/evaluate.py`)

- Execute the `evaluate_and_save_predictions()` function (e.g., by running `python src/evaluate.py`).
- **Experiment ID Retrieval**:
    - The script starts by calling `get_last_experiment_id()` from `src/experiment_utils.py`. This retrieves the ID of the *most recently completed training run* (the one whose ID was just written to `experiments/experiment_id.txt`).
- **Data Loading**: Loads the test set data based on splits in `experiments/splits/`.
- **Model Loading**:
    - Loads the best model checkpoint using `paths.checkpoint_path(experiment_id=...)`, using the retrieved `experiment_id`. This ensures it loads the model trained in the corresponding training run (e.g., `experiments/2_best_model.pt`).
- **Evaluation & Metrics**: Calculates performance metrics (AUC-PR, F1, Recall, MRR, Hits@k, FPR) on the test set.
- **Saving Predictions**: Saves detailed predictions (true labels vs. scores) to a CSV file using `paths.predictions_path(experiment_id=...)`. The filename includes the `experiment_id` (e.g., `experiments/2_test_predictions.csv`).
- **Updating Registry**:
    - Calls `make_experiment_entry(...)`, passing the retrieved `experiment_id` and all relevant configuration and performance metrics.
    - Calls `append_to_registry(...)` to add this entry as a new row in `experiments/experiment_registry.csv`.
    - Calls `sort_and_export_ranking(...)` to re-sort the entire registry based on performance metrics and overwrite `experiments/experiment_ranking.csv`.

## 4. Viewing Results (`src/show_top_experiments.py`)

- Execute the script (e.g., `python src/show_top_experiments.py [n]`, where `n` is the optional number of top experiments to show).
- This script reads the pre-sorted `experiments/experiment_ranking.csv` and displays the top `n` entries, providing a quick overview of the best-performing runs.

## Summary of Key Files per Experiment

For a given run assigned `experiment_id = X`:

- **Checkpoint**: `experiments/X_best_model.pt`
- **Predictions**: `experiments/X_test_predictions.csv`
- **Log Directory**: `experiments/runs/X/` (contains TensorBoard logs and config snapshot)
- **Registry Entry**: A row with `experiment_id=X` in `experiments/experiment_registry.csv`
- **Ranking**: The corresponding row's position in `experiments/experiment_ranking.csv`

This workflow ensures each experiment run uses a unique ID for its artifacts (checkpoints, predictions, logs) and that results are systematically tracked and ranked.