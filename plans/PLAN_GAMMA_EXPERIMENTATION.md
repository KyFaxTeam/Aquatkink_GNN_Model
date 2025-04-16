# Plan: Experimenting with Focal Loss Gamma

**Objective:** Investigate whether adjusting the `gamma` hyperparameter in the Focal Loss function can improve the model's confidence in predicting the positive class (leaks) and lead to better overall performance, particularly for threshold-based metrics like F1 score.

**Background:**
The current model produces low probability scores even for true positive cases (actual leaks), likely due to the extreme class imbalance in the dataset. While Focal Loss is used to mitigate this, the default `gamma` value might not be optimal. Increasing `gamma` forces the model to focus more intensely on hard-to-classify examples (often the rare positive class), potentially increasing the output probabilities for true positives. However, setting `gamma` too high could destabilize training or negatively impact ranking metrics.

**Proposed Steps:**

1.  **Define Gamma Values:** Select a range of `gamma` values to test. Suggested values:
    *   `1.0` (Lower focus on hard examples)
    *   `2.0` (Current/Baseline value, assuming from docs)
    *   `3.0` (Higher focus)
    *   `4.0` (Even higher focus)
    *(Adjust these based on current config and desired exploration range)*

2.  **Modify Configuration/Training Script:** Update `src/config.py` or modify `src/train.py` to accept `gamma` as a parameter (e.g., via command-line argument). Ensure each experiment run uses the intended `gamma` value.

3.  **Train Multiple Models:** Execute the training pipeline (`src/train.py`) separately for each selected `gamma` value. Each run should generate a new `experiment_id`.

4.  **Evaluate Each Model:** For each completed experiment (`experiment_id`):
    *   **Find Optimal Threshold:** Run the notebook cell (or a dedicated script) to load the validation predictions (`val_pred_scores_{id}.npz`) and calculate the optimal F1 threshold based *only* on that experiment's validation results. Save this threshold (`optimal_threshold_{id}.txt`).
    *   **Run Test Set Evaluation:** Execute the evaluation script (`src/evaluate.py`), ensuring it loads the correct model checkpoint and the corresponding optimal threshold (`optimal_threshold_{id}.txt`). The script should calculate and report:
        *   Threshold-independent metrics: AUC-PR, MRR, Hits@k
        *   Threshold-dependent metrics (using the loaded optimal threshold): F1, Recall, Precision, FPR.

5.  **Compare Results:** Collate the test set metrics from all experiments (different `gamma` values). Analyze the trade-offs:
    *   Did increasing `gamma` improve F1/Recall at the optimal threshold?
    *   How did AUC-PR, MRR, and Hits@k change?
    *   Was training stability affected?

6.  **Select Best Gamma:** Based on the comparison, determine the `gamma` value that offers the best performance according to the project's priorities (e.g., maximizing F1, maximizing Recall while maintaining reasonable Precision, or maximizing ranking metrics).

**Workflow Diagram:**

```mermaid
graph TD
    A[Start: Define Gamma Values (e.g., 1.0, 2.0, 3.0, 4.0)] --> B{Loop for each Gamma Value};
    B -- Run Experiment --> C(Modify Config/Train Script with Current Gamma);
    C --> D(Train Model);
    D --> E(Generate val_pred_scores_{id}.npz);
    E --> F(Find Optimal Threshold for this Gamma);
    F --> G(Save optimal_threshold_{id}.txt);
    G --> H(Run evaluate.py for this Gamma);
    H -- Uses optimal_threshold_{id}.txt --> H;
    H --> I(Record Test Metrics for this Gamma);
    I --> B;
    B -- All Gammas Tested --> J(Collate & Compare Metrics Across All Experiments);
    J --> K(Select Best Gamma Value);

    style D fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px
```

**Next Steps:** Implement the changes required to run these experiments (modifying config/scripts) and then execute the training and evaluation runs.