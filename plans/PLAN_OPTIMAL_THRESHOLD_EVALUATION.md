# Plan: Optimal Threshold Evaluation for Leak Localization Model

## Context: Interpreting Low Predicted Probabilities

Based on the analysis of the validation set predictions, we observed the following:

1.  **Model Output is a Probability:** The GNN model outputs a probability (between 0 and 1) for each edge, representing the estimated likelihood of a leak (class 1) on that edge. It doesn't directly predict "1" or "0".
2.  **Low Probabilities for Actual Leaks:** Even for edges that truly had leaks (y\_true=1), the predicted probabilities were low, typically ranging from 0.075 to 0.105, and never reaching the standard 0.5 threshold.
3.  **Class Imbalance Impact:** This is largely due to the extreme class imbalance in the data (many non-leaky edges, very few leaky ones). The model learns that predicting low probabilities is generally correct, making it difficult to assign high confidence to the rare positive class without increasing false positives significantly. The probabilities are effectively "biased" low due to the data distribution.
4.  **Metric Implications:** Using a fixed 0.5 threshold resulted in zero F1, Recall, and Precision because no prediction reached this level, leading to zero calculated True Positives.

## Strategy: Focus on Ranking and Optimal Thresholding

Given the low absolute probabilities, a more effective evaluation strategy involves:

1.  **Focusing on Ranking:** Prioritize metrics that evaluate the model's ability to rank the true leak highly, even if its absolute probability is low. Key metrics include **AUC-PR**, **MRR**, and **Hits@k**.
2.  **Finding an Optimal Threshold:** Determine a data-driven probability threshold from the validation set predictions. This threshold should be chosen to optimize a relevant metric (e.g., maximize F1 score) for converting probabilities into binary predictions (0 or 1). This threshold will likely be much lower than 0.5.
3.  **Re-evaluating:** Use this optimal threshold to calculate threshold-dependent metrics (F1, Recall, Precision) on the *test set* for a more realistic performance assessment.

## Proposed Plan Steps

1.  **Analyze Validation Scores:** Load the saved validation scores (`y_true`, `y_score` from the `.npz` file). Implement logic (likely in the analysis notebook) to iterate through potential probability thresholds, calculate the F1 score (or another chosen metric) at each threshold, and identify the threshold that yields the maximum F1 score.
2.  **Save/Pass Optimal Threshold:** Store the determined optimal threshold. This could be saved to a file or passed as an argument.
3.  **Modify Evaluation Script (`src/evaluate.py`):**
    *   Update the script to load or accept the optimal threshold.
    *   When calculating F1, Recall, Precision, use `y_pred = y_score > optimal_threshold` instead of `y_pred = y_score > 0.5`.
    *   Ensure the script continues to calculate and report the threshold-independent metrics (AUC-PR, MRR, Hits@k).
    *   Report which threshold was used for the threshold-dependent metrics.
4.  **Run Evaluation:** Execute the modified `evaluate.py` script on the test set predictions to obtain the final performance metrics using the optimized threshold.

## Workflow Diagram

```mermaid
graph TD
    A[Training Complete] --> B(Load Validation Scores: y_true, y_score);
    B --> C{Find Optimal Threshold};
    C -- Iterate Thresholds & Calculate F1 --> C;
    C -- Best Threshold Found --> D(Save/Record Optimal Threshold);
    D --> E[Modify evaluate.py];
    E -- Use Optimal Threshold --> E;
    E --> F[Run evaluate.py on Test Set];
    F --> G(Report Metrics: AUC-PR, MRR, Hits@k AND F1/Recall/Precision @ Optimal Threshold);

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px