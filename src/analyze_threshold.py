import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
import argparse
import os

def find_optimal_f1_threshold(y_true, y_score):
    """
    Finds the threshold that maximizes the F1 score.

    Args:
        y_true (np.array): True binary labels.
        y_score (np.array): Predicted probabilities.

    Returns:
        tuple: (best_threshold, best_f1_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # Calculate F1 score for each threshold, handling potential division by zero
    # Note: thresholds array is one element shorter than precisions/recalls
    f1_scores = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        # Use precision and recall values corresponding to the *next* point
        # as precision_recall_curve returns thresholds for operating points
        p = precisions[i+1]
        r = recalls[i+1]
        if p + r == 0:
            f1_scores[i] = 0.0
        else:
            f1_scores[i] = 2 * (p * r) / (p + r)

    # Find the threshold that maximizes F1
    if len(f1_scores) == 0:
        # Handle case with no thresholds (e.g., all predictions are the same)
        return 0.5, 0.0 # Default threshold, F1 is 0
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx]

    return best_threshold, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction file to find optimal F1 threshold.")
    parser.add_argument("prediction_file", type=str, help="Path to the prediction CSV file (e.g., experiments/2/2_test_predictions.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.prediction_file):
        print(f"Error: Prediction file not found at {args.prediction_file}")
        exit(1)

    print(f"Analyzing file: {args.prediction_file}")
    df = pd.read_csv(args.prediction_file)

    # Check expected columns - adjust if needed based on evaluate.py output
    if 'y_true' not in df.columns or 'y_score_prob' not in df.columns:
        print(f"Error: CSV must contain 'y_true' and 'y_score_prob' columns.")
        # Fallback check for older format
        if 'y_score' in df.columns and 'y_score_prob' not in df.columns:
             print("Found 'y_score' column, assuming it contains probabilities.")
             prob_col = 'y_score'
        else:
            exit(1)
    else:
        prob_col = 'y_score_prob'


    y_true = df['y_true'].values
    y_score = df[prob_col].values

    # Ensure y_true is binary
    if not np.all(np.isin(y_true, [0, 1])):
        print("Warning: 'y_true' contains non-binary values. Attempting to treat non-zero as 1.")
        y_true = (y_true != 0).astype(int)

    # Check if there are any positive samples
    if np.sum(y_true) == 0:
        print("Error: No positive samples (y_true=1) found in the data. Cannot calculate F1 score.")
        exit(1)

    print(f"Loaded {len(y_true)} predictions. Number of positive samples: {np.sum(y_true)}")

    best_threshold, best_f1 = find_optimal_f1_threshold(y_true, y_score)

    print(f"\nAnalysis Complete:")
    print(f"  Optimal Threshold (Maximizing F1 on this set): {best_threshold:.6f}")
    print(f"  Maximum F1 Score (at this threshold):          {best_f1:.6f}")