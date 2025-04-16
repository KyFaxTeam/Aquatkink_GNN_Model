import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import argparse

from src.config import (
    DATA_DIR, HIDDEN_DIM, NUM_LAYERS, DROPOUT, GNN_TYPE, NORM_TYPE,
    MLP_HIDDEN_DIM, BATCH_SIZE, DEVICE
)
from src import paths
from src.datasets import WDNLeakDataset
from src.models import WDNLeakGNN
from src.utils import compute_auc_pr, compute_mrr, compute_hits_at_k
from src.experiment_utils import get_last_experiment_id, append_to_registry, sort_and_export_ranking, make_experiment_entry


def evaluate_and_save_predictions(experiment_id_to_eval): # Add argument
    # --- Experiment ID ---
    # Use the provided experiment_id instead of fetching the last one
    experiment_id = experiment_id_to_eval
    print(f"[Evaluation] Using specified experiment_id: {experiment_id}")

    # --- Load Optimal Threshold ---
    optimal_threshold = 0.1021 # Default threshold
    threshold_path = paths.optimal_threshold_path(experiment_id)
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                optimal_threshold = float(f.read().strip())
            print(f"[Evaluation] Loaded optimal threshold: {optimal_threshold:.4f} from {threshold_path}")
        except Exception as e:
            print(f"[Warning] Could not load or parse threshold from {threshold_path}: {e}. Using default 0.5.")
    else:
        print(f"[Warning] Optimal threshold file not found: {threshold_path}. Using default 0.5.")


    # Load dataset and test split
    dataset = WDNLeakDataset(root=DATA_DIR)
    # Load test indices for the specified fold
    split_dir = paths.SPLIT_DIR
    test_idx_path = paths.split_index_path("test") # Removed fold argument
    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(f"Test indices file not found: {test_idx_path}")
    test_idx = np.load(test_idx_path)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]
    model = WDNLeakGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        gnn_type=GNN_TYPE,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        norm_type=NORM_TYPE,
    ).to(DEVICE)
    checkpoint = torch.load(paths.checkpoint_path(experiment_id=experiment_id), map_location=DEVICE, weights_only=False) # Removed fold argument
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate
    all_targets, all_preds, all_meta = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            all_targets.append(batch.y.cpu().numpy())
            all_preds.append(out.cpu().numpy())
            # Save scenario index for traceability
            if hasattr(batch, 'scenario_idx'):
                all_meta.extend(batch.scenario_idx.cpu().numpy())
            else:
                all_meta.extend([i] * len(batch.y))

    y_true = np.concatenate(all_targets)
    y_logits = np.concatenate(all_preds) # Rename to indicate logits
    from scipy.special import expit # Import sigmoid function
    y_score = expit(y_logits) # Apply sigmoid to get probabilities

    # Metrics
    auc_pr = compute_auc_pr(y_true, y_score) # AUC-PR uses probabilities
    from sklearn.metrics import f1_score, recall_score, precision_score
    # Use the loaded optimal threshold on probabilities
    y_pred_binary = y_score > optimal_threshold
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    mrr = compute_mrr(y_true, y_score) # MRR uses probabilities
    hits1 = compute_hits_at_k(y_true, y_score, k=1) # Hits@k use probabilities
    hits3 = compute_hits_at_k(y_true, y_score, k=3)
    hits5 = compute_hits_at_k(y_true, y_score, k=5)
    # Calculate FPR based on the thresholded binary predictions
    fpr = np.sum(y_pred_binary & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0

    print(f"Test set evaluation (using threshold={optimal_threshold:.4f}):")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1 (Thresholded): {f1:.4f}")
    print(f"Recall (Thresholded): {recall:.4f}")
    print(f"Precision (Thresholded): {precision:.4f}")
    print(f"MRR: {mrr:.4f}")
    # --- Experiment Management: Registry Update ---
    # experiment_id is already fetched at the beginning
    model_type = "WDNLeakGNN"
    hyperparameters = {
        "lr": globals().get('LEARNING_RATE', None),
        "batch_size": BATCH_SIZE,
        "gamma": globals().get('GAMMA', None),
        "weight_decay": globals().get('WEIGHT_DECAY', None)
    }
    notes = ""
    entry = make_experiment_entry(
        experiment_id=experiment_id,
        model_type=model_type,
        data_dir=DATA_DIR,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        gnn_type=GNN_TYPE,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        norm_type=NORM_TYPE,
        hyperparameters=hyperparameters,
        notes=f"Thresholded metrics (F1, Recall, Precision) calculated using optimal threshold {optimal_threshold:.4f} from validation set.",
        auc_pr=auc_pr,
        f1=f1, # Store F1 calculated with optimal threshold
        recall=recall, # Store Recall calculated with optimal threshold
        precision=precision, # Store Precision calculated with optimal threshold
        mrr=mrr,
        hits1=hits1,
        hits3=hits3,
        hits5=hits5,
        optimal_threshold=optimal_threshold # Also store the threshold used
    )
    append_to_registry(entry)
    sort_and_export_ranking()
    print(f"[Experiment Management] Results appended to registry and ranking updated for experiment_id: {experiment_id}")
    print(f"Hits@1: {hits1:.4f}")
    print(f"Hits@3: {hits3:.4f}")
    print(f"Hits@5: {hits5:.4f}")
    print(f"FPR: {fpr:.4f}")

    # Save predictions and metadata
    results_df = pd.DataFrame({
        "scenario_idx": all_meta,
        "y_true": y_true,
        "y_score_prob": y_score, # Save probabilities
        "y_score_logit": y_logits # Also save original logits for reference
    })
    results_path = paths.predictions_path(experiment_id=experiment_id) # Removed fold argument
    results_df.to_csv(results_path, index=False)
    print(f"Saved detailed predictions to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained WDNLeakGNN model.")
    parser.add_argument("experiment_id", type=int, help="The ID of the experiment to evaluate.")
    args = parser.parse_args()
    evaluate_and_save_predictions(args.experiment_id) # Pass the parsed ID