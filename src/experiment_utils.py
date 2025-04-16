import os
import csv
import json
from datetime import datetime

EXPERIMENT_ID_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_id.txt"))
EXPERIMENT_REGISTRY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_registry.csv"))
EXPERIMENT_RANKING_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_ranking.csv"))

EXPERIMENT_REGISTRY_COLUMNS = [
    "experiment_id", "date", "model_type", "data_dir", "node_in_dim", "edge_in_dim", "hidden_dim", "num_layers",
    "gnn_type", "mlp_hidden_dim", "dropout", "norm_type", "hyperparameters", "notes", "AUC-PR", "F1", "recall", "precision", # Added precision
    "mrr", "hits@1", "hits@3", "hits@5", "optimal_threshold" # Added new metrics
]

def get_last_experiment_id():
    if not os.path.exists(EXPERIMENT_ID_FILE):
        return 0
    with open(EXPERIMENT_ID_FILE, "r") as f:
        return int(f.read().strip())

def increment_experiment_id():
    last_id = get_last_experiment_id()
    new_id = last_id + 1
    with open(EXPERIMENT_ID_FILE, "w") as f:
        f.write(str(new_id))
    return new_id

def append_to_registry(entry: dict):
    file_exists = os.path.exists(EXPERIMENT_REGISTRY_FILE)
    with open(EXPERIMENT_REGISTRY_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EXPERIMENT_REGISTRY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def sort_and_export_ranking(sort_by=["AUC-PR", "F1", "recall"], descending=True):
    if not os.path.exists(EXPERIMENT_REGISTRY_FILE):
        return
    with open(EXPERIMENT_REGISTRY_FILE, "r", newline="") as csvfile:
        reader = list(csv.DictReader(csvfile))
    # Convert metrics to float for sorting
    for row in reader:
        for key in sort_by:
            try:
                row[key] = float(row[key])
            except Exception:
                row[key] = float("-inf")
    sorted_rows = sorted(reader, key=lambda x: tuple(x[k] for k in sort_by), reverse=descending)
    with open(EXPERIMENT_RANKING_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EXPERIMENT_REGISTRY_COLUMNS)
        writer.writeheader()
        for row in sorted_rows:
            # Convert metrics back to string for writing
            for key in sort_by:
                row[key] = str(row[key])
            writer.writerow(row)

def make_experiment_entry(
    experiment_id,
    model_type,
    data_dir,
    node_in_dim,
    edge_in_dim,
    hidden_dim,
    num_layers,
    gnn_type,
    mlp_hidden_dim,
    dropout,
    norm_type,
    hyperparameters,
    notes,
    auc_pr,
    f1,
    recall,
    precision, # Added precision parameter
    mrr,
    hits1,
    hits3,
    hits5,
    optimal_threshold # Added new metric parameters
):
    return {
        "experiment_id": experiment_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "data_dir": data_dir,
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "gnn_type": gnn_type,
        "mlp_hidden_dim": mlp_hidden_dim,
        "dropout": dropout,
        "norm_type": norm_type,
        "hyperparameters": json.dumps(hyperparameters),
        "notes": notes,
        "AUC-PR": auc_pr,
        "F1": f1,
        "recall": recall,
        "precision": precision, # Added precision to the returned dict
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@5": hits5,
        "optimal_threshold": optimal_threshold # Added new metrics to the returned dict
    }