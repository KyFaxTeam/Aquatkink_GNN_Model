import csv
import os
import sys

RANKING_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_ranking.csv"))

def show_top_experiments(n=5):
    if not os.path.exists(RANKING_FILE):
        print("Ranking file not found. Run evaluation to generate experiment_ranking.csv.")
        return
    with open(RANKING_FILE, "r", newline="") as csvfile:
        reader = list(csv.DictReader(csvfile))
        print(f"Top {min(n, len(reader))} Experiments:")
        if not reader:
            print("No experiments found.")
            return
        header = reader[0].keys()
        print("\t".join(header))
        for row in reader[:n]:
            print("\t".join(str(row[h]) for h in header))

if __name__ == "__main__":
    n = 5
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    show_top_experiments(n)