# Stage 6: Evaluation and Error Analysis – Detailed Plan

---

## Objective
Thoroughly evaluate the trained GNN model’s performance on the test set, analyze its errors, and gain insights for further improvement and interpretability.

---

## 1. Automated Evaluation

- **Compute all relevant metrics on the test set:**
  - AUC-PR (Area Under the Precision-Recall Curve)
  - Recall, Precision, F1-Score (for the "leak" class)
  - MRR (Mean Reciprocal Rank)
  - Hits@k (k=1, 3, 5)
  - FPR (False Positive Rate on no-leak cases)
- **Save detailed prediction results:**
  - For each test scenario, store predicted probabilities, true labels, and scenario metadata in a CSV or HDF5 file for further analysis.

**Justification:**  
Comprehensive metrics and result logging are essential for understanding both detection and localization performance, especially in imbalanced settings.

---

## 2. Error Analysis

- **Identify and categorize errors:**
  - False Negatives (missed leaks)
  - False Positives (false alarms)
  - Analyze the distribution of errors by pipe, scenario type, leak severity, sensor placement, etc.
- **Visualize error cases:**
  - For selected scenarios, plot the network graph with:
    - True leak location
    - Top-k predicted pipes
    - Sensor locations and their readings
  - Use color-coding to distinguish true/false positives/negatives.
- **Aggregate error statistics:**
  - Which pipes or regions are most error-prone?
  - Are certain types of leaks (e.g., small, remote) harder to detect?

**Justification:**  
Error analysis reveals model weaknesses, guides future improvements, and helps interpret model decisions.

---

## 3. Baseline Comparison

- **Implement and evaluate baseline methods:**
  - Simple thresholding on pressure drops.
  - Tabular ML classifier (e.g., logistic regression) on aggregated features.
- **Compare GNN performance to baselines using the same metrics.**

**Justification:**  
Demonstrates the added value of the GNN and contextualizes its performance.

---

## 4. In-Depth Analysis in Notebooks

- **Provide Jupyter notebooks for:**
  - Interactive exploration of predictions and errors.
  - Custom visualizations (e.g., pressure time series, embedding projections).
  - Case studies of challenging scenarios.

**Justification:**  
Notebooks enable flexible, reproducible, and collaborative analysis.

---

## 5. Reporting and Documentation

- **Summarize findings in markdown reports:**
  - Key metrics, error patterns, and visualizations.
  - Lessons learned and recommendations for next steps.

---

## Mermaid Diagram: Evaluation & Error Analysis Workflow

```mermaid
flowchart TD
    A[Trained Model + Test Set] --> B[Compute Metrics (AUC-PR, F1, MRR, Hits@k)]
    B --> C[Save Predictions & Metadata]
    C --> D[Error Categorization (FP, FN, etc.)]
    D --> E[Visualize Network & Errors]
    E --> F[Aggregate Error Stats]
    F --> G[Compare to Baselines]
    G --> H[Notebook Analysis & Reporting]