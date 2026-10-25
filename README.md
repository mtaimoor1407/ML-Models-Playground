# ML Models Playground

A **Streamlit learning prototype** for experimenting with classification, regression, and clustering on tabular CSV data. The interface lets you select a model, adjust selected hyperparameters, train it, and inspect predictions and task-specific metrics.

## Models implemented

| Task | Available models |
| --- | --- |
| Classification | Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Voting Classifier |
| Regression | Decision Tree, Random Forest |
| Clustering | K-means |

The voting classifier can combine Logistic Regression, Decision Tree, and SVM base estimators.

## Intended workflow

1. Upload a training CSV and inspect its preview and summary statistics.
2. Select the target column, or choose `None` for clustering.
3. Choose a model and adjust the available settings.
4. Train the model.
5. Upload a separate test CSV with matching feature names and types.
6. Inspect predictions and, where labels are available, evaluation metrics.

Rows with missing values are dropped. Categorical features are one-hot encoded; numeric features are standardized. The model and preprocessing artifacts are written to `models/` and `artifacts/`.

## Metrics and plots in the interface

- **Classification:** accuracy, weighted precision/recall/F1, confusion matrix, and binary ROC/AUC where probability prediction is supported.
- **Regression:** R², MSE, MAE, residual scatter plot, and residual distribution.
- **Clustering:** K-means inertia and a scatter plot using the first two processed features.

Results depend on the uploaded data, preprocessing, and selected model; this repository does not represent a single benchmark score.

## Local development

```bash
git clone https://github.com/mtaimoor1407/ML-Models-Playground.git
cd ML-Models-Playground
python -m venv .venv
```

Activate the environment using `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run from the repository root so the relative `models/` and `artifacts/` paths resolve as expected.

## Current status and limitations

The repository is a prototype and needs reliability fixes before it can be presented as a fully tested application:

- In `app.py`, `model.predict(X_test_processed)` currently runs outside the test-upload guard. On a fresh start, the variables may not exist, producing a `NameError`. Training also needs a guard when no training CSV has been supplied.
- The fitted target-label encoder should be reused for test labels rather than refitted on them.
- Task detection is a heuristic: an object target or a target with at most 20 unique values is treated as classification. Some regression datasets need an explicit task override.
- Training and test CSVs must have compatible feature schemas. The existing preprocessing assumes both categorical and numeric transformations can be applied.
- Clustering evaluation and plots need input checks, including enough processed dimensions for a two-dimensional plot.
- Repeatable comparisons need a consistent held-out split and controlled random seeds.

## Repository guide

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, training, prediction, and visualizations |
| `src/preprocess.py` | Task detection, feature transformations, target encoding |
| `src/model_selection.py` | Available estimators by task |
| `data/` | Data files |
| `models/` | Trained model artifacts |
| `artifacts/` | Saved encoders and scaler |

**Tools:** Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib, Seaborn, joblib.
