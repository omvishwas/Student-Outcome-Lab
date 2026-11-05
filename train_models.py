"""Train and save the five requested student outcome classifiers."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset_full.csv"
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "Target"
RANDOM_STATE = 42


def main() -> None:
    # Load the semicolon-delimited source and normalize its stray-tab header.
    data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} was not found.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    # Encode labels for sklearn while retaining original labels in the held-out CSV.
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET_COLUMN].astype(str))
    X = data.drop(columns=TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = encoder.inverse_transform(y_test)
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    # Scaling is kept inside each pipeline so it is learned only from training rows.
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ]),
        "naive_bayes": Pipeline([
            ("classifier", GaussianNB()),
        ]),
        "support_vector_machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    rows = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "AUC": roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="macro"
            ),
            "Precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "Recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "F1": f1_score(y_test, predicted, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predicted),
        })
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
