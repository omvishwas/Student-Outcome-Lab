"""Streamlit dashboard for the saved student success classifiers."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_LABELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}


@st.cache_resource
def load_assets():
    """Load pre-trained estimators and metadata once per Streamlit process."""
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    estimators = {
        label: joblib.load(MODEL_DIR / f"{filename}.joblib")
        for label, filename in MODEL_LABELS.items()
    }
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(MODEL_DIR / "metrics_summary.csv")
    return encoder, estimators, feature_names, metrics


def score_rows(actual, predicted, probabilities, encoder):
    """Calculate the same six metrics used during model training."""
    encoded_actual = encoder.transform(actual.astype(str))
    encoded_predicted = encoder.transform(predicted.astype(str))
    return {
        "Accuracy": accuracy_score(encoded_actual, encoded_predicted),
        "AUC": roc_auc_score(
            encoded_actual, probabilities, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(
            encoded_actual, encoded_predicted, average="macro", zero_division=0
        ),
        "Recall": recall_score(encoded_actual, encoded_predicted, average="macro", zero_division=0),
        "F1": f1_score(encoded_actual, encoded_predicted, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(encoded_actual, encoded_predicted),
    }


def main():
    st.set_page_config(page_title="Student Outcome Lab", page_icon="🎓", layout="wide")
    st.title("Student Outcome Lab")
    st.write("Explore saved classifiers for predicting dropout, enrollment, and graduation.")

    try:
        encoder, estimators, feature_names, saved_metrics = load_assets()
    except FileNotFoundError:
        st.error("Trained model files are missing. Run `python train_models.py` from the project folder first.")
        st.stop()

    with st.sidebar:
        st.header("Evaluation data")
        use_bundled = st.checkbox("Use bundled test_data.csv", value=True)
        uploaded_file = st.file_uploader("Or upload a test CSV", type=["csv"])
        selected_model = st.selectbox("Classifier", list(MODEL_LABELS))

    if not use_bundled and uploaded_file is None:
        st.info("Choose the bundled test set or upload a CSV to continue.")
        st.stop()
    try:
        test_frame = pd.read_csv(ROOT / "test_data.csv") if use_bundled else pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"The CSV could not be read. Check its delimiter and encoding. Details: {error}")
        st.stop()

    test_frame.columns = test_frame.columns.str.strip()
    missing = [column for column in feature_names if column not in test_frame.columns]
    if missing:
        st.error("The CSV is missing required feature columns: " + ", ".join(missing))
        st.stop()
    if "Target" not in test_frame.columns:
        st.error("The CSV must include a `Target` column with the known class labels.")
        st.stop()
    if test_frame["Target"].isna().any():
        st.error("The `Target` column contains blank values. Remove those rows and upload again.")
        st.stop()

    X_test = test_frame[feature_names]
    actual = test_frame["Target"].astype(str)
    try:
        encoder.transform(actual)
    except ValueError:
        st.error("The `Target` column contains a class label that was not present during training.")
        st.stop()
    absent_classes = [label for label in encoder.classes_ if label not in set(actual)]
    if absent_classes:
        st.error(
            "The test CSV must contain at least one row for every trained class to calculate "
            "macro AUC and ROC curves. Missing classes: " + ", ".join(absent_classes)
        )
        st.stop()

    chosen = estimators[selected_model]
    predictions_encoded = chosen.predict(X_test)
    predictions = encoder.inverse_transform(predictions_encoded.astype(int))
    probabilities = chosen.predict_proba(X_test)
    values = score_rows(actual, predictions, probabilities, encoder)

    st.subheader(f"{selected_model} performance")
    metric_tiles = st.columns(6)
    for tile, (name, value) in zip(metric_tiles, values.items()):
        tile.metric(name, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(actual, predictions, labels=encoder.classes_)
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=axis,
        )
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Actual outcome")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            actual, predictions, labels=encoder.classes_,
            target_names=encoder.classes_, zero_division=0, output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("One-vs-rest ROC curves")
    actual_binary = label_binarize(actual, classes=encoder.classes_)
    figure, axis = plt.subplots(figsize=(8, 5))
    for class_index, class_name in enumerate(encoder.classes_):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            actual_binary[:, class_index], probabilities[:, class_index]
        )
        axis.plot(false_positive_rate, true_positive_rate, label=class_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC by outcome class")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Saved model comparison")
    st.dataframe(saved_metrics.set_index("Model").round(4), use_container_width=True)

    download_frame = test_frame.copy()
    download_frame["Predicted_Target"] = predictions
    for class_index, class_name in enumerate(encoder.classes_):
        download_frame[f"Probability_{class_name}"] = probabilities[:, class_index]
    st.download_button(
        "Download predictions as CSV",
        data=download_frame.to_csv(index=False).encode("utf-8"),
        file_name="student_outcome_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
