# Student Outcome Classification Using Machine Learning

## Problem statement

This project classifies a student's recorded academic outcome as **Dropout**, **Enrolled**, or **Graduate** using demographic, admission, and academic attributes from the supplied dataset.

Five classification algorithms are trained and evaluated on the same stratified 20% holdout test set. The evaluation uses Accuracy, macro one-vs-rest AUC, macro Precision, macro Recall, macro F1-score, and Matthews Correlation Coefficient (MCC).

### Feature selection

All 36 predictor columns supplied with the dataset are retained; no separate feature-selection algorithm is applied. `Target` is excluded from the input matrix and used only as the label.

This is a **retrospective classification task** using the full student record, including semester-performance columns. Those semester results would not be available for a strictly pre-enrollment prediction. Scaling is fitted only on training data inside the applicable pipelines.

## Dataset description

- **Source:** UCI Machine Learning Repository — [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
- **Instances:** 4,424 student records.
- **Features:** 36 input columns and one target column, `Target`.
- **Classes:** Dropout, Enrolled, and Graduate.
- **Preprocessing:** The semicolon-separated `dataset_full.csv` is read as UTF-8 with BOM support. Header whitespace, including the stray tab in one header, is stripped. `Target` is label-encoded for training. A stratified 80/20 split uses `random_state=42`; the test rows are saved as `test_data.csv` with original string labels. Scaling is fitted inside the relevant model pipelines.

## GitHub repository

[Student Outcome Lab — GitHub Repository](https://github.com/omvishwas/Student-Outcome-Lab)

The repository includes the source code, `requirements.txt`, this README, test data, saved model artifacts for all five classifiers, and preprocessing artifacts.

## Models used

The following five classifiers are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (k=11)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Performance comparison

All models are evaluated on the same stratified test set. Precision, Recall, and F1 are macro-averaged; multiclass AUC is calculated using macro one-vs-rest averaging.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7684 | 0.8778 | 0.7070 | 0.6754 | 0.6826 | 0.6150 |
| Decision Tree | 0.7458 | 0.8511 | 0.6890 | 0.6609 | 0.6698 | 0.5780 |
| K-Nearest Neighbors (k=11) | 0.6949 | 0.8082 | 0.6245 | 0.5787 | 0.5822 | 0.4874 |
| Gaussian Naive Bayes | 0.6599 | 0.7908 | 0.5669 | 0.5586 | 0.5581 | 0.4303 |
| Support Vector Machine (RBF) | 0.7582 | 0.8664 | 0.7051 | 0.6676 | 0.6785 | 0.5986 |

### Model-wise observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Best overall on this split. It achieves the highest Accuracy (0.7684), AUC (0.8778), macro Precision (0.7070), macro Recall (0.6754), macro F1 (0.6826), and MCC (0.6150). |
| Decision Tree | Accuracy is 0.7458 and AUC is 0.8511, with macro F1 of 0.6698 and MCC of 0.5780. It trails Logistic Regression and the RBF SVM across the reported metrics. |
| K-Nearest Neighbors (k=11) | Accuracy is 0.6949 and AUC is 0.8082. Its macro Precision (0.6245), Recall (0.5787), F1 (0.5822), and MCC (0.4874) are below the Decision Tree, Logistic Regression, and SVM results. |
| Gaussian Naive Bayes | Lowest scores in this comparison: Accuracy 0.6599, AUC 0.7908, macro F1 0.5581, and MCC 0.4303. |
| Support Vector Machine (RBF) | Runner-up overall: Accuracy 0.7582 and AUC 0.8664, with macro F1 0.6785 and MCC 0.5986. Its macro Precision (0.7051) and Recall (0.6676) are close to Logistic Regression. |

### Overall winner

**Logistic Regression** is the overall winner on this stratified 80/20 split. It leads on all six reported metrics. This conclusion is specific to the selected split and evaluation metrics.

## Streamlit application

The deployed application provides the following functionality:

- Select one of the five saved models and evaluate an uploaded CSV or the bundled `test_data.csv`.
- View Accuracy, AUC, Precision, Recall, F1-score, and MCC, along with a confusion matrix and classification report.
- Compare saved models, inspect one-vs-rest ROC curves, and download row-level predictions with class probabilities.
- Load existing Joblib model artifacts without retraining models on uploaded test data.

**Live application:** [Student Outcome Lab — Streamlit](https://student-outcome-lab-n28w8pdcdgde8bzyt2qjdu.streamlit.app/)
