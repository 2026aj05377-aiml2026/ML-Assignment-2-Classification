
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configure the application
st.set_page_config(
    page_title="Classification Model Comparison",
    page_icon="📊",
    layout="wide"
)

# Application heading
st.title("Classification Model Comparison")

st.write(
    "Compare five machine learning classification models "
    "using the Breast Cancer Wisconsin Diagnostic Dataset."
)

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the test features and target column",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")

    st.write("Dataset Preview:")
    st.dataframe(test_df.head())

    st.write("Dataset Shape:", test_df.shape)

    st.subheader("Select Classification Model")

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib"
    }

    selected_model = st.selectbox(
        "Choose a model:",
        list(model_files.keys())
    )

    model_path = Path(__file__).resolve().parent / "model" / model_files[selected_model]

    model = joblib.load(model_path)

    st.success(f"{selected_model} model loaded successfully!")

    feature_columns = list(model.feature_names_in_)
    missing_columns = [col for col in feature_columns if col not in test_df.columns]

    if "target" not in test_df.columns:
        st.error("The uploaded CSV must contain a target column.")
        st.stop()

    if missing_columns:
        st.error(f"Missing feature columns: {missing_columns}")
        st.stop()

    X_uploaded = test_df[feature_columns]
    y_uploaded = test_df["target"]

    predictions = model.predict(X_uploaded)

    st.write("First 10 predictions:", predictions[:10])

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        confusion_matrix
    )

    st.subheader("Model Evaluation Results")

    # Calculate metrics using the uploaded test data
    probabilities = model.predict_proba(X_uploaded)[:, 1]

    accuracy = accuracy_score(y_uploaded, predictions)
    precision = precision_score(y_uploaded, predictions, zero_division=0)
    recall = recall_score(y_uploaded, predictions, zero_division=0)
    f1 = f1_score(y_uploaded, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_uploaded, predictions)

    # AUC requires both target classes
    if y_uploaded.nunique() == 2:
        auc = roc_auc_score(y_uploaded, probabilities)
        auc_display = f"{auc:.4f}"
    else:
        auc_display = "N/A"

    # Display all six metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", auc_display)
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_uploaded, predictions, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant (0)", "Actual Benign (1)"],
        columns=["Predicted Malignant (0)", "Predicted Benign (1)"]
    )

    st.dataframe(cm_df)
