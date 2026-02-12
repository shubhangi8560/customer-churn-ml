import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ==========================
# Title
# ==========================
st.title("Bank Customer Churn Prediction")

# ==========================
# Model Selection
# ==========================
model_option = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# ==========================
# Load Selected Model
# ==========================
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_files[model_option])
scaler = joblib.load("model/scaler.pkl")

# ==========================
# File Upload
# ==========================
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unnecessary columns if present
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Encode categorical columns
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

    if "Geography" in data.columns:
        data["Geography"] = data["Geography"].astype("category").cat.codes

    # Separate features & target
    if "Exited" in data.columns:
        X = data.drop("Exited", axis=1)
        y = data["Exited"]

        X = scaler.transform(X)

        predictions = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Metrics
        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y, predictions))
        st.write("AUC Score:", roc_auc_score(y, probs))

        st.text("Classification Report")
        st.text(classification_report(y, predictions))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

    else:
        st.error("Uploaded file must contain 'Exited' column.")
