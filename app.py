import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")
st.markdown("ML-powered fraud detection with adjustable risk threshold")

# Sidebar threshold slider
st.sidebar.header("Model Settings")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

st.sidebar.markdown(f"Current Threshold: **{threshold}**")

# ==========================
# 1. Manual Prediction Section
# ==========================
st.header("🔍 Single Transaction Prediction")

features = []
for i in range(30):  # Adjust if feature count differs
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict Transaction"):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    probability = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    st.write(f"Fraud Probability: **{probability:.4f}**")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

# ==========================
# 2. Bulk CSV Upload
# ==========================
st.header("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    scaled_data = scaler.transform(data)
    probabilities = model.predict_proba(scaled_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    data["Fraud_Probability"] = probabilities
    data["Prediction"] = predictions

    st.dataframe(data.head())

# ==========================
# 3. Model Performance Section
# ==========================
st.header("📊 Model Performance")

# Optional: Load test data if available
try:
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()

    X_test_scaled = scaler.transform(X_test)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha="center", va="center")

        st.pyplot(fig)

    # ROC Curve
    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")

        st.pyplot(fig2)

except:
    st.info("Upload X_test.csv and y_test.csv to enable performance metrics.")
