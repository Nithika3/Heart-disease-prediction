import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

# Page title
st.title("📊 Model Comparison")

# Load precomputed metrics (example values shown below)
comparison_data = {
    "Model": ["XGBoost", "Random Forest", "Logistic Regression"],
    "Accuracy": [0.89, 0.85, 0.81],
    "AUC Score": [0.92, 0.87, 0.84],
    "Precision": [0.88, 0.84, 0.80],
    "Recall": [0.90, 0.82, 0.78],
    "F1 Score": [0.89, 0.83, 0.79]
}

df_comparison = pd.DataFrame(comparison_data)

# Show metrics table
st.subheader("📈 Performance Metrics")
st.dataframe(df_comparison)

# Load test data and probabilities for ROC curve
y_test = joblib.load("y_test.joblib")
y_prob_xgb = joblib.load("y_prob_xgb.joblib")
y_prob_rf = joblib.load("y_prob_rf.joblib")
y_prob_log = joblib.load("y_prob_log.joblib")

# Plot ROC Curves
st.subheader("📉 ROC Curve Comparison")

fpr1, tpr1, _ = roc_curve(y_test, y_prob_xgb)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_rf)
fpr3, tpr3, _ = roc_curve(y_test, y_prob_log)

fig, ax = plt.subplots()
ax.plot(fpr1, tpr1, label=f"XGBoost (AUC = {auc(fpr1, tpr1):.2f})")
ax.plot(fpr2, tpr2, label=f"Random Forest (AUC = {auc(fpr2, tpr2):.2f})")
ax.plot(fpr3, tpr3, label=f"Logistic Regression (AUC = {auc(fpr3, tpr3):.2f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
st.pyplot(fig)
