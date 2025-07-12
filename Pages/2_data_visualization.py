import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.title("📊 Dataset Visualization")
st.write("""
This page provides insights into the dataset used for heart disease prediction.
Below are some basic visualizations to help understand the distribution and importance of features.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    # Clean column names: remove spaces, standardize casing
    df.columns = df.columns.str.strip()

    # Rename target column if it's labeled as "Heart Disease"
    if 'Heart Disease' in df.columns:
        df = df.rename(columns={'Heart Disease': 'target'})

    return df

df = load_data()

# Show the dataset (optional)
if st.checkbox("Show Raw Dataset"):
    num_rows = st.slider("Select number of rows to view:", min_value=5, max_value=len(df), value=10)
    st.dataframe(df.head(num_rows))

# Feature selection dropdown
selected_feature = st.selectbox("Select a feature to visualize", df.columns)

# Distribution plot
fig, ax = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax)
plt.title(f"Distribution of {selected_feature}")
st.pyplot(fig)

# Correlation heatmap
# st.subheader("Correlation Heatmap")
# fig2, ax2 = plt.subplots(figsize=(10, 6))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
# st.pyplot(fig2)

# Target variable distribution (Heart Disease)
if 'target' in df.columns:
    st.subheader("Target Class Distribution")
    class_counts = df['target'].value_counts()
    st.bar_chart(class_counts)

    # Optionally show as percentages
    percentages = class_counts / class_counts.sum() * 100
    st.write("Class distribution (%):")
    st.write(percentages.round(2))
else:
    st.warning("No 'target' column found in the dataset.")
