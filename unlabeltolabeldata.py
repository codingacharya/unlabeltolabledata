import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO

# Streamlit Page Configuration
st.set_page_config(page_title="Unstructured Data Separator", layout="wide")

# Title
st.title("📊 Unstructured Data Separator & Labeling")

# Upload File
uploaded_file = st.file_uploader("📂 Upload your CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    # Read Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raw_text = uploaded_file.read().decode("utf-8")
        df = pd.DataFrame({'Text': raw_text.split("\n")})  # Convert text to DataFrame

    st.subheader("📜 Uploaded Data Preview")
    st.dataframe(df.head())

    # Feature Selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. Please upload a dataset with numerical features.")
    else:
        st.sidebar.subheader("⚙️ Clustering Settings")
        n_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)

        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[numeric_cols])

        # Display Transformed Data
        st.subheader("📌 Clustered Data")
        st.dataframe(df)

        # PCA for 2D Visualization
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df[numeric_cols])
        df["PCA_1"] = df_pca[:, 0]
        df["PCA_2"] = df_pca[:, 1]

        # Visualization
        st.subheader("📊 Clustering Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(df["PCA_1"], df["PCA_2"], c=df["Cluster"], cmap="viridis", alpha=0.7)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("K-Means Clustering Visualization")
        plt.colorbar(scatter, label="Cluster")
        st.pyplot(fig)

        # Download Processed Data
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        st.download_button(label="📥 Download Transformed Data",
                           data=output,
                           file_name="structured_data.csv",
                           mime="text/csv")
