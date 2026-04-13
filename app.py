from pathlib import Path
import pandas as pd
import streamlit as st

from src.data import load_cluster_data
from src.train import train_and_save_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation")

if st.button("Train / Refresh Clustering Model"):
    best_k, results_df = train_and_save_model()
    st.success(f"Training complete. Best number of clusters: {best_k}")
    st.dataframe(results_df)

data_path = ROOT / "data" / "raw" / "mall_customers.csv"
if data_path.exists():
    df = load_cluster_data(data_path)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
else:
    st.warning("Place mall_customers.csv inside data/raw before using the app.")
