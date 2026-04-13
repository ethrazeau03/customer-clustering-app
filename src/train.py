from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .data import load_cluster_data
from .logger import setup_logger

LOGGER = setup_logger("cluster_train")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def evaluate_clusters(df: pd.DataFrame, features: list[str], k_values=range(3, 9)):
    results = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(df[features])
        score = silhouette_score(df[features], labels)
        results.append({"k": k, "silhouette_score": float(score)})
    return pd.DataFrame(results)


def train_and_save_model():
    df = load_cluster_data(ROOT / "data" / "raw" / "mall_customers.csv")
    features = ["Age", "Annual_Income", "Spending_Score"]
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Missing expected feature: {feature}")

    results_df = evaluate_clusters(df, features)
    best_k = int(results_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(df[features])

    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(features, ARTIFACT_DIR / "feature_columns.joblib")
    results_df.to_csv(ARTIFACT_DIR / "cluster_scores.csv", index=False)
    df_with_clusters.to_csv(ARTIFACT_DIR / "clustered_output.csv", index=False)

    LOGGER.info("Best k selected: %s", best_k)
    return best_k, results_df


if __name__ == "__main__":
    train_and_save_model()
