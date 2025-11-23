
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import joblib
import argparse
from pathlib import Path

def load_and_prepare(csv_path: str, test_size=0.2, random_state=42, n_components=4):
    df = pd.read_csv(csv_path)
    # Expect a 'Class' column: 1 = fraud, 0 = legit
    assert 'Class' in df.columns, "Dataset must contain a 'Class' target column."
    y = df['Class'].values.astype(int)
    X = df.drop(columns=['Class']).values.astype(float)

    # Optional: shuffle to avoid any ordering effects
    X, y = shuffle(X, y, random_state=random_state)

    # Train/test split (stratified due to class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reduce dimensionality for quantum circuit
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "X_train_pca": X_train_pca, "X_test_pca": X_test_pca,
        "scaler": scaler, "pca": pca
    }

def save_artifacts(out_dir: str, scaler, pca):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    joblib.dump(pca, f"{out_dir}/pca.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to creditcard.csv dataset")
    parser.add_argument("--out", default="./artifacts", help="Where to save scaler/pca")
    parser.add_argument("--n_components", type=int, default=4)
    args = parser.parse_args()

    bundle = load_and_prepare(args.csv, n_components=args.n_components)
    save_artifacts(args.out, bundle["scaler"], bundle["pca"])
    print("Saved scaler and PCA to", args.out)

if __name__ == "__main__":
    main()
