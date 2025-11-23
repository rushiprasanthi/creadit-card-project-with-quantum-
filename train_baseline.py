
import argparse
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_prep import load_and_prepare, save_artifacts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="./artifacts")
    args = parser.parse_args()

    data = load_and_prepare(args.csv, n_components=4)
    X_train_scaled = data["X_train_scaled"]
    y_train = data["y_train"]
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]

    # Simple and fast baseline
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train_scaled, y_train)

    # Save scaler/pca + model
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_artifacts(args.out, data["scaler"], data["pca"])
    joblib.dump(clf, f"{args.out}/baseline_logreg.pkl")

    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    print(f"Baseline LogisticRegression — Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

if __name__ == "__main__":
    main()
