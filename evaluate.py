
import argparse
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import load_and_prepare
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--artifacts", default="./artifacts")
    args = parser.parse_args()

    data = load_and_prepare(args.csv, n_components=4)
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]
    X_test_pca = data["X_test_pca"]

    # Load classical model
    baseline = joblib.load(f"{args.artifacts}/baseline_logreg.pkl")

    # Load quantum weights/config (lazy import to avoid heavy deps if not needed)
    import pennylane as qml
    from train_quantum import build_vqc, predict_proba

    qcfg = joblib.load(f"{args.artifacts}/quantum_config.pkl")
    weights = np.load(f"{args.artifacts}/quantum_weights.npy", allow_pickle=True)

    # Evaluate classical
    y_pred_classical = baseline.predict(X_test_scaled)
    print("=== Classical (Logistic Regression) ===")
    print(classification_report(y_test, y_pred_classical, digits=4))
    print(confusion_matrix(y_test, y_pred_classical))

    # Evaluate quantum
    circuit, _ = build_vqc(n_qubits=qcfg["qubits"], n_layers=qcfg["layers"])
    probs = predict_proba(circuit, weights, X_test_pca)
    y_pred_quantum = (probs >= 0.5).astype(int)
    print("\n=== Quantum (VQC) ===")
    print(classification_report(y_test, y_pred_quantum, digits=4))
    print(confusion_matrix(y_test, y_pred_quantum))

    # Simple matplotlib chart: probability distribution for quantum model
    plt.figure()
    plt.hist(probs[y_test==0], bins=50, alpha=0.7, label="Legit (y=0)")
    plt.hist(probs[y_test==1], bins=50, alpha=0.7, label="Fraud (y=1)")
    plt.xlabel("Quantum predicted probability of fraud")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Quantum Model Probability Distributions")
    plt.tight_layout()
    plt.savefig("quantum_prob_hist.png")
    print("Saved chart: quantum_prob_hist.png")

if __name__ == "__main__":
    main()
