
import argparse
from pathlib import Path
import numpy as np
import joblib
import pennylane as qml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_prep import load_and_prepare, save_artifacts

def make_device(n_qubits=4, shots=None):
    # Default statevector simulator (shots=None gives exact expectation values)
    return qml.device("default.qubit", wires=n_qubits, shots=shots)

def angle_embed(x):
    # x is a 1D vector length <= n_qubits; use RX for angle encoding
    for i, val in enumerate(x):
        qml.RX(val, wires=i)

def variational_block(params):
    """A simple layer of trainable rotations with ring entanglement."""
    n_qubits = len(params) // 2
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
        qml.RZ(params[i + n_qubits], wires=i)
    # Ring entanglement (CNOT i->i+1)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[n_qubits - 1, 0])

def build_vqc(n_qubits=4, n_layers=2):
    dev = make_device(n_qubits=n_qubits)

    # Total params per layer: 2 * n_qubits
    weight_shape = (n_layers, 2 * n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        angle_embed(x)
        for l in range(n_layers):
            variational_block(weights[l])
        # Measure the first qubit expectation Z
        return qml.expval(qml.PauliZ(0))

    return circuit, weight_shape

def predict_proba(circuit, weights, X):
    # Convert expectation in [-1,1] to probability in [0,1]
    exps = np.array([circuit(x, weights) for x in X])
    return (exps + 1.0) / 2.0

def train_vqc(X_train, y_train, n_qubits=4, n_layers=2, epochs=25, lr=0.1, seed=42):
    np.random.seed(seed)
    circuit, weight_shape = build_vqc(n_qubits=n_qubits, n_layers=n_layers)
    weights = 0.01 * np.random.randn(*weight_shape)

    opt = qml.GradientDescentOptimizer(stepsize=lr)

    def loss_fn(weights):
        probs = predict_proba(circuit, weights, X_train)
        # Binary cross-entropy (numerically stable clip)
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        y = y_train
        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    for epoch in range(epochs):
        weights = opt.step(loss_fn, weights)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} — loss: {loss_fn(weights):.4f}")
    return circuit, weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="./artifacts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--qubits", type=int, default=4)
    args = parser.parse_args()

    data = load_and_prepare(args.csv, n_components=args.qubits)
    X_train = data["X_train_pca"]
    y_train = data["y_train"]
    X_test = data["X_test_pca"]
    y_test = data["y_test"]

    # Optionally downsample for speed (class-balanced sample)
    # Here we take up to 2000 samples if available
    max_train = min(2000, len(X_train))
    X_train_small = X_train[:max_train]
    y_train_small = y_train[:max_train]

    circuit, weights = train_vqc(
        X_train_small, y_train_small, n_qubits=args.qubits, n_layers=args.layers, epochs=args.epochs
    )

    # Save artifacts
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_artifacts(args.out, data["scaler"], data["pca"])
    np.save(f"{args.out}/quantum_weights.npy", weights)
    joblib.dump({"layers": args.layers, "qubits": args.qubits}, f"{args.out}/quantum_config.pkl")

    # Quick test metrics
    probs = predict_proba(circuit, weights, X_test)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    print(f"Quantum VQC — Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
    print("Saved quantum weights and config to", args.out)

if __name__ == "__main__":
    main()
