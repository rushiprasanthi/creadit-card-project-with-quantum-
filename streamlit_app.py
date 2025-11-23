
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Quantum Fraud Detector", layout="centered")

st.title("🧮 Quantum-Enhanced Fraud Detection (Demo)")

st.write("""
This app compares a **classical baseline** with a **quantum-inspired** model.
- Upload a CSV with the same feature columns used during training (except 'Class').
- Or type values manually (after PCA/scaler, the model expects the trained pipeline).
""")

art_dir = Path("./artifacts")
if not art_dir.exists():
    st.warning("Artifacts not found. Please run training scripts first to generate ./artifacts.")
else:
    scaler = joblib.load(art_dir / "scaler.pkl")
    pca = joblib.load(art_dir / "pca.pkl")
    baseline = joblib.load(art_dir / "baseline_logreg.pkl")

    # Lazy import quantum only when needed
    from train_quantum import build_vqc, predict_proba

    qcfg = joblib.load(art_dir / "quantum_config.pkl")
    weights = np.load(art_dir / "quantum_weights.npy", allow_pickle=True)
    circuit, _ = build_vqc(n_qubits=qcfg["qubits"], n_layers=qcfg["layers"])

    st.subheader("🔽 Try a Single Transaction")
    with st.form("single_txn"):
        amount = st.number_input("Amount", min_value=0.0, value=50.0, step=1.0)
        time = st.number_input("Time (seconds)", min_value=0.0, value=10000.0, step=1.0)
        # Add more numeric inputs as needed; for demo we keep two and pad zeros
        submitted = st.form_submit_button("Predict")

    def preprocess_row(row_np):
        # row_np is shape (n_features,), we will pad/trim to training features length
        row_np = row_np.reshape(1, -1)
        X_scaled = scaler.transform(row_np)
        X_pca = pca.transform(X_scaled)
        return X_scaled, X_pca

    if submitted:
        # Minimal feature vector example: [Time, Amount] + zeros to match scaler
        # In real use, match exactly the training feature columns order!
        # Here we assume scaler was fit on N features; we create an N-dim vector with two filled.
        n_features = scaler.mean_.shape[0]
        vec = np.zeros(n_features, dtype=float)
        # Heuristic: set first two positions
        vec[0] = time
        vec[1] = amount
        Xs, Xp = preprocess_row(vec)

        # Classical
        pred_c = baseline.predict(Xs)[0]
        prob_c = baseline.predict_proba(Xs)[0,1]

        # Quantum
        prob_q = predict_proba(circuit, weights, Xp)[0]
        pred_q = int(prob_q >= 0.5)

        st.write(f"**Classical** → Prob(Fraud)={prob_c:.3f}  •  Pred={pred_c}")
        st.write(f"**Quantum** → Prob(Fraud)={prob_q:.3f}  •  Pred={pred_q}")

    st.subheader("📤 Batch Upload (CSV)")
    uploaded = st.file_uploader("Upload CSV with feature columns (no 'Class' column required)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        X_scaled = scaler.transform(df.values.astype(float))
        X_pca = pca.transform(X_scaled)

        # Predictions
        prob_c = baseline.predict_proba(X_scaled)[:,1]
        pred_c = (prob_c >= 0.5).astype(int)

        prob_q = predict_proba(circuit, weights, X_pca)
        pred_q = (prob_q >= 0.5).astype(int)

        out = df.copy()
        out["prob_classical"] = prob_c
        out["pred_classical"] = pred_c
        out["prob_quantum"] = prob_q
        out["pred_quantum"] = pred_q

        st.write("Preview of predictions:")
        st.dataframe(out.head(20))
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
