
# AQVH918 – Quantum ML: Fraud Detection (Hackathon Skeleton)

This is a **ready-to-run** skeleton you can use during the hackathon.

## Project Structure
```
aqvh918_skeleton/
├── README.md
├── requirements.txt
├── data_prep.py
├── train_baseline.py
├── train_quantum.py
├── evaluate.py
└── streamlit_app.py
```
- **data_prep.py**: Loads dataset, splits, scales, and applies PCA.
- **train_baseline.py**: Trains a classical baseline (Logistic Regression).
- **train_quantum.py**: Trains a simple VQC with PennyLane (4-qubit, angle encoding).
- **evaluate.py**: Compares classical vs quantum on the test set and prints metrics.
- **streamlit_app.py**: Simple UI to try transactions and see predictions.
- **requirements.txt**: Python deps.

## Dataset
Use the **Credit Card Fraud Detection** dataset (CSV with columns including `Class`), or adapt the loader to your data.
- Download: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Place the CSV path in the CLI args or environment variable.

## Quickstart
```bash
# 1) Install deps (recommend a fresh venv)
pip install -r requirements.txt

# 2) Train classical baseline
python train_baseline.py --csv ./creditcard.csv

# 3) Train quantum model (uses small subset for speed)
python train_quantum.py --csv ./creditcard.csv

# 4) Evaluate & compare
python evaluate.py --csv ./creditcard.csv

# 5) Run the demo app
streamlit run streamlit_app.py
```

## Notes
- The quantum model here uses 4 PCA components to keep the circuit small and training fast on simulators.
- You can increase qubits/components later, but keep hackathon time in mind.
- For a fair comparison, always report **precision/recall/F1** in addition to accuracy due to class imbalance.
