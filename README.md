
Project: Quantum-Enhanced Credit Card Fraud Detection 
---------------------------------------------------------------------------------

A reproducible experimental system demonstrating a hybrid classical + quantum-inspired pipeline for credit card fraud detection. The repository contains:

- a classical baseline model (Logistic Regression),
- a quantum-inspired Variational Quantum Circuit (VQC) implemented using PennyLane,
- data preparation, evaluation, and a Streamlit demo for ad-hoc and batch predictions.

This README documents architecture, internals, operational capabilities and productionization guidance so the project is recruiter- and production-ready in presentation.

Table of contents
-----------------
- Project overview
- Core features & intelligent workflows
- System architecture (detailed)
- Technical stack and rationale
- Folder structure
- Execution / pipeline workflow (step-by-step)
- Module & API documentation (internal)
- Installation & environment setup
- Running the project (commands & examples)
- Evaluation & expected outputs
- Scalability, performance & productionization considerations
- Engineering tradeoffs & challenges
- Roadmap & future improvements
- Contributing
- License
- Author

1 — Project overview
--------------------

What this project does
- Trains and evaluates two models to detect credit-card fraud:
  - Classical baseline: Logistic Regression on scaled features.
  - Quantum-inspired model: A Variational Quantum Circuit (VQC) VQC implemented with PennyLane and a default.qubit simulator.
- Provides an interactive Streamlit demo for single-transaction and batch CSV predictions.
- Produces and saves ML artifacts (scaler, PCA, logistic model, PCA-based quantum weights & config) in ./artifacts for reproducible inference.

Real-world problem solved
- Detecting fraudulent credit card transactions in an imbalanced dataset where precision/recall and interpretability matter.
- Demonstrates research-oriented hybrid classical/quantum workflows and how such systems can be integrated into standard ML pipelines.

Why it matters
- Shows how dimensionality reduction (PCA) and classical preprocessing can be used to make quantum circuit-based classifiers tractable.
- Provides a blueprint for hybrid systems where classical preprocessing + quantum models are packaged, evaluated and served.
- Useful for proof-of-concept and portfolio demonstrations for roles in ML engineering, hybrid quantum computing, and MLOps.

Key engineering goals
- Reproducible ML artifacts and deterministic pipelines (scaler/pca saved with joblib).
- Keep quantum computation tractable by limiting qubit count through PCA.
- Clear separation of preprocessing, baseline training, quantum training, evaluation and serving/demo.

2 — Core features & intelligent workflows
-----------------------------------------

Detailed feature breakdown
- Data preparation: load, shuffle, stratified train/test split, feature standardization (StandardScaler), PCA dimensionality reduction for quantum model.
- Classical baseline: fast, explainable LogisticRegression trained on scaled data and saved as baseline_logreg.pkl.
- Quantum-inspired model: VQC using angle-encoding, layered trainable rotations (RY, RZ) and ring entanglement (CNOTs) — trained with a gradient descent optimizer (PennyLane's GradientDescentOptimizer).
- Artifact management: scaler.pkl, pca.pkl, baseline_logreg.pkl, quantum_weights.npy, quantum_config.pkl — all saved in ./artifacts.
- Evaluation: classification_report and confusion_matrix for both models, plus a saved histogram (quantum_prob_hist.png).
- Demo: Streamlit app with lazy quantum module loading, single-transaction form and CSV batch prediction with downloadable predictions.

Automation & orchestration
- Scripts are designed as CLI entrypoints so they can be wired into automated pipelines:
  - data_prep.py to produce artifacts
  - train_baseline.py and train_quantum.py to produce model artifacts
  - evaluate.py to generate metrics and diagnostic plots
  - streamlit_app.py to serve the demo
- Artifacts enable separation of training and serving phases: training pipeline produces artifacts consumed by the demo.

Operational capabilities & engineering highlights
- Lightweight baseline for fast iteration.
- Quantum training uses a simulator (default.qubit) for deterministic expectation values (shots=None by default).
- The quantum pipeline downsamples training to at most 2000 examples for training speed.
- Streamlit app performs lazy quantum imports to reduce startup cost where quantum evaluation is not needed.

3 — System architecture (deep)
-------------------------------

High-level architecture 
--------------------------------
Data CSV (creditcard.csv)
        |
        v
  data_prep.py  —> outputs: scaler.pkl, pca.pkl
        |
        +----------------------------+
        |                            |
        v                            v
train_baseline.py               train_quantum.py
(LogisticRegression)            (PennyLane VQC)
        |                            |
        |                            +-> PCA(n_components = qubits) -> VQC -> weights -> quantum_weights.npy, quantum_config.pkl
        |                            |
baseline_logreg.pkl               saved artifacts
        |
        v
evaluate.py  -> loads baseline + quantum artifacts, prints metrics and saves quantum_prob_hist.png
        |
        v
streamlit_app.py  -> uses artifacts for interactive inference (single & batch), lazy loads quantum code

Backend flow and request lifecycle (serving inference)
- Streamlit (demo) is the serving component for interactive use.
- On start, streamlit_app.py checks ./artifacts:
  - Loads scaler.pkl, pca.pkl, baseline_logreg.pkl immediately.
  - Only when quantum inference is used it lazy-imports train_quantum.build_vqc and predict_proba, and loads quantum_weights.npy & quantum_config.pkl.
- Single-transaction flow:
  - User provides a minimal input (Time, Amount) -> app pads to expected length -> scaler.transform -> pca.transform -> classical baseline predict/predict_proba on scaled features and quantum predict_proba on PCA features -> display and optional CSV download.
- Batch flow:
  - CSV uploaded -> df values -> scaler.transform -> pca.transform -> vectorized baseline and quantum predictions -> appended columns and downloadable file.

Data & processing pipeline
- Ingest CSV with a 'Class' column (1 = fraud, 0 = legitimate).
- Shuffle + stratified train/test split to respect class imbalance.
- Fit StandardScaler on training set; persist scaler.
- Fit PCA on scaled training set: PCA n_components is tuned to n_qubits for the VQC model.
- Baseline trains on scaled features; quantum trains on PCA-transformed features.
- Artifacts saved together to decouple training from serving.

Variational Quantum Circuit (VQC)
- Angle encoding: RX rotations using PCA features as angles.
- Variational block: per-qubit RY and RZ parameters, then ring entanglement through CNOTs.
- Measurement: expectation of PauliZ on qubit-0 mapped from [-1,1] to [0,1] probability space via (exp + 1) / 2.
- Optimization: PennyLane GradientDescentOptimizer with step size configurable via train_quantum.py (lr parameter inside train_vqc currently set to 0.1).
- Weight initialization: small Gaussian noise (0.01 * normal).

4 — Technical stack and why each was chosen
-------------------------------------------

Languages & runtimes
- Python 3.8+ — broad ecosystem compatibility for ML and quantum libraries.

Core libraries
- pandas, numpy — data handling and numeric computation.
- scikit-learn — StandardScaler, PCA, LogisticRegression, train/test split and evaluation metrics. Chosen for reliability and consistency for baseline models and preprocessing.
- PennyLane — quantum circuit construction, qnode abstraction and hybrid training. Chosen for multi-backend support and easy integration with classical numeric workflows.
- qiskit + pennylane-qiskit — provided in requirements to allow the PennyLane Qiskit plugin for backend access if switching to Qiskit backends.
- matplotlib — lightweight plotting for evaluation artifacts.
- streamlit — rapid-proof-of-concept interactive demo and local serving.
- joblib — efficient persistence of sklearn-like objects, used for scaler, pca and baseline model artifacts.

Why these technologies
- The stack prioritizes reproducibility and developer experience. scikit-learn for established preprocessing and baseline modeling, and PennyLane to express VQCs and run them deterministically on statevector simulators before attempting hardware execution.

5 — Folder structure (professional tree)
-----------------------------------------
Root (project)
- artifacts/                 # generated at runtime: scaler.pkl, pca.pkl, baseline_logreg.pkl, quantum_weights.npy, quantum_config.pkl
- data_prep.py               # ingestion, scaling, PCA, artifact saving
- train_baseline.py          # trains and persists Logistic Regression baseline
- train_quantum.py           # builds, trains VQC and persists weights/config
- evaluate.py                # loads artifacts and prints metrics + plots
- streamlit_app.py           # interactive demo (lazy quantum imports)
- requirements.txt
- README.md
- baseline_logreg.pkl        # (optional if already committed)
- scaler.pkl                 # (optional if already committed)
- pca.pkl                    # (optional if already committed)
- quantum_weights.npy        # (optional if already committed)
- quantum_config.pkl         # (optional if already committed)

File purpose summary
- data_prep.py: central entry for data preparation; produces scaler/pca that must be used consistently across training & serving.
- train_baseline.py: trains an explainable baseline and persists it.
- train_quantum.py: contains quantum circuit building (angle_embed, variational_block, build_vqc), training loop (train_vqc) and artifact persistence.
- evaluate.py: compares model outputs and generates a probability distribution histogram for quantum model predictions.
- streamlit_app.py: demo UI, demonstrates lazy-loading patterns to avoid heavy quantum dependencies for users focused only on classical baseline.

6 — Workflow / execution pipeline (step-by-step)
------------------------------------------------

1. Environment setup
   - Create isolated Python environment and install requirements (see Installation).

2. Preprocessing & artifact generation
   - data_prep.py loads dataset (expects a 'Class' column), stratifies train/test splits, fits StandardScaler and PCA and saves them.
   - CLI: python data_prep.py --csv ./creditcard.csv --out ./artifacts

3. Train classical baseline (fast)
   - Uses StandardScaler output from data_prep and trains LogisticRegression on scaled features.
   - CLI: python train_baseline.py --csv ./creditcard.csv --out ./artifacts

4. Train quantum model (VQC)
   - Uses PCA-reduced features (n_components == n_qubits) to fit a VQC using PennyLane on default.qubit simulator.
   - Training downsamples to at most 2000 samples for speed in this example; tune for full dataset evaluation.
   - CLI: python train_quantum.py --csv ./creditcard.csv --out ./artifacts --epochs 25 --layers 2 --qubits 4

5. Evaluation
   - evaluate.py loads baseline and quantum artifacts, computes classification metrics and saves a histogram.
   - CLI: python evaluate.py --csv ./creditcard.csv --artifacts ./artifacts

6. Serving / Demo
   - streamlit_app.py loads artifacts and allows single/batch predictions.
   - CLI: streamlit run streamlit_app.py

7 — API / modules documentation (internal)
------------------------------------------

data_prep.py
- load_and_prepare(csv_path: str, test_size=0.2, random_state=42, n_components=4):
  - Returns dictionary with raw and transformed train/test arrays, scaler and pca objects.
  - Key design: PCA dimensionality reduction (n_components) is intended for VQC size control.
- save_artifacts(out_dir: str, scaler, pca): saves scaler and pca via joblib.

train_baseline.py
- Loads artifacts via load_and_prepare(..., n_components=4).
- Trains LogisticRegression(max_iter=1000). Saves baseline_logreg.pkl and artifacts (scaler/pca).

train_quantum.py
- make_device(n_qubits=4, shots=None): returns qml.device("default.qubit", wires=n_qubits, shots=shots).
- angle_embed(x): RX-based angle encoding for each qubit.
- variational_block(params): per-qubit RY and RZ parameters arranged as 2 * n_qubits vector per layer, then ring entanglement via CNOTs.
- build_vqc(n_qubits, n_layers): returns a qnode and weight_shape used to initialize weights.
- predict_proba(circuit, weights, X): runs circuit sequentially for each sample and maps expectation to probability: (exp + 1)/2.
- train_vqc(X_train, y_train, n_qubits, n_layers, epochs, lr, seed): runs GradientDescentOptimizer, prints progress every 5 epochs and returns trained circuit & weights.

evaluate.py
- Loads baseline and quantum artifacts (lazy import of PennyLane).
- Computes classification_report and confusion_matrix for both models.
- Saves a histogram of quantum predicted probabilities to quantum_prob_hist.png.

streamlit_app.py
- Loads scaler/pca/baseline and lazily imports build_vqc and predict_proba only when artifacts exist and quantum inference happens.
- Offers both single transaction form (pads feature vector to scaler dimensionality) and CSV batch upload.
- Output columns added: prob_classical, pred_classical, prob_quantum, pred_quantum.

Request/Response flows for production API (recommendation)
- Design a small REST interface (FastAPI suggested) with endpoints:
  - POST /predict/single -> accepts JSON features -> returns both classical & quantum predictions and probabilities.
  - POST /predict/batch -> accepts CSV or JSON array -> returns predictions as downloadable CSV or JSON list.
  - GET /health -> readiness/liveness check.
  - GET /metrics -> Prometheus metrics for requests, latencies, and inference counts.

8 — Installation guide (professional)
-------------------------------------

Prerequisites
- Python 3.8 or later
- Recommended: 8+ CPU cores, 16+ GB RAM for local experimentation; quantum simulation can be CPU-intensive.
- Optional: Access to Qiskit backend or cloud quantum hardware for real-device runs.

Create virtual environment and install
```bash
# Create and activate venv (UNIX)
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Note: PennyLane + pennylane-qiskit + qiskit can have non-Python system dependencies; follow the Qiskit and PennyLane docs if you intend to use actual hardware or Qiskit Aer.

Configure dataset
- Download Credit Card Fraud dataset from Kaggle:
  - https://www.kaggle.com/mlg-ulb/creditcardfraud
- Place CSV in project root or path of your choice and pass its path to the scripts via --csv.

9 — Running the project (exact commands)
----------------------------------------

1) Generate preprocessing artifacts (optional — the training scripts will also call load_and_prepare internally)
```bash
python data_prep.py --csv ./creditcard.csv --out ./artifacts --n_components 4
```

2) Train classical baseline
```bash
python train_baseline.py --csv ./creditcard.csv --out ./artifacts
# Example printed output:
# Baseline LogisticRegression — Acc: 0.9992  Prec: 0.9000  Rec: 0.6700  F1: 0.7667
```

3) Train quantum VQC (adjust epochs/layers/qubits for experiments)
```bash
python train_quantum.py --csv ./creditcard.csv --out ./artifacts --epochs 25 --layers 2 --qubits 4
# Example printed output:
# Epoch 5/25 — loss: 0.6940
# ...
# Quantum VQC — Acc: 0.9987  Prec: 0.5000  Rec: 0.2000  F1: 0.2857
# Saved quantum weights and config to ./artifacts
```

4) Evaluate both models and generate a probability histogram
```bash
python evaluate.py --csv ./creditcard.csv --artifacts ./artifacts
# This prints classification reports and creates quantum_prob_hist.png
```

5) Run the Streamlit demo
```bash
streamlit run streamlit_app.py
# Open http://localhost:8501 to interact.
```

Batch prediction example (Streamlit upload)
- Upload CSV with the same non-Class feature columns used for training (do not include 'Class').
- The app returns a preview and a downloadable CSV with appended columns:
  - prob_classical, pred_classical, prob_quantum, pred_quantum

10 — Scalability & engineering considerations
---------------------------------------------

Performance optimizations
- Vectorized classical predictions via scikit-learn are already fast.
- Quantum inference is currently implemented by iterating sample-per-sample over circuit evaluations:
  - Bottleneck: per-sample qnode execution on CPU simulator.
  - Approaches to reduce latency:
    - Use vectorized QNodes (PennyLane supports vmap/batched QNodes in newer versions).
    - Run on hardware accelerators or specialized simulators (Qiskit Aer, GPU-based quantum simulators).
    - Use approximate surrogates: train a small classical model to approximate the VQC predictions for batch scoring.

Scalability design
- For productionization:
  - Replace Streamlit with a REST server (FastAPI) behind a WSGI/ASGI server (uvicorn/gunicorn).
  - Containerize the service (Docker), use an image per model version.
  - Use a model registry (MLflow) or artifact storage (S3) for artifacts and versioning.
  - Autoscale inference pods via Kubernetes Horizontal Pod Autoscaler.

Fault tolerance & reliability
- Persist artifacts in durable storage (S3) and load them at startup.
- Implement retries for long-running quantum jobs and timeouts for inference requests.
- Graceful degradation: serve classical baseline predictions if quantum service unavailable.

Queueing & asynchronous batch processing
- For large batches, push jobs to a queue (RabbitMQ, SQS) and process asynchronously with worker pools.
- Use batching to amortize quantum circuit overhead.

Caching & database optimizations
- Cache preprocessing outputs (scaler/pca transformed arrays) for repeated queries.
- Index and pre-compute features for high-frequency clients.

11 — Challenges & engineering decisions
---------------------------------------

Technical challenges solved
- Making a quantum classifier tractable: PCA dimensionality reduction ties the number of features to the number of qubits.
- Balancing simulation speed with training fidelity by downsampling training data to 2000 for VQC training while still using full test set for evaluation.
- Making the demo accessible by lazy-loading heavy quantum dependencies to minimize install/start friction.

Tradeoffs & design choices
- Simulator vs real hardware: default.qubit statevector simulator (shots=None) gives deterministic expectations but is not representative of noisy hardware — choice favors reproducibility and debugability.
- PCA for dimensionality reduction loses some information but is necessary to keep qubit count small.
- Downsampling speeds experimentation but may reduce quantum model generalization.

12 — Future improvements & roadmap
----------------------------------
Short-term (near-term)
- Add a production REST API (FastAPI) and separate serving image.
- Containerize with a Dockerfile and provide a docker-compose for local testing.
- Add CI checks (linting, unit tests) and a Python package layout for easier reuse.

Mid-term
- Integrate MLFlow or DVC for artifact/version tracking.
- Add class-imbalance handling: SMOTE, class-weighted loss or focal loss for VQC.
- Replace the simple gradient descent with more advanced optimizers (Adam) or PennyLane's advanced optimizers. Consider parameter-shift gradient acceleration.

Long-term / research
- Run VQC on real quantum hardware via pennylane-qiskit plugin and Qiskit backends; add noise-aware training strategies.
- Explore hybrid ensembles where the classical and quantum model are combined via stacking/blending.
- Implement asynchronous batched quantum inference using vectorized QNodes or parallel worker pools that use specialized simulators.

Observability & MLOps
- Add logging, structured traces and Prometheus metrics (inference latency, QPU usage, failure rates).
- Add integration tests for artifact reproducibility.

13 — Contribution
-----------------
Contributions are welcome. Suggested workflow:
- Fork this repository.
- Create a feature branch: git checkout -b feat/my-feature
- Add tests/documentation and update README where appropriate.
- Open a Pull Request with a clear description of changes and rationale.

Please use conventional commits and keep PRs focused.

14 — License
-------------
This repository currently does not include a license file. For open-source distribution and use, add a license (e.g., MIT). Example quick-add:
```bash
# Add MIT license (example)
curl -o LICENSE https://opensource.org/licenses/MIT
git add LICENSE && git commit -m "chore: add MIT license"
```

15 — Author
-----------
- Repository owner: rushiprasanthi (GitHub)
- Profile: https://github.com/rushiprasanthi

Acknowledgements & references
- Credit Card Fraud Detection dataset — Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
- PennyLane docs — https://pennylane.ai
- Qiskit docs — https://qiskit.org
- scikit-learn docs — https://scikit-learn.org
