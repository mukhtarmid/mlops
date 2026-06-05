# 🚀 MLOps Zero to Hero

A hands-on MLOps learning repository built while following the **MLOps Zero to Hero** course by Abhishek Veeramalla.

This repository documents my journey from training a simple machine learning model in a notebook to building a complete production-grade MLOps system with:

* Git & GitHub
* CI/CD Pipelines
* Docker
* Kubernetes
* MLflow
* FastAPI
* Monitoring & Observability
* AWS Deployment
* End-to-End MLOps Workflows

---

# 🎯 Goal

Most machine learning tutorials stop after achieving a good model accuracy.

Real-world machine learning systems require much more:

```text
Data Collection
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Registry
      ↓
Deployment
      ↓
Monitoring
      ↓
Retraining
```

This repository is focused on learning and implementing each stage of that lifecycle.

---

# 📚 Course Reference

This project follows concepts from:

* MLOps Zero to Hero by Abhishek Veeramalla
* Practical MLOps implementations
* Production-grade deployment practices
* Real-world ML system design

---

# 🏗️ Current Project Structure

```text
mlops-zero-to-hero/
│
├── data/
│
├── models/
│   └── model.pkl
│
├── notebooks/
│
├── src/
│   └── train.py
│
├── pyproject.toml
├── README.md
```

---

# 🧠 Current Project

The first project trains a Random Forest classifier using the Iris dataset.

Workflow:

```text
Iris Dataset
      ↓
Random Forest Training
      ↓
Trained Model
      ↓
model.pkl
```

The trained model is saved as a reusable artifact:

```text
models/model.pkl
```

---

# ⚙️ Setup

## Clone Repository

```bash
git clone <repository-url>
cd mlops-zero-to-hero
```

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -e .
```

---

# 🚀 Train Model

Run:

```bash
python src/train.py
```

or

```bash
train-model
```

Expected output:

```text
Model saved
```

---

# 📦 Dependencies

Current dependencies:

* scikit-learn
* pandas
* joblib

Managed through:

```text
pyproject.toml
```

---

# ❓ Why This Is Not MLOps Yet

Current workflow:

```text
Run script manually
Save model manually
Deploy manually
Monitor manually
```

This is often called:

```text
NotebookOps
```

Real MLOps looks like:

```text
Git Push
    ↓
CI Pipeline
    ↓
Train Model
    ↓
Validate Model
    ↓
Package Model
    ↓
Deploy
    ↓
Monitor
    ↓
Retrain
```

Everything is automated, reproducible, and observable.

---

# 🛣️ Learning Roadmap

Planned additions:

* [ ] Git Best Practices
* [ ] Project Packaging
* [ ] Dockerization
* [ ] FastAPI Model Serving
* [ ] GitHub Actions CI/CD
* [ ] MLflow Tracking
* [ ] Model Registry
* [ ] Kubernetes Deployment
* [ ] Monitoring with Prometheus
* [ ] Dashboards with Grafana
* [ ] AWS Deployment
* [ ] Automated Retraining Pipelines

---

# 📖 Key Takeaway

> Training a model is only the beginning. MLOps is the discipline of making machine learning systems reliable, scalable, reproducible, and production-ready.

---

# 🤝 Acknowledgements

Special thanks to Abhishek Veeramalla and the MLOps Zero to Hero community for providing practical, hands-on learning resources that focus on real-world engineering challenges.
