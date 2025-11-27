# ADHD200 Neuroimaging Pipeline

This repository contains a full end‑to‑end machine learning pipeline for the **ADHD‑200 dataset**, covering preprocessing, feature extraction, classical and deep learning models, and evaluation. The project is designed for reproducibility, modularity, and eventual deployment via Docker + FastAPI.

---

## 📂 Project Overview

### Inputs
- **NIfTI MRI scans** (structural brain images)
- **Phenotypic CSVs** (age, gender, handedness, IQ, diagnosis)

### Outputs
- **Predictions**: Control vs ADHD subtypes (0–3)
- **Metrics**: Accuracy, ROC‑AUC, confusion matrices
- **Visualizations**: ROI bar charts, brain overlays, ROC curves

---

## 🧩 Pipeline Phases

### Phase 1: ADHD‑200 Data Organization
- Scan anatomical dataset structure
- Load phenotypic metadata
- Match subjects to labels
- Save subject index (`phase1/subject_index.csv`)

### Phase 2: Visualization & Splits
- Visualize brain scans (axial, coronal, sagittal)
- Overlay ROI atlases (AAL, Harvard‑Oxford)
- Create stratified train/val/test splits
- Balance dataset with oversampling

### Phase 3A: Feature Extraction (MRI)
- Histogram features
- Texture features (GLCM)
- Deep features (ResNet18 pretrained on ImageNet)

### Phase 3B: Radiomics Feature Extraction
- Radiomics features via **PyRadiomics**
- ROI masks and radiomic descriptors
- Saved as `.csv` and `.pkl` artifacts

### Phase 4A: Classical Models (Phenotypic + Phase 3A)
- Random Forest, SVM
- Train/validation/test reports

### Phase 4B: Classical Models (Radiomics Features)
- Same classifiers trained on radiomics features
- Comparative evaluation

### Phase 4C: Deep Learning Models
- CNNs , LSTM and ViTs trained directly on raw NIfTI scans
- Checkpoints saved (`.pth`)

### Phase 4D: Model Unification
- Ensemble of Phase 4A, 4B, and 4C
- Improved performance through fusion (Incomplete due to kaggle resources)

### Phase 5: Evaluation & Analysis
- ROI statistical analysis with covariates
- Save results (`rej.npy`, `p_adj.npy`, `betas.npy`)
- Visualization: bar charts, brain overlays
- Logistic regression baseline (saved as `model_v1.pkl`)
- Graph Neural Network (PyTorch Geometric)
- Multiclass ROC curves and confusion matrices

---

## 🚀 Deployment (Phase 6)

The pipeline is prepared for containerization and serving:

- **Inference model**: Logistic regression baseline (`model_v1.pkl`)
- **FastAPI REST API**:
  - `GET /health` → service status
  - `POST /predict` → accepts NIfTI file, returns prediction
- **Dockerfile**: reproducible environment with dependencies (`nibabel`, `torch`, `scikit‑learn`, `fastapi`, `uvicorn`)
- **Monitoring**:
  - Prometheus metrics endpoint
  - TensorBoard / Weights & Biases for training dashboards

---

## 📦 Project Structure

DeepLearning/ 
│── ADHD200-Pipeline/ 
│ ├── app/ 
│ │ ├── inference.py # model loading + prediction 
│ │ ├── main.py # FastAPI endpoints 
│ ├── models/ 
│ │ ├── model_v1.pkl # saved logistic regression model 
│ ├── requirements.txt # dependencies 
│ ├── Dockerfile # container build instructions 
│ ── README.md # this file

