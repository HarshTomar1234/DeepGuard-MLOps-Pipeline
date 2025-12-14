# DeepGuard MLOps Pipeline Architecture

> **Overview**: This document describes the architecture of the DeepGuard MLOps pipeline, a production-grade machine learning system for AI-generated image detection.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DeepGuard MLOps Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Kaggle    │───▶│    DVC      │───▶│   MLflow    │───▶│   Flask     │  │
│  │   Dataset   │    │   Pipeline  │    │   Tracking  │    │   Web App   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Raw Data   │    │  Processed  │    │   DagsHub   │    │   User      │  │
│  │  Storage    │    │    Data     │    │   Registry  │    │  Interface  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
DeepGuard-MLOps-Pipeline/
├── data/                          # Data directory (DVC tracked)
│   ├── raw/                       # Original dataset from Kaggle
│   │   ├── train/
│   │   │   ├── REAL/             # Real images
│   │   │   └── FAKE/             # AI-generated images
│   │   └── test/
│   ├── processed/                 # Preprocessed numpy arrays
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   ├── X_test.npy
│   │   └── y_test.npy
│   └── features/                  # Augmented features
│
├── src/                           # Source code modules
│   ├── data/                      # Data processing modules
│   │   ├── data_ingestion.py     # Dataset download & organization
│   │   └── data_preprocessing.py # Image preprocessing & splits
│   ├── features/                  # Feature engineering
│   │   └── feature_engineering.py
│   ├── model/                     # Model-related modules
│   │   ├── model_building.py     # CNN architectures & training
│   │   ├── model_evaluation.py   # Metrics & visualization
│   │   └── register_model.py     # MLflow model registration
│   └── logger/                    # Logging utilities
│
├── models/                        # Trained models (DVC tracked)
│   └── best_model.keras          # Best performing model
│
├── reports/                       # Evaluation outputs
│   ├── metrics.json              # Performance metrics
│   ├── figures/                  # Confusion matrix, ROC, etc.
│   └── experiment_info.json      # MLflow run details
│
├── flask_app/                     # Web application
│   ├── app.py                    # Flask server
│   ├── templates/                # HTML templates
│   └── static/                   # CSS, JS assets
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── docs/                          # Documentation
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC pipeline state
├── params.yaml                    # Hyperparameters & config
└── requirements.txt               # Python dependencies
```

---

## 🔄 DVC Pipeline Stages

The pipeline consists of **6 stages** that execute sequentially:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Data        │────▶│  2. Data        │────▶│  3. Feature     │
│  Ingestion      │     │  Preprocessing  │     │  Engineering    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  6. Model       │◀────│  5. Model       │◀────│  4. Model       │
│  Registration   │     │  Evaluation     │     │  Building       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Stage Details

| Stage | Command | Description | Outputs |
|-------|---------|-------------|---------|
| **1. Data Ingestion** | `python -m src.data.data_ingestion` | Downloads dataset from Kaggle | `data/raw/` |
| **2. Preprocessing** | `python -m src.data.data_preprocessing` | Resizes, normalizes, splits data | `data/processed/` |
| **3. Feature Engineering** | `python -m src.features.feature_engineering` | Applies augmentation | `data/features/` |
| **4. Model Building** | `python -m src.model.model_building` | Trains CNN model | `models/` |
| **5. Evaluation** | `python -m src.model.model_evaluation` | Generates metrics & plots | `reports/` |
| **6. Registration** | `python -m src.model.register_model` | Registers model to MLflow | MLflow Registry |

---

## 🧠 Model Architectures

The pipeline supports multiple CNN architectures configured in `params.yaml`:

### Available Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| `SimpleCNN` | Basic 3-layer CNN | Quick testing |
| `DeeperCNN` | 5-layer CNN with dropout | Better accuracy |
| `EfficientStyleCNN` | Custom efficient architecture | Balanced performance |
| `EfficientNetB0Transfer` | Transfer learning from EfficientNet | Strong baseline |
| `XceptionTransfer` | Transfer learning from Xception | **Recommended** |

### Current Configuration (XceptionTransfer)

```yaml
model:
  architecture: "XceptionTransfer"
  input_shape: [128, 128, 3]

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0001
  early_stopping_patience: 7
```

---

## 📊 MLflow Integration

### Experiment Tracking

Every training run logs:
- **Parameters**: Learning rate, batch size, architecture
- **Metrics**: Accuracy, loss, precision, recall, F1-score, AUC
- **Artifacts**: Model weights, confusion matrix, ROC curve

### Model Registry

After evaluation, models are registered to MLflow with:
- Version control
- Stage transitions (Staging → Production)
- Metadata and lineage tracking

---

## 🌐 Flask Web Application

The Flask app provides a user interface for image classification:

```
User uploads image
        │
        ▼
┌─────────────────┐
│  Flask Server   │
│  (app.py)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load Model     │
│  (best_model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocess &   │
│  Predict        │
└────────┬────────┘
         │
         ▼
   Return Result
   (Real / Fake)
```

---

## 🔧 Configuration Files

### params.yaml

Central configuration for all pipeline parameters:
- Data paths and dataset name
- Preprocessing settings (image size, splits)
- Augmentation parameters
- Model architecture choice
- Training hyperparameters
- MLflow/DagsHub credentials

### dvc.yaml

Defines the pipeline DAG with:
- Stage commands
- Dependencies
- Parameters
- Outputs and metrics

---

## 📈 Metrics & Monitoring

### Training Metrics
- Training/Validation Loss
- Training/Validation Accuracy
- Training Time per Epoch

### Evaluation Metrics
- Test Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

### MLflow Dashboard
- Experiment comparison
- Metric trends across runs
- Model version comparison

---

## 🔐 Environment Variables

Required environment variables (stored in `.env`):

| Variable | Description |
|----------|-------------|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `MLFLOW_TRACKING_URI` | MLflow server URI |
| `KAGGLE_USERNAME` | Kaggle API username |
| `KAGGLE_KEY` | Kaggle API key |

---

## 📚 Related Documentation

- [Setup Guide](SETUP.md) - DagsHub/MLflow configuration
- [Quickstart](QUICKSTART.md) - Running the pipeline from scratch
