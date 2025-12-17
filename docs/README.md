# DeepGuard MLOps Pipeline Documentation

Welcome to the DeepGuard documentation. This guide will help you understand, set up, and run the AI-generated image detection pipeline.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in 30 minutes |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and project structure |
| [SETUP.md](SETUP.md) | DagsHub and MLflow configuration |
| [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md) | Model capabilities and constraints |

---

## Quick Links

### First Time Setup

```powershell
git clone https://github.com/HarshTomar1234/DeepGuard-MLOps-Pipeline.git
cd DeepGuard-MLOps-Pipeline
pip install -r requirements.txt
dvc pull
```

### Run Web App

```powershell
python flask_app/app.py
# Open http://localhost:5000
```

### Run with Docker

```powershell
docker build -t deepguard-app:latest .
docker run -p 8888:5000 deepguard-app:latest
# Open http://localhost:8888
```

### View Pipeline

```powershell
dvc dag
```

---

## Project Overview

DeepGuard is a production-grade MLOps pipeline for detecting AI-generated images. It demonstrates:

- Data Versioning with DVC
- Experiment Tracking with MLflow/DagsHub
- Reproducible Pipelines with DVC stages
- Model Registry for version control
- Containerized Deployment with Docker
- CI/CD Automation with GitHub Actions
- Cloud Infrastructure with AWS (S3, ECR, EKS)

---

## Pipeline Stages

```
Data Ingestion --> Preprocessing --> Feature Engineering --> Model Building --> Evaluation --> Registration
```

Each stage is defined in `dvc.yaml` and tracked for reproducibility.

---

## Configuration

All parameters are centralized in `params.yaml`:
- Model architecture selection
- Training hyperparameters
- Data processing settings
- MLflow configuration

---

## Model Performance

Current model (XceptionTransfer) achieves:
- Training Accuracy: ~99%
- Validation Accuracy: ~95%
- Test Accuracy: ~88%
- ROC-AUC: ~0.98

Performance may vary based on dataset and training parameters.

---

## Deployment Options

| Option | Description |
|--------|-------------|
| Local Flask | Run `python flask_app/app.py` |
| Docker | Build and run container locally |
| Hugging Face | Interactive Gradio interface |
| AWS EKS | Kubernetes deployment with LoadBalancer |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Support

- Create an issue on GitHub
- Check the troubleshooting sections in each doc
- Review DagsHub experiment logs
