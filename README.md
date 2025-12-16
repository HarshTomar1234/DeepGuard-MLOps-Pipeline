# DeepGuard - MLOps Pipeline for Deepfake Detection

DeepGuard is an end-to-end MLOps project designed to detect AI-generated (deepfake) human faces with high precision. This repository demonstrates a complete production-grade machine learning lifecycle, from data ingestion and versioning to model deployment and monitoring.

![Application Overview](images/application_overview.png)

## Project Overview

With the rise of Generative Adversarial Networks (GANs) and Diffusion models, distinguishing between real and synthetic media has become a critical challenge. DeepGuard leverages a Deep Convolutional Neural Network (CNN) trained on the **GenImage** dataset to classify images as either "REAL" or "FAKE" (AI-Generated).

The project is built with a modular MLOps architecture, ensuring reproducibility, scalability, and ease of experimentation.

### Key Features

*   **Robust Detection Model**: Trained on over 140,000 images (Real vs. Fake) achieving >95% accuracy.
*   **End-to-End MLOps Pipeline**: Fully automated pipeline using DVC for data versioning and pipelines.
*   **Experiment Tracking**: Integrated with MLflow and DagsHub to track model parameters, metrics, and artifacts.
*   **Dual Deployment Strategy**:
    *   **Flask API**: Lightweight REST API for local or cloud container deployment.
    *   **Hugging Face Space**: Interactive web interface powered by Gradio for public demonstration.
*   **Frequency Analysis**: Includes FFT (Fast Fourier Transform) visualization to analyze frequency domain artifacts often present in GAN-generated images.

## Model Performance

The model uses a Transfer Learning approach (Xception/VGG based architecture) fine-tuned for artifact detection.

*   **Training Accuracy**: ~99%
*   **Validation Accuracy**: ~95%
*   **Dataset**: 140k images (Tiny GenImage subset) comprising outputs from Stable Diffusion, Midjourney, and other SOTA generators.

## Application Interface

The application provides a clean, user-friendly interface for real-time analysis.

### Main Interface
![DeepGuard Interface](images/app_interface.png)

### Detection Results
The model provides a confidence score and a label for every prediction.

**Fake Image Detection**
![Fake Detection Example](images/fake_image_detection.png)

**Real Image Detection**
![Real Detection Example](images/real_face_detection.png)

## Technical Architecture

This project is structured as a modular pipeline controlled by `dvc.yaml`:

1.  **Data Ingestion**: downloads and verifies the raw dataset.
2.  **Preprocessing**: Resizes, normalizes, and augments image data.
3.  **Feature Engineering**: Prepares data generators and extracting specific features (if applicable).
4.  **Model Training**: Trains the TensorFlow/Keras model with early stopping and checkpointing.
5.  **Evaluation**: Validates performance on a hold-out test set and logs metrics to MLflow.
6.  **Registration**: Versions the final model artifact.

## Technology Stack

*   **Language**: Python 3.10+
*   **Deep Learning**: TensorFlow, Keras
*   **Data Versioning**: DVC (Data Version Control)
*   **Experiment Tracking**: MLflow, DagsHub
*   **Web Frameworks**: Flask (API), Gradio (UI)
*   **Deployment**: Docker, Hugging Face Spaces

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/HarshTomar1234/MLOps-Text-Classification-Pipeline.git
cd MLOps-Text-Classification-Pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask App
```bash
python flask_app/app.py
```
Access the application at `http://localhost:5000`

### 4. Reproduce the Pipeline (DVC)
To retrain or reproduce the entire pipeline from scratch:
```bash
dvc repro
```

## Live Demo

Check out the live demonstration on Hugging Face Spaces:
[DeepGuard on Hugging Face](https://huggingface.co/spaces/Coddieharsh/DeepGuard)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
