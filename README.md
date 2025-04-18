# Image Interpreter

<p align="center">
  <img src="logo.png" alt="Colon Watcher Logo" width="120" align="right">
</p>

A real-time image captioning system for generating descriptive captions from images. **Image Interpreter** leverages a deep learning model combining **ResNet50** for feature extraction and an **LSTM with attention** for caption generation. The system uses **ZenML** pipelines to manage data preprocessing, training, and deployment workflows, with a **FastAPI** service for inference and a **Docker** configuration for easy containerization and deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Model Training](#model-training)
5. [Pipeline (ZenML)](#pipeline-zenml)
6. [API (FastAPI)](#api-fastapi)
7. [Docker](#docker)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Setup & Usage](#setup--usage)
10. [License](#license)
11. [References](#references)
---

## Overview

**Image Interpreter** aims to generate accurate and descriptive captions for images in real-time, suitable for applications like automated image description, accessibility tools, and content analysis. Key features include:

- **Data Preprocessing**: Extracts image features using ResNet50 and processes captions for training.
- **Model Training**: Uses an LSTM-based model with attention for caption generation.
- **Deployment**: Provides a FastAPI endpoint for real-time captioning and a Docker configuration for scalable deployment.

---

## Project Structure

```bash
image_captioning/
├── api/
│   ├── inference.py              # Inference logic for caption generation
│   └── main.py                   # FastAPI application
├── data/
│   └── add_dataset_here/         # Dataset (images and captions.txt)
├── saved_models/
│   └── best_model.pth            # Trained model checkpoint
├── steps/
│   ├── extract_features.py       # Feature extraction step
│   ├── load_captions.py          # Caption loading and cleaning step
│   ├── prepare_tokenizer.py      # Tokenizer preparation step
│   ├── split_dataset.py          # Dataset splitting step
│   ├── train_model.py            # Model training step
│   ├── evaluate_model.py         # Model evaluation step
├── constants.py                  # Constants and configurations
├── models.py                     # Model and dataset class definitions
├── pipeline.py                   # ZenML pipeline definition
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **Flickr8k** dataset for training and evaluation. You can download it from:

| Dataset Name | Size   | Link                                                                 |
| --- |--------|----------------------------------------------------------------------|
| **Flickr8k** | \~1 GB | [Dataset Link](https://www.kaggle.com/datasets/adityajn105/flickr8k) |

### Dataset Structure

- **Images**: Stored in `data/Images/` as JPEG files.
- **Captions**: Stored in `data/captions.txt` with format `image_name,caption`.

**Setup Instructions**:

1. Download the dataset and extract it to `data/`.
2. Ensure `captions.txt` and the `Images/` directory are in the correct location.

---

## Model Training

1. **Preprocessing**:
   - **Feature Extraction**: Uses ResNet50 to extract 2048-dimensional features from images (`steps/extract_features.py`).
   - **Caption Processing**: Cleans and tokenizes captions (`steps/load_captions.py`, `steps/prepare_tokenizer.py`).
2. **Training**:
   - The model combines ResNet50 features with an LSTM and attention mechanism for caption generation (`steps/train_model.py`).
   - Training includes early stopping and learning rate scheduling.
3. **Model Artifacts**:
   - The best model checkpoint (`best_model.pth`) is saved in `saved_models/`.
   - Additional artifacts (e.g., `features.pkl`, `tokenizer.pkl`) are saved in the working directory.

### Training Environment

The model can be trained on a GPU (recommended) or CPU. Google Colab or a local machine with CUDA support is suitable for faster training.

---

## Pipeline (ZenML)

**ZenML** orchestrates the pipeline, ensuring reproducibility and modularity:

- **Steps**:

  - `extract_features.py`: Extracts image features using ResNet50.
  - `load_captions.py`: Loads and cleans captions.
  - `prepare_tokenizer.py`: Creates and saves the tokenizer.
  - `split_dataset.py`: Splits data into train and test sets.
  - `train_model.py`: Trains the captioning model.
  - `evaluate_model.py`: Evaluates the model using BLEU scores.

- **Execution**: Run the pipeline using:

  ```bash
  python pipeline.py
  ```

---

## API (FastAPI)

A **FastAPI** application is provided in `api/main.py` for real-time inference:

- **Endpoints**:
  - `POST /predict`: Accepts an image and returns a generated caption.
  - `GET /health`: Checks the API’s health status.

**Run the API Locally**:

```bash
uvicorn api.main:app --reload
```

Access the interactive API docs at http://127.0.0.1:8000/docs.

**Example Request**:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

**Response**:

```json
{"caption": "a dog running in a park"}
```

---

## Docker

A `Dockerfile` is provided for containerizing the FastAPI service with the trained model:

1. **Build the Image**:

   ```bash
   docker build -t image-captioner:latest .
   ```

2. **Run the Container**:

   ```bash
   docker run -p 8000:8000 -v $(pwd)/content:/content image-captioner:latest
   ```

3. Access the API at http://localhost:8000/docs.

---

## Evaluation Metrics

The model is evaluated using BLEU scores, computed in `steps/evaluate_model.py`. Example metrics (values depend on training):

| Metric     | Value (Example) |
|------------|-----------------|
| **BLEU-1** | 0.52            |
| **BLEU-2** | 0.33            |
| **BLEU-3** | 0.20            |
| **BLEU-4** | 0.12            |

Run the pipeline to generate actual metrics for your dataset.

---

## Setup & Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Abdallah2A/image_interpreter.git
   ```

2. **Install Dependencies**:

   ```bash
   cd image_interpreter
   pip install -r requirements.txt
   ```

3. **Configure ZenML**:

   ```bash
   zenml init
   ```

4. **Download the Dataset**:

   - Download Flickr8k (or your dataset) and place it in `data/dataset/`.

5. **Run the Pipeline**:

   ```bash
   python pipeline.py
   ```

6. **Start the FastAPI Service**:

   ```bash
   uvicorn api.main:app --reload
   ```

7. **Test the Inference**:

   - Use a tool like cURL or Postman to send a POST request to `http://localhost:8000/predict` with an image file.

---

## License

This project is free to use.

---

## References

- PyTorch - Deep learning framework used for model training and inference.
- ZenML - MLOps framework for building portable, production-ready pipelines.
- FastAPI - Modern, high-performance web framework for building APIs with Python.
- Docker - Container platform for packaging and deploying applications.
- Flickr8k Dataset - Dataset for image captioning.

---

Enjoy using **Image Interpreter** and happy captioning!
