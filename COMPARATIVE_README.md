# Image Captioning Project: Version Comparison

This document compares **Version 1** (ResNet50 + LSTM) and **Version 2** (VisionTransformer + Transformer decoder) across key components.

---

## 1. Overview

| Aspect   | Version 1                                                | Version 2                                                |
|----------|----------------------------------------------------------|----------------------------------------------------------|
| Overview | A real-time image captioning system for generating descriptive captions from images. **Image Interpreter** leverages a deep learning model combining **ResNet50** for feature extraction and an **LSTM with attention** for caption generation.                                     | **Image Interpreter** aims to generate accurate and descriptive captions for images in real-time using a Vision Transformer encoder and Transformer decoder.                                     |

---

## 2. Data

| Aspect     | Version 1                            | Version 2                                        |
|------------|--------------------------------------|--------------------------------------------------|
| Dataset    | Flickr8k dataset (~1 GB, single reference) using captions.txt                 | COCO2014 dataset (~13 GB), JSON COCO annotations |
| Annotations| `captions.txt` single-reference     | JSON COCO (`captions_train2014.json`)            |

---

## 3. Model Architecture

| Component      | Version 1                                   | Version 2                                            |
|----------------|---------------------------------------------|------------------------------------------------------|
| Encoder        | ResNet50 (2048-dim features)               | VisionTransformer (patch_size=8, embed_dim=192)     |
| Decoder        | LSTM with attention                         | Transformer decoder (6 layers, 8 heads)             |
| Attention      | Additive attention over LSTM                | Multi-head self/cross attention                      |

---

## 4. Pipeline Steps

| Step Name            | Version 1                                            | Version 2                          |
|----------------------|------------------------------------------------------|------------------------------------|
| Feature Extraction   | `steps/extract_features.py` (ResNet50 → features.pkl) | Data loader loads and transforms images |
| Caption Loading      | `steps/load_captions.py`                             | Integrated in data_loader step     |
| Tokenizer Prep       | `steps/prepare_tokenizer.py`                         | Performed inside data_loader step  |
| Dataset Split        | `steps/split_dataset.py`                             | Part of data_loader step           |
| Training             | `steps/train_model.py`                               | `steps/training.py`                |
| Evaluation           | `steps/evaluate_model.py`                            | `steps/evaluation.py`              |

---

## 5. API & Deployment

| Aspect       | Version 1                                  | Version 2                          |
|--------------|--------------------------------------------|------------------------------------|
| API          | FastAPI (`api/main.py`), POST /predict     | FastAPI (`api/server.py`), POST /caption |
| Upload Type  | Multipart file `file`                     | Multipart file `file`              |
| Docker       | `Dockerfile` bundles app + model          | Similar Dockerfile, exposes port 8000 |

---

## Usage

Both versions follow:

```bash
# Run pipeline
python pipeline.py

# Start API
uvicorn api.server:app --reload

# Inference
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/caption
```
