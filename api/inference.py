import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import tensorflow as tf
from models import ImageCaptionModel
from constants import DEVICE, BEAM_WIDTH
import torch.nn as nn


def load_artifacts(model_path: str, tokenizer_path: str):
    # Load model
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    model = ImageCaptionModel(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load ResNet50 for feature extraction
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1]).to(DEVICE).eval()

    return model, tokenizer, resnet50, vocab_size


def extract_image_features(image: Image.Image, resnet50: nn.Module) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = resnet50(image_tensor)
    return feature.cpu().numpy().squeeze()


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(image: Image.Image, model_path: str, tokenizer_path: str, device: torch.device) -> str:
    # Load artifacts
    model, tokenizer, resnet50, vocab_size = load_artifacts(model_path, tokenizer_path)
    max_length = 35  # Should match training max_length; ideally load from artifact

    # Extract image features
    img_feat = extract_image_features(image, resnet50)

    # Beam search
    img_feat = torch.tensor(img_feat, dtype=torch.float32).to(device).unsqueeze(0)
    start_token = tokenizer.word_index['startseq']
    end_token = tokenizer.word_index['endseq']

    sequences = [[torch.tensor([start_token], dtype=torch.long).to(device), 0.0]]
    finished = []

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1].item() == end_token:
                finished.append([seq, score])
                continue

            seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq.cpu().numpy()],
                                                                       maxlen=max_length, padding='post')
            seq_padded = torch.tensor(seq_padded, dtype=torch.long).to(device)

            with torch.no_grad():
                output = model(img_feat, seq_padded)
            probs = torch.softmax(output, dim=1)
            top_probs, top_idx = probs[0].topk(BEAM_WIDTH)

            for i in range(BEAM_WIDTH):
                next_seq = torch.cat([seq, top_idx[i].unsqueeze(0)])
                next_score = score + torch.log(top_probs[i]).item()
                all_candidates.append([next_seq, next_score])

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:BEAM_WIDTH]

    # Select best sequence
    best = max(finished or sequences, key=lambda x: x[1])

    # Convert to words
    caption = []
    for idx in best[0]:
        word = idx_to_word(idx.item(), tokenizer)
        if word and word not in ['startseq', 'endseq']:
            caption.append(word)

    return ' '.join(caption)
