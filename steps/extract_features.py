import os
import pickle
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import models, transforms
from zenml import step
from constants import BASE_DIR, WORKING_DIR, EXPECTED_FEATURE_DIM, DEVICE
import torch.nn as nn


@step
def extract_features() -> dict:
    features_path = os.path.join(WORKING_DIR, 'features.pkl')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        sample_feature = next(iter(features.values()))
        if sample_feature.shape[-1] == EXPECTED_FEATURE_DIM:
            return features

    print("Extracting features using ResNet50...")
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1]).to(DEVICE).eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = {}
    directory = os.path.join(BASE_DIR, 'Images')
    for img_name in tqdm(os.listdir(directory)):
        try:
            img_path = os.path.join(directory, img_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feature = resnet50(image_tensor)
            features[img_name.split('.')[0]] = feature.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    pickle.dump(features, open(features_path, 'wb'))
    return features
