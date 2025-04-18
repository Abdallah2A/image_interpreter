import os
import re
from tqdm import tqdm
from zenml import step
from constants import BASE_DIR


@step
def load_and_clean_captions(features: dict) -> dict:
    with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
        captions_doc = f.read()

    mapping = {}
    for line in tqdm(captions_doc.split('\n')[1:-1]):
        if not line:
            continue
        tokens = line.split(',', 1)
        if len(tokens) < 2:
            continue
        img_id, caption = tokens
        img_id = img_id.split('.')[0]
        if img_id in features:
            mapping.setdefault(img_id, []).append(caption.strip())

    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = re.sub('[^a-zA-Z\s]', '', caption)
            caption = ' '.join([w for w in caption.split() if len(w) > 1])
            captions[i] = f'startseq {caption} endseq'

    return mapping
