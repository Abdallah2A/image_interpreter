import numpy as np
from zenml import step


@step
def split_dataset(mapping: dict) -> tuple:
    image_ids = list(mapping.keys())
    np.random.shuffle(image_ids)
    split = int(0.9 * len(image_ids))
    train_ids, test_ids = image_ids[:split], image_ids[split:]
    return train_ids, test_ids
