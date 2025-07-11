import os
import random
from typing import Tuple
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from zenml.steps import step


class SampleCaption:
    def __call__(self, captions):
        return random.choice(captions)


@step
def data_loader_step() -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    image_size = 128
    batch_size = 128
    data_root = 'dataset'

    train_transform = transforms.Compose([transforms.Resize(image_size),
                                          transforms.RandomCrop(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          transforms.RandomErasing(p=0.5)])

    val_transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    train_ds = datasets.CocoCaptions(root=os.path.join(data_root, 'train2014'),
                                     annFile=os.path.join(data_root, 'annotations', 'captions_train2014.json'),
                                     transform=train_transform,
                                     target_transform=SampleCaption())
    val_ds = datasets.CocoCaptions(root=os.path.join(data_root, 'val2014'),
                                   annFile=os.path.join(data_root, 'annotations', 'captions_val2014.json'),
                                   transform=val_transform,
                                   target_transform=SampleCaption())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return train_loader, val_loader, tokenizer
