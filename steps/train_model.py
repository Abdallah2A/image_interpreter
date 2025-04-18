import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from zenml import step
from constants import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WORKING_DIR, DEVICE
from models import CaptionDataset, ImageCaptionModel


@step
def train_model(
        train_ids: list,
        mapping: dict,
        features: dict,
        tokenizer,
        vocab_size: int,
        max_length: int
) -> str:
    model = ImageCaptionModel(vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_dataset = CaptionDataset(train_ids, mapping, features, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    best_loss = float('inf')
    patience = 7
    counter = 0
    model_path = os.path.join(WORKING_DIR, 'best_model.pth')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for img_feat, seq, target in tqdm(train_loader):
            img_feat, seq, target = img_feat.to(DEVICE), seq.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(img_feat, seq)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    return model_path
