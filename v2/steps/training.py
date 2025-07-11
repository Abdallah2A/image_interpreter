import os
import torch
import torch.nn as nn
from torch import optim
from zenml.steps import step
from steps.model_definition import TokenDrop


@step
def training_step(train_loader, val_loader, model, tokenizer):
    learning_rate = 1e-4
    nepochs = 200
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    td = TokenDrop(0.5)

    history = {'train_loss': [], 'val_loss': []}
    start_epoch = 0

    for epoch in range(start_epoch, nepochs):
        model.train()
        train_loss = 0.0
        for images, captions in train_loader:
            images = images.to(model.encoder.fc_in.weight.device)
            tokens = tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
            token_ids = tokens.input_ids.to(images.device)
            padding_mask = tokens.attention_mask.to(images.device)

            target_ids = torch.cat(
                (token_ids[:, 1:], torch.zeros(images.size(0), 1, dtype=torch.long, device=images.device)), dim=1)
            tokens_in = td(token_ids)

            with torch.cuda.amp.autocast():
                pred = model(images, tokens_in, padding_mask)
                loss = (loss_fn(pred.transpose(1, 2), target_ids) * padding_mask).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(images.device)
                tokens = tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
                token_ids = tokens.input_ids.to(images.device)
                padding_mask = tokens.attention_mask.to(images.device)

                target_ids = torch.cat(
                    (token_ids[:, 1:], torch.zeros(images.size(0), 1, dtype=torch.long, device=images.device)), dim=1)
                pred = model(images, token_ids, padding_mask)
                loss = (loss_fn(pred.transpose(1, 2), target_ids) * padding_mask).mean()
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)

        print(f"Epoch {epoch + 1}/{nepochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if (epoch + 1) % 5 == 0:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

    final_path = os.path.join('model', 'final_model_v2.pth')
    torch.save(model.state_dict(), final_path)
    return model, history
