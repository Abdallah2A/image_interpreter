import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import tensorflow as tf
from constants import HIDDEN_SIZE, EMBED_DIM, EXPECTED_FEATURE_DIM, DEVICE


class CaptionDataset(Dataset):
    def __init__(self, img_ids, mapping, features, tokenizer, max_length):
        self.samples = []
        for img_id in img_ids:
            for caption in mapping[img_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences([seq[:i]],
                                                                           maxlen=max_length, padding='post')[0]
                    out_seq = seq[i]
                    self.samples.append((features[img_id], in_seq, out_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_feat, seq, target = self.samples[idx]
        img_feat = np.squeeze(img_feat)
        return (torch.tensor(img_feat, dtype=torch.float32),
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        combined = torch.cat((hidden, encoder_outputs), dim=1)
        energy = self.tanh(self.attn(combined))
        attention = self.softmax(self.v(energy))
        context = attention * encoder_outputs
        return context, attention


class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, feature_dim=EXPECTED_FEATURE_DIM):
        super().__init__()
        self.image_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_feat, seq):
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)
        elif img_feat.dim() > 2:
            img_feat = img_feat.squeeze()
            if img_feat.dim() == 1:
                img_feat = img_feat.unsqueeze(0)

        img_out = self.image_net(img_feat)
        emb = self.embedding(seq)
        outputs = []
        hidden = torch.zeros(2, img_feat.size(0), HIDDEN_SIZE).to(DEVICE)
        cell = torch.zeros(2, img_feat.size(0), HIDDEN_SIZE).to(DEVICE)

        for t in range(seq.size(1)):
            context, _ = self.attention(hidden[-1], img_out)
            lstm_input = torch.cat((emb[:, t, :], context), dim=1).unsqueeze(1)
            lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            outputs.append(lstm_out.squeeze(1))

        outputs = torch.stack(outputs, dim=1)
        return self.fc(outputs[:, -1, :])
