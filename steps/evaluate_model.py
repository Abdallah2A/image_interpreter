import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from zenml import step
from constants import BEAM_WIDTH, DEVICE
from models import ImageCaptionModel


@step
def evaluate_model(
        test_ids: list,
        mapping: dict,
        features: dict,
        tokenizer,
        max_length: int,
        model_path: str
) -> dict:
    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def beam_search(model, img_feat, tokenizer, max_length, device, beam_width=BEAM_WIDTH):
        model.eval()
        img_feat = torch.tensor(np.squeeze(img_feat), dtype=torch.float32).to(device).unsqueeze(0)
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
                top_probs, top_idx = probs[0].topk(beam_width)

                for i in range(beam_width):
                    next_seq = torch.cat([seq, top_idx[i].unsqueeze(0)])
                    next_score = score + torch.log(top_probs[i]).item()
                    all_candidates.append([next_seq, next_score])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        if finished:
            best = max(finished, key=lambda x: x[1])
        else:
            best = max(sequences, key=lambda x: x[1])

        caption = []
        for idx in best[0]:
            word = idx_to_word(idx.item(), tokenizer)
            if word and word not in ['startseq', 'endseq']:
                caption.append(word)

        return ' '.join(caption)

    model = ImageCaptionModel(len(tokenizer.word_index) + 1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    actual, predicted = [], []
    for img_id in tqdm(test_ids):
        caps = [c.split()[1:-1] for c in mapping[img_id]]
        actual.append(caps)
        pred = beam_search(model, features[img_id], tokenizer, max_length, DEVICE).split()
        predicted.append(pred)

    bleu_scores = {
        'BLEU-1': corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)),
        'BLEU-2': corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)),
        'BLEU-3': corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)),
        'BLEU-4': corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    }
    return bleu_scores
