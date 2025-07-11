from zenml import pipeline
from steps.extract_features import extract_features
from steps.load_captions import load_and_clean_captions
from steps.prepare_tokenizer import prepare_tokenizer
from steps.split_dataset import split_dataset
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


@pipeline
def image_captioning_pipeline():
    features = extract_features()
    mapping = load_and_clean_captions(features)
    tokenizer, vocab_size, max_length = prepare_tokenizer(mapping)
    train_ids, test_ids = split_dataset(mapping)
    model_path = train_model(train_ids, mapping, features, tokenizer, vocab_size, max_length)
    bleu_scores = evaluate_model(test_ids, mapping, features, tokenizer, max_length, model_path)
    return bleu_scores


if __name__ == "__main__":
    pipeline = image_captioning_pipeline()
    pipeline.run()
