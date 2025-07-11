import tensorflow as tf
from zenml import step


@step
def prepare_tokenizer(mapping: dict) -> tuple:
    all_captions = [cap for caps in mapping.values() for cap in caps]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(cap.split()) for cap in all_captions)
    return tokenizer, vocab_size, max_length
