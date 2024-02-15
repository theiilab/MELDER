import numpy as np
import tensorflow as tf

# Define the character-level language model class
class CharLanguageModel:
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        
        # Define layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, initial_state=None):
        x = self.embedding(inputs)
        if initial_state is not None:
            x, state_h, state_c = self.rnn(x, initial_state=initial_state)
        else:
            x, state_h, state_c = self.rnn(x)
        logits = self.dense(x)
        return logits, state_h, state_c

# Generate text from the character-level language model
def generate_text_char(model, start_string, char2idx, idx2char, num_generate=100, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions, state_h, state_c = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

# Define the word-level language model class
class WordLanguageModel:
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        
        # Define layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, initial_state=None):
        x = self.embedding(inputs)
        if initial_state is not None:
            x, state_h, state_c = self.rnn(x, initial_state=initial_state)
        else:
            x, state_h, state_c = self.rnn(x)
        logits = self.dense(x)
        return logits, state_h, state_c

# Generate text from the word-level language model
def generate_text_word(model, start_string, word2idx, idx2word, num_generate=100, temperature=1.0):
    input_eval = [word2idx[s] for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions, state_h, state_c = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2word[predicted_id])
    return (start_string + ' ' + ' '.join(text_generated))
