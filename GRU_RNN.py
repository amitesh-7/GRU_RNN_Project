import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,GRU, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. Load Data
try:
    with open('hamlet.txt', 'r') as file:
        text_data = file.read()
except FileNotFoundError:
    print("Error: The text file was not found. Please ensure 'hamlet.txt' is in the same directory.")
    exit()

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data]) # Build the word-to-index vocabulary.

# Save the tokenizer to reuse it later for decoding.
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Create Input Sequences
# Split the text into lines to create training samples.
lines = text_data.split('\n')

# Convert lines of text into sequences of integers.
input_sequences = tokenizer.texts_to_sequences(lines)
# Filter out empty or very short sequences.
input_sequences = [seq for seq in input_sequences if len(seq) > 1]

# Create n-gram sequences for training
sequences = []
for line in input_sequences:
    for i in range(1, len(line)):
        n_gram_sequence = line[:i+1]
        sequences.append(n_gram_sequence)

# Pad all sequences to the same length for model input.
max_sequence_len = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Split data into inputs (X) and labels (y).
X = sequences[:, :-1] # All tokens except the last.
y = sequences[:, -1]  # The last token.

# One-hot encode the labels (y) for the categorical cross-entropy loss function.
vocab_size = len(tokenizer.word_index) + 1
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 4. Build the LSTM Model
model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=100))
model.add(GRU(200,return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(150))
model.add(Dense(vocab_size,activation='softmax'))
model.build(input_shape=(None, max_sequence_len - 1))

# 5. Compile and Train the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model.
print("\nStarting model training...")
model.fit(X, y, epochs=100, verbose=1)
print("Model training complete.")

# 6. Save the Trained Model
model.save('text_generation_model.h5')
print("Model and tokenizer have been saved.")


# 7. Generate New Text
# Start with an initial text to prompt the model.
seed_text = "I have heard"
next_words = 100

print(f"\n--- Generating Text\nSeed: '{seed_text}'")

for _ in range(next_words):
    # Prepare the seed text for prediction.
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Predict the next word's index.
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

    # Convert the index back to a word.
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break

    # Append the new word and repeat.
    seed_text += " " + output_word

print("\nGenerated Text:")
print(seed_text)