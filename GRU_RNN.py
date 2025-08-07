import nltk
import numpy as np
import tensorflow as tf
import pickle
from nltk.corpus import gutenberg
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout

# 2. Download and Load Hamlet Text Corpus

nltk.download('gutenberg')  # Download Gutenberg corpus
data = gutenberg.raw('shakespeare-hamlet.txt')  # Load raw text of Hamlet

# 3. Save the Text and Read as Lowercase

with open('hamlet.txt', 'w') as file:
    file.write(data)  # Save text to a file

with open('hamlet.txt', 'r') as file:
    text = file.read().lower()  # Read file and convert to lowercase

# 4. Tokenize the Entire Text


tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])  # Fit tokenizer on full text
total_words = len(tokenizer.word_index) + 1  # Calculate vocabulary size

# 5. Generate n-gram Sequences from Each Line

input_sequence = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]  # Convert line to token list
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]  # Create n-gram sequences
        input_sequence.append(n_gram_sequence)

# 6. Pad Sequences to the Same Length

max_seq_len = max(len(x) for x in input_sequence)  # Find max sequence length
input_sequence = pad_sequences(input_sequence, padding='pre', maxlen=max_seq_len)  # Pad all sequences

# 7. Prepare Features (X) and Labels (Y)

x, y = input_sequence[:, :-1], input_sequence[:, -1]  # Last word is the label
y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # One-hot encode the labels

# 8. Split Data into Training and Testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# 9. Define the GRU-Based Model 

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100))  # Word embedding layer
model.add(GRU(200, return_sequences=True))  # First GRU layer
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(GRU(150))  # Second GRU layer
model.add(Dense(total_words, activation='softmax'))  # Output layer

# Build and display model summary
model.build(input_shape=(None, max_seq_len - 1))
model.summary()
 
# 10. Compile and Train the Model 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    verbose=1,
    epochs=100
)

# 11. Predict the Next Word Function

def predicts_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]  # Tokenize input text

    # Trim to match input length used during training
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')  # Pad input
    predicted = model.predict(token_list, verbose=0)  # Make prediction
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Get index of highest probability

    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

# 12. Test the Prediction Function
input_text = "Is not this something more then"
print(f"Input Text: {input_text}")

max_sequence_len = model.input_shape[1] + 1
next_word = predicts_next_word(model, tokenizer, input_text, max_sequence_len)

print(f"Predicted Next Word: {next_word}")

# # 13. Save Model and Tokenizer
# 
model.save('predicts_next_word_LSTM.h5')  # Save trained model

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)  # Save tokenizer
