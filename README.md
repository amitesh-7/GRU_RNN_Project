Text Generation with a GRU Neural Network

This project uses a Gated Recurrent Unit (GRU) network, built with TensorFlow and Keras, to generate new text in the style of "The Adventures of Sherlock Holmes" by Arthur Conan Doyle.
📝 Description

The Python script (GNU_RNN.py) reads the full text of the book, trains an LSTM model to learn the patterns, vocabulary, and sentence structure, and then uses the trained model to generate new text one word at a time.
⚙️ How It Works

The process involves several key steps:

    Load Data: The script reads the hamlet.txt file.

    Tokenization: The text is broken down into a sequence of numerical tokens, where each unique word is assigned an integer. The tokenizer is saved as tokenizer.pickle.

    Sequence Creation: The script creates input-output pairs (n-grams) from the text. For example, for the sentence "the cat sat", it creates pairs like (the cat, sat).

    Model Building: An GRU model is constructed with an Embedding layer, an GRU layer, and a Dense output layer.

    Training: The model is trained on the prepared sequences for 100 epochs to learn the linguistic patterns of the source text.

    Saving: The trained model is saved as text_generation_model.h5, and the tokenizer is saved as tokenizer.pickle.

    Text Generation: After training, the script uses a "seed text" (e.g., "I have heard") to predict the next word, appends it, and repeats the process to generate a new passage.

📋 Requirements

To run this script, you will need Python and the following libraries:

    tensorflow

    numpy

You can install them using pip:

pip install tensorflow numpy

▶️ How to Run

    Place the Data File: Make sure the text file hamlet.txt is in the same directory as the Python script.

    Execute the Script: Run the script from your terminal.

    python GRU_RNN.py

    Output: The script will first print a summary of the model and then show the training progress for 100 epochs. After training is complete, it will save the model and tokenizer files and print a sample of the newly generated text.

📂 Files Generated

    tokenizer.pickle: A saved copy of the tokenizer containing the word-to-index vocabulary.

    text_generation_model.h5: The trained Keras model, including its architecture and learned weights.
