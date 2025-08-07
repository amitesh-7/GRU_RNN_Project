# 📚 Next Word Prediction using GRU (Hamlet Text)

This project uses a GRU-based deep learning model to predict the **next word** in a sentence, trained on the classic text **Hamlet** by William Shakespeare. It leverages **natural language processing (NLP)** techniques and **neural networks** to learn word patterns and generate predictions.

---

## 🚀 Features

- Loads and processes Shakespeare's *Hamlet* using NLTK
- Builds n-gram sequences for training
- GRU-based neural network architecture
- One-word prediction based on input text
- Trains on tokenized and padded sequences
- Saves trained model and tokenizer for future use

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts word indices into dense vectors
- **GRU Layers**: Two GRU (Gated Recurrent Unit) layers to capture word dependencies
- **Dropout Layer**: Prevents overfitting
- **Dense Output Layer**: Predicts the next word using softmax

---

## 🛠️ Requirements

Install required packages using:

```bash
pip install -r requirements.txt
