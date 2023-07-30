import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Load the fine-tuned DistilBERT model and tokenizer
# model_path = './TTC4900Model'  # Replace with the path to your fine-tuned model
# tokenizer = DistilBertTokenizer.from_pretrained(model_path)
# model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Define class labels
class_labels = ["commodities", "compliance", "delays", "environmental", "financial health", "supplier market"]

# Function for preprocessing text
def preprocess_text(title, paragraph):
    # Combine the title and paragraph
    text = title + ' ' + paragraph

    # Lowercasing
    text = text.lower()

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization or stemming (you can choose one based on your requirement)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back to form a preprocessed sentence
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Function to make predictions
def predict_class(title, paragraph):
    input_text = title + " " + paragraph
    preprocessed_text = preprocess_text(input_text)

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(preprocessed_text, truncation=True, padding=True, return_tensors="tf")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Make prediction
    with torch.no_grad():
        logits = model([input_ids, attention_mask])[0]

    # Get predicted class label and probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
    predicted_class = class_labels[probabilities.argmax()]

    return predicted_class, probabilities

# Streamlit app
def main():
    st.title("Multi-Class Classification App")
    st.write("Enter a title and paragraph to predict its class.")

    # Input fields for title and paragraph
    title_input = st.text_input("Enter the title:")
    paragraph_input = st.text_area("Enter the paragraph:")

    if st.button("Predict"):
        # Perform prediction
        if title_input and paragraph_input:
            predicted_class, probabilities = predict_class(title_input, paragraph_input)
            st.write("Predicted Class:", predicted_class)

            # Display probabilities of all classes
            st.write("Probabilities:")
            probabilities_df = pd.DataFrame({"Class": class_labels, "Probability": probabilities})
            st.dataframe(probabilities_df)

if __name__ == "__main__":
    main()
