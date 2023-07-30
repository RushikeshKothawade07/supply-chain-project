from fastapi import FastAPI, HTTPException, Body
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import string
import nltk  # Add the import statement for the 'nltk' module
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
app = FastAPI()
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# Load the tokenizer from distilbert_tokenizer3 folder
tokenizer_path = "distilbert_tokenizer3"
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

# Load the model from distilled_combined_model3 folder
model_path = "distilled_combined_model3"

# Define class labels
class_labels = ["commodities", "compliance", "delays", "environmental", "financial health", "supplier market"]

# Function to preprocess text
def preprocess_text(text):
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
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back to form a preprocessed sentence
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Function to make predictions
def predict_class(title, paragraph):
    input_text = title + " " + paragraph
    preprocessed_text = preprocess_text(input_text)

    # Tokenize the input text and convert to tensor
    inputs = tokenizer.encode_plus(preprocessed_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class = class_labels[predicted_class_index]

    return predicted_class, probabilities[0].tolist()  # Convert probabilities tensor to a list


@app.get("/")
def home():
    return {"health_check": "ooooOK"}
# API endpoint for prediction
@app.post("/predict/")
async def predict(data: dict = Body(...)):
    try:
        title = data.get("title")
        paragraph = data.get("paragraph")

        if not title or not paragraph:
            raise HTTPException(status_code=400, detail="Both 'title' and 'paragraph' fields are required.")

        predicted_class, probabilities = predict_class(title, paragraph)
        result = {
            "predicted_class": predicted_class,
            "probabilities": {class_labels[i]: prob.item() if isinstance(prob, torch.Tensor) else prob for i, prob in enumerate(probabilities)},
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))
