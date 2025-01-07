from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Set model and tokenizer path
model_path = './trained_model'  # Adjust the path as per your deployment setup
tokenizer_path = './trained_model'  # Same for tokenizer

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route('/')
def home():
    return "Welcome to the ESI Model Inference API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the input has the 'texts' field (allowing multiple texts)
    data = request.json
    if not data or 'texts' not in data:
        return jsonify({'error': 'No text provided'}), 400

    texts = data['texts']
    if isinstance(texts, str):  # If only one text is passed as a string, wrap it in a list
        texts = [texts]

    # Tokenize input
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128, is_split_into_words=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    predicted_labels = []

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1).tolist()

    # Return predictions as a list of labels
    return jsonify({'predicted_labels': predicted_labels})

if __name__ == "__main__":
    # Ensure app listens on the correct port from environment variables
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
