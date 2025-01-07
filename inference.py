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
    # Ensure the input has the 'text' field
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    return jsonify({'predicted_label': predicted_label})

if __name__ == "__main__":
    # Ensure app listens on the correct port from environment variables
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
