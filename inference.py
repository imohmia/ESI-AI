from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = './trained_model'  # Your model path
tokenizer_path = './trained_model'  # Same for the tokenizer

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    return jsonify({'predicted_label': predicted_label})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
