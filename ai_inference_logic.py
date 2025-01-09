import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

MODEL_URL = "https://drive.google.com/uc?id=1aFrGA06dTz5y0jk29T0Ou3pwQMToNYZt&export=download"
MODEL_FILE = "model.safetensors"

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")

# Download the model
download_model()

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
state_dict = load_file(MODEL_FILE)  # Use safetensors loader
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", state_dict=state_dict)
print("Model and tokenizer loaded successfully.")

# Function to apply post-processing logic
def apply_post_processing(input_text, predicted_esi_level, logits):
    critical_keywords = [
        "vomiting blood", "unresponsive", "not breathing", "loss of vision", "severe pain", "slurred speech",
        "chest pain", "difficulty breathing", "stroke", "severe bleeding", "fainting", "heart attack",
        "profuse bleeding", "head trauma", "trauma", "severe head injury", "unconscious", "seizure lasting longer than 5 minutes",
        "heavy bleeding", "bleeding heavily", "blood loss", "hemorrhage", "bleeding uncontrollably", "bleeding profusely",
        "bleeding a lot", "severe blood loss"
    ]
    pregnancy_keywords_critical = [
        "reduced fetal movement", "no fetal movement", "severe abdominal pain during pregnancy",
        "fetal movement stopped", "baby not moving", "severe bleeding during pregnancy",
        "preterm labor symptoms", "third trimester pain", "pregnant and in pain",
        "contractions with severe pain"
    ]
    confidence = torch.softmax(logits, dim=0).max().item()

    # Apply rules for post-processing
    if "pregnant" in input_text.lower():
        if "pain" in input_text.lower():
            if "severe" in input_text.lower() or confidence < 0.85:
                return 1
            return 2
        return max(2, predicted_esi_level)

    if any(keyword in input_text.lower() for keyword in pregnancy_keywords_critical):
        return 1
    if any(keyword in input_text.lower() for keyword in critical_keywords):
        return 1
    if confidence < 0.7:
        return max(2, predicted_esi_level)
    return predicted_esi_level

# Prediction logic
def predict_with_logic(input_texts):
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        predicted_esi_levels = [pred + 1 for pred in predictions]  # Convert to ESI Levels (1-5)

        # Apply post-processing logic
        adjusted_predictions = [
            apply_post_processing(input_texts[i], predicted_esi_levels[i], logits[i])
            for i in range(len(input_texts))
        ]
    return adjusted_predictions

# Example usage
if __name__ == "__main__":
    test_cases = [
        "Sudden severe chest pain and shortness of breath | Age: 67 | Gender: Male",
        "Unresponsive and not breathing | Age: 45 | Gender: Female",
        "Stroke symptoms with one-sided weakness and slurred speech | Age: 60 | Gender: Male",
        "Cough and runny nose | Age: 30 | Gender: Female"
    ]
    predictions = predict_with_logic(test_cases)
    for text, prediction in zip(test_cases, predictions):
        print(f"Input: {text}")
        print(f"Predicted ESI Level: {prediction}")
