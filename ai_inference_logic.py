import os
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model download link
MODEL_URL = "https://drive.google.com/uc?id=1au-vR0Qwd1bIIDRD1DzOO7e7Gk1Fa1t0"
MODEL_FILE = "model.safetensors"

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
        print("Model downloaded successfully.")
        

# Download the model
download_model()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSequenceClassification.from_pretrained("./", state_dict=load_file(MODEL_FILE))

# Define the post-processing logic
def apply_post_processing(input_text, predicted_esi_level, logits):
    # Define keyword lists and processing logic
    critical_keywords = [
        "vomiting blood", "unresponsive", "not breathing", "loss of vision", "severe pain", "slurred speech",
        "chest pain", "difficulty breathing", "stroke", "severe bleeding", "fainting", "heart attack",
        "profuse bleeding", "head trauma", "trauma", "severe head injury", "unconscious", "seizure lasting longer than 5 minutes",
        "heavy bleeding"
    ]
    pregnancy_keywords_critical = [
        "reduced fetal movement", "no fetal movement", "severe abdominal pain during pregnancy"
    ]
    confidence = torch.softmax(logits, dim=0).max().item()

    if "pregnant" in input_text.lower():
        if "pain" in input_text.lower():
            if "severe" in input_text.lower() or confidence < 0.85:
                return 1
            return 2
        return max(2, predicted_esi_level)

    if any(keyword in input_text.lower() for keyword in critical_keywords):
        return 1

    if confidence < 0.7:
        return max(2, predicted_esi_level)

    return predicted_esi_level

# Function to handle predictions
def predict_with_logic(input_texts):
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1).tolist()
        predicted_esi_levels = [pred + 1 for pred in predicted_classes]  # Convert to ESI Levels (1-5)

        # Apply post-processing logic
        adjusted_esi_levels = [
            apply_post_processing(input_texts[i], predicted_esi_levels[i], logits[i])
            for i in range(len(input_texts))
        ]
    return adjusted_esi_levels

# Example usage
test_cases = [
    "Unresponsive and not breathing | Age: 45 | Gender: Female",
    "Pregnant woman with abdominal pain and no fetal movement for hours",
    "Sore throat and mild fever",
    "Request for a sick leave note",
]

adjusted_predictions = predict_with_logic(test_cases)
for i, case in enumerate(test_cases):
    print(f"Input: {case}")
    print(f"Adjusted ESI Level: {adjusted_predictions[i]}")
