import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSequenceClassification.from_pretrained("./", state_dict=torch.load(MODEL_FILE))
print("Model and tokenizer loaded successfully.")

# Define the post-processing logic
def apply_post_processing(input_text, predicted_esi_level, logits):
   # Define expanded rules for adjusting predictions
    critical_keywords = [
        "vomiting blood", "unresponsive", "not breathing", "loss of vision", "severe pain", "slurred speech",
        "chest pain", "difficulty breathing", "stroke", "severe bleeding", "fainting", "heart attack",
        "profuse bleeding", "head trauma", "trauma", "severe head injury", "unconscious", "seizure lasting longer than 5 minutes",
        "heavy bleeding", "bleeding heavily", "blood loss", "hemorrhage", "bleeding uncontrollably", "bleeding profusely", "bleeding a lot", "severe blood loss"
    ]
    pregnancy_keywords_critical = [
        "reduced fetal movement", "no fetal movement", "severe abdominal pain during pregnancy", 
        "fetal movement stopped", "baby not moving", "severe bleeding during pregnancy", 
        "preterm labor symptoms", "third trimester pain", "pregnant and in pain", 
        "contractions with severe pain"
    ]
    moderate_keywords = [
        "persistent cough", "high fever", "rash", "dehydration", "infection symptoms",
        "ear pain with fever", "pain with swelling", "nausea and dizziness", 
        "uncontrolled vomiting", "child with severe pain"
    ]
    borderline_keywords = [
        "cough", "sore throat", "runny nose", "headache", "mild bruising"
    ]
    cough_keywords = ["cough with blood", "cough and fever", "cough with difficulty breathing"]
    # Get confidence scores
    confidence = torch.softmax(logits, dim=0).max().item()

    # 1. Pregnancy: Generic check for "pregnant" and "pain"
    if "pregnant" in input_text.lower():
        if "pain" in input_text.lower():
            if "severe" in input_text.lower() or confidence < 0.85:
                return 1  # Escalate to Level 1 for severe pain or low confidence
            return 2  # Escalate to Level 2 for generic pain cases
        # Generic escalation for any mention of "pregnant"
        return max(2, predicted_esi_level)

    # 2. Critical Pregnancy Keywords
    if any(keyword in input_text.lower() for keyword in pregnancy_keywords_critical):
        return 1  # Escalate to Level 1

    # 3. General Critical Keywords
    if any(keyword in input_text.lower() for keyword in critical_keywords):
        return 1  # Escalate to Level 1
        
    # 4. Cough Handling
    if "cough" in input_text.lower():
        if any(keyword in input_text.lower() for keyword in cough_keywords):
            return 3  # Escalate to Level 3 for combined symptoms
        return 4  # Default to Level 4 for isolated cough

    # 5. General Moderate Keywords
    if any(keyword in input_text.lower() for keyword in moderate_keywords):
        return max(2, predicted_esi_level)  # Ensure at least Level 2

    # 6. Borderline Cases
    if any(keyword in input_text.lower() for keyword in borderline_keywords):
        return min(4, predicted_esi_level)  # Ensure at most Level 4

    # 7. Low Confidence Handling
    if confidence < 0.7:  # Confidence threshold for escalation
        return max(2, predicted_esi_level)

    # 8. Default: No rule applies
    return predicted_esi_level

# Define the prediction function
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
