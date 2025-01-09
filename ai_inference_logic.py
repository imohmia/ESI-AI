import os
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model file path
MODEL_FILE = "model.safetensors"

# Check if the model exists locally
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found. Please ensure the model is uploaded to the correct directory.")

# Load tokenizer and model
print("Loading tokenizer and model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("./")  # Ensure tokenizer matches the model

    # Load the state dictionary using the safetensors library
    state_dict = load_file(MODEL_FILE)

    # Load the model with the state dictionary
    model = AutoModelForSequenceClassification.from_pretrained(
        "./",
        state_dict=state_dict,
        trust_remote_code=True
    )
    print("Tokenizer and model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load the tokenizer or model: {e}")

# Define your prediction function
def predict_with_logic(input_texts: list[str]):
    # Tokenize inputs
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()
    return logits, [pred + 1 for pred in predictions]  # Adjust for ESI levels starting at 1

# Post-processing function
def apply_post_processing(input_text, predicted_esi_level, logits):
    # Define expanded rules for adjusting predictions
    critical_keywords = [
        "vomiting blood", "unresponsive", "not breathing", "loss of vision", "severe pain", "slurred speech",
        "chest pain", "difficulty breathing", "stroke", "severe bleeding", "fainting", "heart attack",
        "profuse bleeding", "head trauma", "severe head injury", "unconscious", "seizure lasting longer than 5 minutes"
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
    low_risk_keywords = [
        "chronic back pain", "mild headache", "mild rash", "sprained ankle"
    ]
    
    # Get confidence scores
    confidence = torch.softmax(logits, dim=0).max().item()

    # Refined Priority-Based Keyword Logic
    if any(keyword in input_text.lower() for keyword in critical_keywords):
        return 1  # Critical cases are always Level 1
    if any(keyword in input_text.lower() for keyword in pregnancy_keywords_critical):
        return 1  # Critical pregnancy cases escalate to Level 1
    if any(keyword in input_text.lower() for keyword in moderate_keywords):
        return max(2, predicted_esi_level)  # Ensure at least Level 2
    if any(keyword in input_text.lower() for keyword in borderline_keywords):
        return min(4, predicted_esi_level)  # Ensure at most Level 4
    if any(keyword in input_text.lower() for keyword in low_risk_keywords):
        return 5  # Ensure Level 5 for non-urgent cases

    # Handle low-confidence cases
    if confidence < 0.75:  # Increased threshold for critical safety
        return max(2, predicted_esi_level)

    # Default behavior: Return the predicted level
    return predicted_esi_level

# Test cases
def evaluate_cases():
    large_test_cases = [
        "Sudden severe chest pain and difficulty breathing | Age: 67 | Gender: Male",
        "Child with a sore throat and mild fever for 2 days | Age: 5 | Gender: Female",
        "Profuse bleeding from a leg wound after a fall | Age: 30 | Gender: Male",
        "Persistent cough and weight loss over months | Age: 65 | Gender: Male",
        "High fever and severe ear pain | Age: 10 | Gender: Female",
        # Add more test cases as needed
    ]

    logits, predicted_esi_levels = predict_with_logic(large_test_cases)
    results = []

    for i, test in enumerate(large_test_cases):
        adjusted_esi_level = apply_post_processing(test, predicted_esi_levels[i], logits[i])
        results.append({
            "input": test,
            "original_predicted_esi": predicted_esi_levels[i],
            "adjusted_esi": adjusted_esi_level
        })
        print(f"Input: {test}\nOriginal Predicted ESI Level: {predicted_esi_levels[i]}\nAdjusted ESI Level: {adjusted_esi_level}\n")
    return results

if __name__ == "__main__":
    evaluate_cases()
