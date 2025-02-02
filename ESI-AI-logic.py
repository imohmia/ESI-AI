from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = r"C:\Users\mmmaf\OneDrive\Desktop\i\Triage\fine_tuned_clinical_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test cases
large_test_cases = [
    # Critical cases
    "Sudden severe chest pain and shortness of breath | Age: 67 | Gender: Male",
    "Unresponsive and not breathing | Age: 45 | Gender: Female",
    "Stroke symptoms with one-sided weakness and slurred speech | Age: 60 | Gender: Male",
    "Severe head trauma after a fall | Age: 28 | Gender: Female",
    "Bleeding profusely from a deep cut on the leg | Age: 35 | Gender: Male",
    "Child hit by a car and unresponsive",
    "Heavy bleeding from a wound",
    
    # Moderate cases
    "High fever for three days and persistent cough | Age: 5 | Gender: Female",
    "Pregnant woman with abdominal pain and no fetal movement for hours | Age: 30 | Gender: Female",
    "Dizziness and nausea after dehydration | Age: 45 | Gender: Male",
    "Child with severe ear pain and high fever | Age: 7 | Gender: Male",
    "Teenager with high fever and painful rash",
    
    # Borderline or simple cases
    "Cough",
    "Sore throat",
    "Runny nose",
    "Pain in wrist",
    "Bruise on arm",
    "Request for medication refill",
    "Request for a sick leave note",
    "im pregnant i have pain",
]

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

# Tokenize the test cases
inputs = tokenizer(large_test_cases, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

# Convert predictions to ESI levels (add 1 to match original levels)
predicted_esi_levels = [pred + 1 for pred in predictions]

# Apply post-processing
for i, test in enumerate(large_test_cases):
    adjusted_esi_level = apply_post_processing(test, predicted_esi_levels[i], logits[i])
    print(f"Input: {test}")
    print(f"Original Predicted ESI Level: {predicted_esi_levels[i]}")
    print(f"Adjusted ESI Level: {adjusted_esi_level}")
    print()
