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
    # Define critical keywords based on severity
    critical_keywords = [
        "vomiting blood", "unresponsive", "not breathing", "loss of vision", "severe pain", "slurred speech",
        "difficulty breathing", "stroke", "severe bleeding", "fainting", "heart attack", "head trauma",
        "severe head injury", "unconscious", "seizure lasting longer than 5 minutes", "heavy bleeding",
        "bleeding uncontrollably", "hemorrhage"
    ]
    pregnancy_keywords_critical = [
        "reduced fetal movement", "no fetal movement", "severe abdominal pain during pregnancy", 
        "fetal movement stopped", "severe bleeding during pregnancy", "preterm labor symptoms"
    ]
    chest_pain_keywords = [
        "chest pain and difficulty breathing", "chest pain with sweating", "chest pain radiating to left arm",
        "chest pain with dizziness", "severe chest pain", "chest pain after exercise"
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

    # Critical cases: Immediate escalation to Level 1
    if any(keyword in input_text.lower() for keyword in critical_keywords):
        return 1  # Escalate to Level 1

    # Pregnancy-related critical cases
    if any(keyword in input_text.lower() for keyword in pregnancy_keywords_critical):
        return 1  # Escalate to Level 1

    # High-risk chest pain cases (specific conditions)
    if any(keyword in input_text.lower() for keyword in chest_pain_keywords):
        return 1  # Escalate to Level 1 for specific chest pain contexts

    # General chest pain cases: Assess accompanying factors
    if "chest pain" in input_text.lower():
        if "difficulty breathing" in input_text.lower() or "severe" in input_text.lower():
            return 1  # Escalate if combined with critical symptoms
        else:
            return max(2, predicted_esi_level)  # Moderate risk without additional critical signs

    # High-risk cases: Moderate (Level 2)
    if any(keyword in input_text.lower() for keyword in moderate_keywords):
        return max(2, predicted_esi_level)  # Ensure at least Level 2

    # Borderline or low-risk cases
    if any(keyword in input_text.lower() for keyword in borderline_keywords):
        return min(4, predicted_esi_level)  # Ensure at most Level 4

    if any(keyword in input_text.lower() for keyword in low_risk_keywords):
        return 5  # Escalate downward to Level 5 for non-urgent cases

    # Low confidence handling: Escalate if uncertain
    if confidence < 0.7:  # Confidence threshold for escalation
        return max(2, predicted_esi_level)

    # Default: Return the predicted level if no rule applies
    return predicted_esi_level
    
# Test cases
def evaluate_cases():
    large_test_cases = [
               "Unresponsive and not breathing | Age: 45 | Gender: Female",
        "Vomiting blood and feeling dizzy | Age: 50 | Gender: Male",
        "Profuse bleeding after a car accident | Age: 34 | Gender: Male",
        "Persistent cough and fever for 3 days | Age: 40 | Gender: Female",
        "Mild sore throat and runny nose | Age: 25 | Gender: Male",
        "Chest pain and difficulty breathing | Age: 67 | Gender: Male",
        "Severe headache and slurred speech | Age: 60 | Gender: Male",
        "Child with a rash and high fever for 2 days | Age: 5 | Gender: Male",
        "Swollen ankle after falling down stairs | Age: 30 | Gender: Female",
        "Shortness of breath after a bee sting | Age: 25 | Gender: Male",
        "Child fell and hit head, now drowsy and vomiting | Age: 7 | Gender: Male",
        "Uncontrolled vomiting and severe dehydration | Age: 18 | Gender: Female",
        "Pregnant woman reporting no fetal movement | Age: 31 | Gender: Female (pregnant)",
        "Persistent back pain with no new symptoms | Age: 48 | Gender: Male",
        "Seizure lasting longer than 5 minutes | Age: 42 | Gender: Female",
        "Loss of vision in one eye and nausea | Age: 72 | Gender: Female",
        "Heart palpitations and mild dizziness | Age: 50 | Gender: Male",
        "Heavy bleeding from a deep cut on the arm | Age: 27 | Gender: Female",
        "Coughing up blood and chest discomfort | Age: 56 | Gender: Female",
        "Stroke symptoms: slurred speech and facial drooping | Age: 70 | Gender: Female",
        "High fever and severe ear pain | Age: 8 | Gender: Female",
        "Profuse nosebleed that won't stop | Age: 28 | Gender: Male",
        "Persistent abdominal pain for 3 days | Age: 50 | Gender: Male",
        "Third-degree burns on hand from boiling water | Age: 35 | Gender: Female",
        "Mild headache after hitting head on a door | Age: 29 | Gender: Male",
        "Severe abdominal pain and vomiting during pregnancy | Age: 28 | Gender: Female (pregnant)",
        "Child with a sore throat and mild fever | Age: 4 | Gender: Male",
        "Persistent weight loss and chronic cough | Age: 65 | Gender: Male",
        "Trauma from a motorcycle accident | Age: 40 | Gender: Male",
        "Pain and swelling in the knee after a fall | Age: 32 | Gender: Female",
        "Severe difficulty breathing after eating peanuts | Age: 16 | Gender: Male",
        "Child with persistent high fever and lethargy | Age: 3 | Gender: Female",
        "Sudden severe chest pain and profuse sweating | Age: 55 | Gender: Male",
        "Mild rash on arms, no other symptoms | Age: 20 | Gender: Female",
        "Broken arm with deformity after a fall | Age: 12 | Gender: Male",
        "Child vomiting repeatedly after a head injury | Age: 9 | Gender: Female",
        "Swollen eye and blurry vision after a chemical splash | Age: 45 | Gender: Male",
        "Persistent cough with blood for a week | Age: 55 | Gender: Male",
        "Sudden loss of consciousness during exercise | Age: 24 | Gender: Male",
        "Pregnant woman with severe bleeding | Age: 30 | Gender: Female (pregnant)",
        "Child with wheezing and severe difficulty breathing | Age: 6 | Gender: Male",
        "Severe allergic reaction with hives and swelling | Age: 22 | Gender: Female",
        "Painful urination and blood in urine | Age: 33 | Gender: Female",
        "Fainting after feeling lightheaded | Age: 21 | Gender: Male",
        "Profuse sweating and nausea after climbing stairs | Age: 52 | Gender: Female",
        "Severe chest pain radiating to the left arm | Age: 59 | Gender: Male",
        "Persistent ear pain with discharge | Age: 27 | Gender: Female",
        "Mild sore throat and cough | Age: 18 | Gender: Male",
        "Burns on legs from a house fire | Age: 45 | Gender: Male",
        "Child choking on a small toy, unable to breathe | Age: 2 | Gender: Female",
        "Fractured wrist after a car accident | Age: 40 | Gender: Female",
        "Severe abdominal pain and dizziness | Age: 48 | Gender: Male",
        "Slurred speech and numbness in one arm | Age: 64 | Gender: Male",
        "Pain and redness in the eye after rubbing it | Age: 29 | Gender: Female",
        "Mild fever and cough for a week | Age: 60 | Gender: Male",
        "Child with diarrhea and vomiting for 2 days | Age: 5 | Gender: Female",
        "Severe headache and vomiting after a head injury | Age: 37 | Gender: Male",
        "Sudden onset of confusion and inability to speak | Age: 74 | Gender: Female",
        "Broken leg with open wound after falling | Age: 39 | Gender: Female",
        "Shortness of breath and chest tightness | Age: 50 | Gender: Male",
        "Persistent dry cough for over a month | Age: 62 | Gender: Male",
        "Severe bleeding from a scalp wound | Age: 41 | Gender: Male",
        "Mild bruising on the knee after hitting a table | Age: 23 | Gender: Female",
        "Sharp abdominal pain and bloating | Age: 54 | Gender: Male",
        "Child with a persistent ear infection and fever | Age: 7 | Gender: Male",
        "Profuse bleeding after being hit by a falling object | Age: 38 | Gender: Male",
        "Sudden vision loss and severe headache | Age: 58 | Gender: Female",
        "Pregnant woman with contractions every 5 minutes | Age: 30 | Gender: Female (pregnant)",
        "Severe abdominal pain and fever after surgery | Age: 50 | Gender: Male",
        "Shortness of breath after being stung by multiple bees | Age: 43 | Gender: Male",
        "Child unable to keep fluids down due to vomiting | Age: 4 | Gender: Male",
        "Severe leg pain and inability to walk after a fall | Age: 55 | Gender: Female",
        "Mild rash with no other symptoms | Age: 31 | Gender: Female",
        "High fever, chills, and body aches | Age: 37 | Gender: Male",
        "Severe chest pain and fainting | Age: 62 | Gender: Male",
        "Child with swollen tonsils and high fever | Age: 6 | Gender: Female",
        "Severe allergic reaction with throat swelling | Age: 28 | Gender: Female",
        "Persistent fatigue and breathlessness | Age: 49 | Gender: Male",
        "Coughing blood and weight loss | Age: 65 | Gender: Male",
        "Mild sprain after tripping on stairs | Age: 19 | Gender: Female",
        "Child with a fever and dehydration | Age: 3 | Gender: Female",
        "Profuse bleeding from a dog bite | Age: 33 | Gender: Male",
        "Severe back pain after lifting heavy objects | Age: 42 | Gender: Male",
        "Sudden loss of vision in one eye | Age: 67 | Gender: Male",
        "Child crying and holding ear with fever | Age: 8 | Gender: Male",
        "Severe abdominal pain and pale skin | Age: 60 | Gender: Female",
        "Shortness of breath and swollen lips after medication | Age: 46 | Gender: Female",
        "Child with a swollen face after eating peanuts | Age: 3 | Gender: Male",
        "Severe chest pain after exercising | Age: 45 | Gender: Male",
        "Pregnant woman experiencing contractions and severe bleeding | Age: 32 | Gender: Female (pregnant)",
        "Persistent earache with fever and dizziness | Age: 21 | Gender: Female",
        "Child crying and refusing to move arm after a fall | Age: 5 | Gender: Male",
        "High fever and chills after a recent trip abroad | Age: 40 | Gender: Female",
        "Severe abdominal pain and black stools | Age: 53 | Gender: Male",
        "Shortness of breath while climbing stairs | Age: 60 | Gender: Male",
        "Child choking and unable to speak | Age: 4 | Gender: Female",
        "Sudden loss of balance and inability to walk | Age: 68 | Gender: Male",
        "Sudden sharp chest pain and sweating | Age: 58 | Gender: Female",
        "Persistent vomiting and inability to keep food down | Age: 34 | Gender: Male"     
    ]

    logits, predicted_esi_levels = predict_with_logic(large_test_cases)
    results = {
        "predictions": []
    }

    for i, test in enumerate(large_test_cases):
        adjusted_esi_level = apply_post_processing(test, predicted_esi_levels[i], logits[i])
        results["predictions"].append({
            "input_text": test,
            "prediction": {
                "original_predicted_esi": predicted_esi_levels[i],
                "adjusted_esi": adjusted_esi_level
            }
        })
    return results

if __name__ == "__main__":
    output = evaluate_cases()
    print(output)
