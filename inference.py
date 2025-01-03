from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import sys

# Set the directory where the model files are stored
model_dir = os.path.dirname(__file__)  # Directory of the current script
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def predict(input_text: str, age: int) -> int:
    """
    Predicts the label for the given input text and age.

    Args:
        input_text (str): The chief complaint or input text.
        age (int): The age of the patient.

    Returns:
        int: The predicted class label.
    """
    # Combine input text and age into a single string
    input_text_with_age = f"Chief Complaint: {input_text} Age: {age}"

    # Tokenize the input
    inputs = tokenizer(input_text_with_age, return_tensors="pt")

    # Generate predictions
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()

    return predicted_class

if __name__ == "__main__":
    # Check if the script was called with the required arguments
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_text> <age>")
        sys.exit(1)

    # Parse input text and age from command-line arguments
    input_text = sys.argv[1]
    age = int(sys.argv[2])

    # Run prediction
    predicted_label = predict(input_text, age)

    # Print the predicted label
    print(predicted_label)
