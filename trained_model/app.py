import requests
import os

# Google Drive file URL (the one you shared)
file_id = "1au-vR0Qwd1bIIDRD1DzOO7e7Gk1Fa1t0"  # Replace this with the actual file ID
url = f"https://drive.google.com/uc?id={file_id}"

# Path to save the file on your server
model_path = "trained_model/model.safetensors"  # Ensure this is the path where you want to save the file

# Check if the model file already exists, if not, download it
if not os.path.exists(model_path):
    print("Downloading the model file from Google Drive...")
    response = requests.get(url)
    with open(model_path, "wb") as file:
        file.write(response.content)
    print("Model file downloaded successfully!")
else:
    print("Model file already exists.")

# Now, load the model (if it's downloaded or already present)
from transformers import BertForSequenceClassification, AutoTokenizer
import torch

# Define the path to the tokenizer and model
tokenizer_path = "trained_model"  # Adjust to the correct path

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Set the model to evaluation mode and move it to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")
