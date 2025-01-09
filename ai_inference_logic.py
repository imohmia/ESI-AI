import os
import requests

# Model download details
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

# Run the download function
download_model()

# Placeholder for deployment
print("Deployment setup successful. Model downloading and setup will be done in subsequent steps.")
