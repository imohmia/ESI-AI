from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os

app = FastAPI()

# Set the directory where the model files are stored
model_dir = os.path.dirname(__file__)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

class PredictionInput(BaseModel):
    input_text: str
    age: int

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Predicts the label for the given input text and age.
    """
    input_text_with_age = f"Chief Complaint: {data.input_text} Age: {data.age}"
    inputs = tokenizer(input_text_with_age, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    return {"predicted_class": predicted_class}
