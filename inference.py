from fastapi import FastAPI
from ai_inference_logic import predict

app = FastAPI()

@app.post("/predict")
def get_prediction(data: dict):
    input_text = data["input_text"]
    prediction = predict(input_text)
    return {"prediction": prediction}
