from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_inference_logic import predict_with_logic

app = FastAPI()

class PredictRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        predictions = predict_with_logic(data.texts)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
