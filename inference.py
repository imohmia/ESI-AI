from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ai_inference_logic import predict_with_logic
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Define input data structure
class PredictRequest(BaseModel):
    texts: List[str] = Field(..., example=["Input text 1", "Input text 2"])

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # Validate input
        if not data.texts:
            raise HTTPException(status_code=400, detail="No texts provided for prediction.")
        
        # Run predictions
        predictions = predict_with_logic(data.texts)

        # Structure the response
        response = {
            "predictions": [
                {
                    "input_text": text,
                    "prediction": pred
                }
                for text, pred in zip(data.texts, predictions)
            ]
        }
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/")
def read_root():
    return {"message": "Server is running. Use the /predict endpoint for predictions."}

@app.get("/favicon.ico")
def favicon():
    return {"message": "Favicon not implemented"}
