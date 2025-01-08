from fastapi import FastAPI
from ai_inference_logic import predict_with_logic

app = FastAPI()

@app.get("/")
def read_root():
    return {"message":
