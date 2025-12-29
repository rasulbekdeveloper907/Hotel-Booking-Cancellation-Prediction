import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path("models/baseline/best_baseline_model.joblib")

app = FastAPI(
    title="Soccer Goals Prediction API",
    version="1.0"
)

pipeline = None  


# --------------------------------------------------
# Load model ON STARTUP (NOT import time)
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class PlayerInput(BaseModel):
    games: int
    time: int
    xG: float
    assists: int
    xA: float
    shots: int
    key_passes: int
    yellow_cards: int
    red_cards: int
    position: str
    team_title: str
    npg: int
    npxG: float
    xGChain: float
    xGBuildup: float
    league: str
    season: int


class PredictionOutput(BaseModel):
    predicted_goals_t_plus_1: float


# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Predict
# --------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PlayerInput):
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([data.model_dump()])
    pred = pipeline.predict(df)[0]

    return PredictionOutput(
        predicted_goals_t_plus_1=round(float(pred), 4)
    )