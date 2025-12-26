# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# Load trained PIPELINE
# --------------------------------------------------
MODEL_PATH = "models/baseline/best_baseline_model.joblib"
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Soccer Goals Prediction API",
    version="1.0"
)

# --------------------------------------------------
# Input schema (MATCHES TRAINING DATA)
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

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}



# --------------------------------------------------
# Prediction
# --------------------------------------------------
@app.post("/predict")
def predict(data: PlayerInput):

    df = pd.DataFrame([data.dict()])
    pred = pipeline.predict(df)[0]

    return {
        "predicted_goals_t_plus_1": round(float(pred), 4)
    }
