import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path("Models\Advanced\Stacking_Classifier_best_model.joblib")

app = FastAPI(
    title=" Hotel Booking Cancelled Prediction API",
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
from pydantic import BaseModel
from typing import Optional

class HotelBookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: Optional[int] = 0
    company: Optional[int] = 0
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str
    reservation_status_date: str
    city: str


class PredictionOutput(BaseModel):
    is_canceled: int
    cancellation_probability: float



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
def predict(data: HotelBookingInput):
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    # Input → DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Prediction
    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]

    return PredictionOutput(
        is_canceled=int(pred),
        cancellation_probability=round(float(proba), 4)
    )