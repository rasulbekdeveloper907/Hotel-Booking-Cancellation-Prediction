import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = r"../Models\Advanced\Stacking_Classifier_best_model.joblib"
pipeline = joblib.load(MODEL_PATH)

def predict(     hotel,     lead_time,     arrival_date_year,     arrival_date_month,    arrival_date_week_number,    arrival_date_day_of_month,    stays_in_weekend_nights,
    stays_in_week_nights,    adults,    children,    babies,    meal,    country,    market_segment,    distribution_channel,    is_repeated_guest,    previous_cancellations,
    previous_bookings_not_canceled,     reserved_room_type,    assigned_room_type,    booking_changes,    deposit_type,    agent,    company,    days_in_waiting_list,    customer_type,
    adr,     required_car_parking_spaces,    total_of_special_requests,    reservation_status,    reservation_status_date,    city
):
    df = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "reservation_status": reservation_status,
        "reservation_status_date": reservation_status_date,
        "city": city
    }])

    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Hotel"),
        gr.Number(label="Lead Time"),
        gr.Number(label="Arrival Year"),
        gr.Textbox(label="Arrival Month (e.g. July)"),
        gr.Number(label="Arrival Week Number"),
        gr.Number(label="Arrival Day of Month"),
        gr.Number(label="Stays in Weekend Nights"),
        gr.Number(label="Stays in Week Nights"),
        gr.Number(label="Adults"),
        gr.Number(label="Children"),
        gr.Number(label="Babies"),
        gr.Textbox(label="Meal"),
        gr.Textbox(label="Country"),
        gr.Textbox(label="Market Segment"),
        gr.Textbox(label="Distribution Channel"),
        gr.Number(label="Is Repeated Guest (0 or 1)"),
        gr.Number(label="Previous Cancellations"),
        gr.Number(label="Previous Bookings Not Canceled"),
        gr.Textbox(label="Reserved Room Type"),
        gr.Textbox(label="Assigned Room Type"),
        gr.Number(label="Booking Changes"),
        gr.Textbox(label="Deposit Type"),
        gr.Number(label="Agent"),
        gr.Number(label="Company"),
        gr.Number(label="Days in Waiting List"),
        gr.Textbox(label="Customer Type"),
        gr.Number(label="ADR"),
        gr.Number(label="Required Car Parking Spaces"),
        gr.Number(label="Total of Special Requests"),
        gr.Textbox(label="Reservation Status"),
        gr.Textbox(label="Reservation Status Date"),
        gr.Textbox(label="City")
    ],
    outputs="json",
    title="Hotel Booking Cancellation Prediction",
    description="Enter booking details to predict whether the reservation will be canceled"
)

if __name__ == "__main__":
    demo.launch()