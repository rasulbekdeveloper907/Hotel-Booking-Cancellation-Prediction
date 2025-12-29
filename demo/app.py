import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = "Models\Advanced\Stacking_Classifier_best_model.joblib"
pipeline = joblib.load(MODEL_PATH)

def predict(
    games, time, xG, assists, xA, shots, key_passes,
    yellow_cards, red_cards, position, team_title,
    npg, npxG, xGChain, xGBuildup, league, season
):
    df = pd.DataFrame([{
        "games": games,
        "time": time,
        "xG": xG,
        "assists": assists,
        "xA": xA,
        "shots": shots,
        "key_passes": key_passes,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
        "position": position,
        "team_title": team_title,
        "npg": npg,
        "npxG": npxG,
        "xGChain": xGChain,
        "xGBuildup": xGBuildup,
        "league": league,
        "season": season
    }])

    pred = pipeline.predict(df)[0]
    return round(float(pred), 4)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Games"),
        gr.Number(label="Minutes Played"),
        gr.Number(label="xG"),
        gr.Number(label="Assists"),
        gr.Number(label="xA"),
        gr.Number(label="Shots"),
        gr.Number(label="Key Passes"),
        gr.Number(label="Yellow Cards"),
        gr.Number(label="Red Cards"),
        gr.Textbox(label="Position (e.g. F S)"),
        gr.Textbox(label="Team Title"),
        gr.Number(label="Non-penalty Goals"),
        gr.Number(label="Non-penalty xG"),
        gr.Number(label="xGChain"),
        gr.Number(label="xGBuildup"),
        gr.Textbox(label="League (e.g. ASerie)"),
        gr.Number(label="Season")
    ],
    outputs=gr.Number(label="Predicted Booking Cancelled (t+1)"),
    title=" Hotels Booking Cancelled Prediction",
    description="Best Stacking Classifier Pipeline"
)

if __name__ == "__main__":
    demo.launch()