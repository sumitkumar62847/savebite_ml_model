import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load trained objects
artifacts = joblib.load("artifacts.joblib")
model = artifacts["model"]
encoders = artifacts["encoders"]
item_encoder = artifacts["item_encoder"]

app = FastAPI()

class RecommendRequest(BaseModel):
    restaurant_id: str
    weather: str
    season: str
    meal: str
    day: str

def recommend_items(restaurant_id, weather, season, meal, day):
    # Create input dataframe
    input_df = pd.DataFrame({
        "RestaurantID": [restaurant_id],
        "WeatherType": [weather],
        "SeasonType": [season],
        "MealType": [meal],
        "Day": [day]
    })

    # Encode with fitted encoders
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

    # Predict probabilities
    probs = model.predict_proba(input_df)[0]

    # Top 5 item indices
    top_indices = np.argsort(probs)[-5:][::-1]
    top_items = item_encoder.inverse_transform(top_indices)

    return [int(x) for x in top_items]

@app.post("/recommend")
def recommend(req: RecommendRequest):
    items = recommend_items(
        restaurant_id=req.restaurant_id,
        weather=req.weather,
        season=req.season,
        meal=req.meal,
        day=req.day,
    )
    return {"recommended_items": items}