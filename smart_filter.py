import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# NEW: imports for API
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------
# 1. LOAD & TRAIN MODEL
# ----------------------

orders = pd.read_excel("GeneratedOrderData_updated.xlsx")
restaurants = pd.read_excel("Generated_RestaurantData.xlsx")

print("Orders shape:", orders.shape)
print("Restaurants shape:", restaurants.shape)

df = pd.merge(orders, restaurants, on="RestaurantID", how="left")

print("\nMerged Data Sample:")
print(df.head())

# Columns that we will label-encode
categorical_cols = [
    "RestaurantID", "WeatherType", "SeasonType", "MealType", "Day",
    "TypeRestaurant", "CuisineType", "LocationType", "State/City"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features and target
X = df[["RestaurantID", "WeatherType", "SeasonType", "MealType", "Day"]]
y = df["ItemID"]

# Encode target
item_encoder = LabelEncoder()
y_encoded = item_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate once at startup
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# ----------------------
# 2. RECOMMEND FUNCTION
# ----------------------

def recommend_items(restaurant_id, weather, season, meal, day):
    input_df = pd.DataFrame({
        "RestaurantID": [restaurant_id],
        "WeatherType": [weather],
        "SeasonType": [season],
        "MealType": [meal],
        "Day": [day]
    })

    # Encode using fitted encoders
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

    # Predict probabilities
    probs = model.predict_proba(input_df)[0]

    # Top 5 item indices by probability
    top_indices = np.argsort(probs)[-5:][::-1]

    # Convert back to original ItemID values
    top_items = item_encoder.inverse_transform(top_indices)

    return top_items

# ----------------------
# 3. FASTAPI APP (API LAYER)
# ----------------------

app = FastAPI()

# This defines what JSON we expect from frontend / Node
class RecommendRequest(BaseModel):
    restaurant_id: str
    weather: str
    season: str
    meal: str
    day: str

# This is the API endpoint: POST /recommend
@app.post("/recommend")
def recommend(req: RecommendRequest):
    items = recommend_items(
        restaurant_id=req.restaurant_id,
        weather=req.weather,
        season=req.season,
        meal=req.meal,
        day=req.day
    )

    # Convert numpy.int64 values → normal Python int
    items_list = [int(x) for x in items]

    return {
        "recommended_items": items_list
    }