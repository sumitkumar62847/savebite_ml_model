import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load data
orders = pd.read_excel("GeneratedOrderData_updated.xlsx")
restaurants = pd.read_excel("Generated_RestaurantData.xlsx")

df = pd.merge(orders, restaurants, on="RestaurantID", how="left")

categorical_cols = [
    "RestaurantID", "WeatherType", "SeasonType", "MealType", "Day",
    "TypeRestaurant", "CuisineType", "LocationType", "State/City"
]

# 2. Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 3. Features and target
X = df[["RestaurantID", "WeatherType", "SeasonType", "MealType", "Day"]]
y = df["ItemID"]

item_encoder = LabelEncoder()
y_encoded = item_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# model = RandomForestClassifier(n_estimators=200, random_state=42)

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=20,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate once
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# 5. Save everything needed for prediction
artifacts = {
    "model": model,
    "encoders": encoders,
    "item_encoder": item_encoder,
    "categorical_cols": categorical_cols,
}

joblib.dump(artifacts, "artifacts.joblib")
print("Saved artifacts to artifacts.joblib")