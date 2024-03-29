from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
# Load the trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the request body model
class HouseInput(BaseModel):
    region: str
    province: str
    locality: str
    zip_code: int
    latitude: float
    longitude: float
    construction_year: int
    total_area_sqm: float
    surface_land_sqm: float
    nbr_frontages: int
    nbr_bedrooms: int
    equipped_kitchen: str
    terrace_sqm: float
    garden_sqm: float
    state_building: str
    primary_energy_consumption_sqm: float
    epc: str
    heating_type: str
    cadastral_income: float

# Define the prediction endpoint
@app.post("/predict")
async def predict_house_price(input_data: HouseInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Make prediction
    prediction = model.predict(input_df)

    # Return prediction
    return {"predicted_price": prediction[0]}

# Run the FastAPI app
if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

''' sample test data {
  "region": "Brussels",
  "province": "Brussels",
  "locality": "Brussels",
  "zip_code": 1000,
  "latitude": 50.8503,
  "longitude": 4.3517,
  "construction_year": 1970,
  "total_area_sqm": 120,
  "surface_land_sqm": 150,
  "nbr_frontages": 3,
  "nbr_bedrooms": 2,
  "equipped_kitchen": "SEMI_EQUIPPED",
  "terrace_sqm": 10,
  "garden_sqm": 30,
  "state_building": "Renovated",
  "primary_energy_consumption_sqm": 180,
  "epc": "D",
  "heating_type": "Electric",
  "cadastral_income": 1200
}
'''
