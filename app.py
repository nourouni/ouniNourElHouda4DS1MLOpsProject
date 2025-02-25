from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a pre-trained Decision Tree model",
    version="1.0.0",
    contact={
        "name": "Nour El Houda Ouni",
        "email": "nour@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, scaler, and training columns
model, scaler = joblib.load("model.pkl")
prepared_data = pd.read_pickle("prepared_data.pkl")
expected_features = prepared_data.drop(columns=['Churn']).columns.tolist()

class CustomerInput(BaseModel):
    State: str = Field(..., 
                     alias="State",
                     description="US State abbreviation",
                     example="LA")
    
    Account_length: int = Field(...,
                              alias="Account length",
                              ge=0,
                              description="Number of months the customer has been with the company",
                              example=117)
    
    Area_code: int = Field(...,
                         alias="Area code",
                         description="Area code number",
                         example=408)
    
    International_plan: str = Field(...,
                                  alias="International plan",
                                  description="Whether the customer has an international plan",
                                  example="no",
                                  pattern="^(yes|no)$")
    
    Voice_mail_plan: str = Field(...,
                               alias="Voice mail plan",
                               description="Whether the customer has a voice mail plan",
                               example="no",
                               pattern="^(yes|no)$")
    
    Number_vmail_messages: int = Field(...,
                                     alias="Number vmail messages",
                                     ge=0,
                                     description="Number of voice mail messages",
                                     example=0)
    
    Total_day_minutes: float = Field(...,
                                   alias="Total day minutes",
                                   ge=0,
                                   description="Total minutes of day calls",
                                   example=184.5)
    
    Total_day_calls: int = Field(...,
                               alias="Total day calls",
                               ge=0,
                               description="Total number of day calls",
                               example=97)
    
    Total_day_charge: float = Field(...,
                                   alias="Total day charge",
                                   ge=0,
                                   description="Total charge for day calls",
                                   example=31.37)
    
    Total_eve_minutes: float = Field(...,
                                   alias="Total eve minutes",
                                   ge=0,
                                   description="Total minutes of evening calls",
                                   example=351.6)
    
    Total_eve_calls: int = Field(...,
                               alias="Total eve calls",
                               ge=0,
                               description="Total number of evening calls",
                               example=80)
    
    Total_eve_charge: float = Field(...,
                                   alias="Total eve charge",
                                   ge=0,
                                   description="Total charge for evening calls",
                                   example=29.89)
    
    Total_night_minutes: float = Field(...,
                                     alias="Total night minutes",
                                     ge=0,
                                     description="Total minutes of night calls",
                                     example=215.8)
    
    Total_night_calls: int = Field(...,
                                 alias="Total night calls",
                                 ge=0,
                                 description="Total number of night calls",
                                 example=90)
    
    Total_night_charge: float = Field(...,
                                     alias="Total night charge",
                                     ge=0,
                                     description="Total charge for night calls",
                                     example=9.71)
    
    Total_intl_minutes: float = Field(...,
                                    alias="Total intl minutes",
                                    ge=0,
                                    description="Total minutes of international calls",
                                    example=8.7)
    
    Total_intl_calls: int = Field(...,
                                alias="Total intl calls",
                                ge=0,
                                description="Total number of international calls",
                                example=4)
    
    Total_intl_charge: float = Field(...,
                                    alias="Total intl charge",
                                    ge=0,
                                    description="Total charge for international calls",
                                    example=2.35)
    
    Customer_service_calls: int = Field(...,
                                       alias="Customer service calls",
                                       ge=0,
                                       description="Number of calls to customer service",
                                       example=1)

@app.post("/predict/",
         summary="Predict Customer Churn",
         description="Predicts whether a customer is likely to churn based on their service usage patterns",
         response_description="Churn Prediction Result",
         tags=["Predictions"])
async def predict(data: CustomerInput):
    """
    Make a churn prediction for a customer based on their service usage data.
    
    Requires all customer service usage metrics and account information.
    Returns a boolean prediction (True = likely to churn, False = likely to stay).
    
    **Example Request Body:**
    ```json
    {
      "State": "LA",
      "Account length": 117,
      "Area code": 408,
      "International plan": "no",
      "Voice mail plan": "no",
      "Number vmail messages": 0,
      "Total day minutes": 184.5,
      "Total day calls": 97,
      "Total day charge": 31.37,
      "Total eve minutes": 351.6,
      "Total eve calls": 80,
      "Total eve charge": 29.89,
      "Total night minutes": 215.8,
      "Total night calls": 90,
      "Total night charge": 9.71,
      "Total intl minutes": 8.7,
      "Total intl calls": 4,
      "Total intl charge": 2.35,
      "Customer service calls": 1
    }
    ```
    """
    try:
        input_dict = data.dict(by_alias=True)
        df = pd.DataFrame([input_dict])
        
        # Manual encoding
        df['International plan_Yes'] = df['International plan'].map({'yes': 1, 'no': 0})
        df['Voice mail plan_Yes'] = df['Voice mail plan'].map({'yes': 1, 'no': 0})
        
        # One-hot encode State
        df = pd.get_dummies(df, columns=['State'], prefix='State')
        
        # Reindex to match training features
        df = df.reindex(columns=expected_features, fill_value=0)
        
        # Scale numerical features
        num_cols = [col for col in expected_features if col not in ['International plan_Yes', 'Voice mail plan_Yes'] 
                   and not col.startswith('State_')]
        df[num_cols] = scaler.transform(df[num_cols])
        
        prediction = model.predict(df)
        return {"churn_prediction": bool(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Customer Churn Prediction API - Visit /docs for interactive documentation"}
