from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"greeting": "hello"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    
    X_pred = pd.DataFrame({
        'key':["value"],
        'pickup_datetime':[pickup_datetime],
        'pickup_longitude':[float(pickup_longitude)],
        'pickup_latitude':[float(pickup_latitude)],
        'dropoff_longitude':[float(dropoff_longitude)],
        'dropoff_latitude':[float(dropoff_latitude)],
        'passenger_count':[int(passenger_count)],
    })
    
    pipeline = joblib.load('model.joblib')
    
    res = pipeline.predict(X_pred)[0]
    
    return {'fare':res}
