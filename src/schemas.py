from pydantic import BaseModel

class AdvertisingInput(BaseModel):
    TV: float
    Radio: float
    Newspaper : float

class PredictionOutput(BaseModel):
    sales_prediction: float