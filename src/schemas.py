from pydantic import BaseModel, Field

class AdvertisingInput(BaseModel):
    tv: float = Field(..., ge=0, description="TV (1000$)")
    radio: float = Field(..., ge=0, description="Radio (1000$)")
    newspaper : float = Field(..., ge=0, description="Newspaper (1000$)")

class PredictionOutput(BaseModel):
    sales_prediction: float