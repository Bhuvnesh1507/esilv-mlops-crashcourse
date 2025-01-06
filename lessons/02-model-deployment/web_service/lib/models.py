from pydantic import BaseModel, Field
 
 
class WineData(BaseModel):
    type: str = Field(..., description="Type of wine (red or white)")
    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    pH: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol content")
 
    class Config:
        schema_extra = {
            "example": {
                "type": "white",
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45.0,
                "total_sulfur_dioxide": 170.0,
                "density": 1.001,
                "pH": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8
            }
        }
 
 
class PredictionOut(BaseModel):
    quality_prediction: float = Field(..., description="Predicted wine quality score (1-10)")