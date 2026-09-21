from pydantic import BaseModel, Field, StrictStr, field_validator


class PredictionRequest(BaseModel):
    text: StrictStr = Field(..., min_length=1, max_length=600)


class BatchPredictionRequest(BaseModel):
    text: list[StrictStr] = Field(..., min_length=1, max_length=600)

    @field_validator("text")
    @classmethod
    def validate_texts(cls, texts):
        for text in texts:
            if not 1 <= len(text) <= 600:
                raise ValueError("Each complaint must contain 1 to 600 characters")
        return texts


class PredictionResponse(BaseModel):
    category: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]