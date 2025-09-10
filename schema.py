from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Transaction(BaseModel):
    amount: float = Field(..., ge=0)
    merchant_category: Literal["grocery","ecommerce","fuel","travel","electronics","restaurant"]
    channel: Literal["pos","web","mobile","atm"]
    country: Literal["CA","US","MX","GB","FR","IN"]
    customer_age: int = Field(..., ge=18, le=120)
    customer_tenure_months: int = Field(..., ge=0, le=600)
    is_night: int = Field(..., ge=0, le=1)

    @field_validator("is_night")
    @classmethod
    def _int01(cls, v):
        if v not in (0,1):
            raise ValueError("is_night must be 0 or 1")
        return v
