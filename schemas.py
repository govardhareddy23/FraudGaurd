from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TransactionInput(BaseModel):
    time:   float = Field(default=0.0,   description="Seconds since first transaction")
    amount: float = Field(...,           description="Transaction amount in USD", gt=0)
    v1:  float = 0.0; v2:  float = 0.0; v3:  float = 0.0; v4:  float = 0.0
    v5:  float = 0.0; v6:  float = 0.0; v7:  float = 0.0; v8:  float = 0.0
    v9:  float = 0.0; v10: float = 0.0; v11: float = 0.0; v12: float = 0.0
    v13: float = 0.0; v14: float = 0.0; v15: float = 0.0; v16: float = 0.0
    v17: float = 0.0; v18: float = 0.0; v19: float = 0.0; v20: float = 0.0
    v21: float = 0.0; v22: float = 0.0; v23: float = 0.0; v24: float = 0.0
    v25: float = 0.0; v26: float = 0.0; v27: float = 0.0; v28: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "time": 406, "amount": 149.62,
                "v1": -1.359807, "v2": -0.072781, "v3": 2.536347,
                "v4": 1.378155, "v14": -0.311169, "v17": -0.587759
            }
        }


class PredictionResponse(BaseModel):
    transaction_id:    int
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str
    alert_triggered:   bool
    amount:            float
    timestamp:         datetime


class TransactionRecord(BaseModel):
    id:                int
    timestamp:         datetime
    amount:            float
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str
    alert_triggered:   bool

    class Config:
        from_attributes = True


class AlertRecord(BaseModel):
    id:               int
    transaction_id:   int
    timestamp:        datetime
    fraud_probability: float
    amount:           float
    status:           str
    notes:            Optional[str] = None

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate:         float
    avg_amount:         float
    avg_fraud_amount:   float
    open_alerts:        int
    transactions_today: int
    fraud_today:        int
