from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import get_db, Transaction, Alert
from models.schemas import TransactionInput, PredictionResponse
from ml.predictor import predict_fraud, load_model

router = APIRouter()

@router.on_event("startup") if False else None  # handled in lifespan


@router.post("/", response_model=PredictionResponse)
def predict(tx: TransactionInput, db: Session = Depends(get_db)):
    feature_dict = tx.model_dump()

    result = predict_fraud(feature_dict)

    record = Transaction(
        amount=tx.amount, time_val=tx.time,
        v1=tx.v1,   v2=tx.v2,   v3=tx.v3,   v4=tx.v4,
        v5=tx.v5,   v6=tx.v6,   v7=tx.v7,   v8=tx.v8,
        v9=tx.v9,   v10=tx.v10, v11=tx.v11, v12=tx.v12,
        v13=tx.v13, v14=tx.v14, v15=tx.v15, v16=tx.v16,
        v17=tx.v17, v18=tx.v18, v19=tx.v19, v20=tx.v20,
        v21=tx.v21, v22=tx.v22, v23=tx.v23, v24=tx.v24,
        v25=tx.v25, v26=tx.v26, v27=tx.v27, v28=tx.v28,
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        alert_triggered=result["alert_triggered"],
    )
    db.add(record)
    db.flush()

    if result["alert_triggered"]:
        alert = Alert(
            transaction_id=record.id,
            fraud_probability=result["fraud_probability"],
            amount=tx.amount,
        )
        db.add(alert)

    db.commit()
    db.refresh(record)

    return PredictionResponse(
        transaction_id=record.id,
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        alert_triggered=result["alert_triggered"],
        amount=tx.amount,
        timestamp=record.timestamp,
    )
