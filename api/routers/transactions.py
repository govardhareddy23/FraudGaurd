from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.database import get_db, Transaction
from api.models.schemas import TransactionRecord

router = APIRouter()


@router.get("/", response_model=List[TransactionRecord])
def list_transactions(
    limit: int = Query(50, le=200),
    fraud_only: bool = False,
    db: Session = Depends(get_db),
):
    q = db.query(Transaction)
    if fraud_only:
        q = q.filter(Transaction.is_fraud == True)
    return q.order_by(Transaction.timestamp.desc()).limit(limit).all()


@router.get("/{tx_id}", response_model=TransactionRecord)
def get_transaction(tx_id: int, db: Session = Depends(get_db)):
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(404, "Transaction not found")
    return tx
