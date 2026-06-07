from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import get_db, Transaction, Alert
from models.schemas import DashboardStats

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
def get_stats(db: Session = Depends(get_db)):
    total  = db.query(func.count(Transaction.id)).scalar() or 0
    frauds = db.query(func.count(Transaction.id)).filter(Transaction.is_fraud == True).scalar() or 0
    avg_amt = db.query(func.avg(Transaction.amount)).scalar() or 0.0
    avg_fraud_amt = (
        db.query(func.avg(Transaction.amount))
        .filter(Transaction.is_fraud == True)
        .scalar() or 0.0
    )
    open_alerts = db.query(func.count(Alert.id)).filter(Alert.status == "open").scalar() or 0

    today = date.today()
    txn_today   = db.query(func.count(Transaction.id)).filter(
        func.date(Transaction.timestamp) == today).scalar() or 0
    fraud_today = db.query(func.count(Transaction.id)).filter(
        Transaction.is_fraud == True,
        func.date(Transaction.timestamp) == today).scalar() or 0

    return DashboardStats(
        total_transactions=total,
        fraud_transactions=frauds,
        fraud_rate=round(frauds / total * 100, 4) if total else 0.0,
        avg_amount=round(float(avg_amt), 2),
        avg_fraud_amount=round(float(avg_fraud_amt), 2),
        open_alerts=open_alerts,
        transactions_today=txn_today,
        fraud_today=fraud_today,
    )


@router.get("/chart/hourly")
def hourly_chart(db: Session = Depends(get_db)):
    rows = (
        db.query(
            func.extract("hour", Transaction.timestamp).label("hour"),
            func.count(Transaction.id).label("total"),
            func.sum(func.cast(Transaction.is_fraud, db.bind.dialect.name == "postgresql"
                               and "integer" or "integer")).label("fraud"),
        )
        .group_by("hour")
        .order_by("hour")
        .all()
    )
    return [{"hour": int(r.hour), "total": r.total, "fraud": int(r.fraud or 0)} for r in rows]
