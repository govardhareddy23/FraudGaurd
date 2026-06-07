from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import get_db, Alert
from models.schemas import AlertRecord

router = APIRouter()


@router.get("/", response_model=List[AlertRecord])
def list_alerts(status: str = "open", db: Session = Depends(get_db)):
    return (
        db.query(Alert)
        .filter(Alert.status == status)
        .order_by(Alert.timestamp.desc())
        .limit(100)
        .all()
    )


@router.patch("/{alert_id}")
def update_alert(alert_id: int, status: str, notes: str = None, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = status
    if notes:
        alert.notes = notes
    db.commit()
    return {"ok": True}
