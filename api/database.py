from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/fraud_detection"
)

# Use pool_pre_ping to check connection health automatically
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id            = Column(Integer, primary_key=True, index=True)
    timestamp     = Column(DateTime, default=datetime.utcnow)
    amount        = Column(Float, nullable=False)
    time_val      = Column(Float, default=0.0)

    # PCA features
    v1  = Column(Float, default=0.0); v2  = Column(Float, default=0.0)
    v3  = Column(Float, default=0.0); v4  = Column(Float, default=0.0)
    v5  = Column(Float, default=0.0); v6  = Column(Float, default=0.0)
    v7  = Column(Float, default=0.0); v8  = Column(Float, default=0.0)
    v9  = Column(Float, default=0.0); v10 = Column(Float, default=0.0)
    v11 = Column(Float, default=0.0); v12 = Column(Float, default=0.0)
    v13 = Column(Float, default=0.0); v14 = Column(Float, default=0.0)
    v15 = Column(Float, default=0.0); v16 = Column(Float, default=0.0)
    v17 = Column(Float, default=0.0); v18 = Column(Float, default=0.0)
    v19 = Column(Float, default=0.0); v20 = Column(Float, default=0.0)
    v21 = Column(Float, default=0.0); v22 = Column(Float, default=0.0)
    v23 = Column(Float, default=0.0); v24 = Column(Float, default=0.0)
    v25 = Column(Float, default=0.0); v26 = Column(Float, default=0.0)
    v27 = Column(Float, default=0.0); v28 = Column(Float, default=0.0)

    fraud_probability = Column(Float, nullable=False)
    is_fraud          = Column(Boolean, nullable=False)
    risk_level        = Column(String(10), nullable=False)
    alert_triggered   = Column(Boolean, default=False)


class Alert(Base):
    __tablename__ = "alerts"

    id                 = Column(Integer, primary_key=True, index=True)
    transaction_id     = Column(Integer, nullable=False)
    timestamp          = Column(DateTime, default=datetime.utcnow)
    fraud_probability  = Column(Float, nullable=False)
    amount             = Column(Float, nullable=False)
    status             = Column(String(20), default="open")   # open | reviewed | dismissed
    notes              = Column(Text, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
