from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routers import predict, transactions, dashboard, alerts
from api.database import engine, Base
from api.ml.predictor import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Fraud Detection API...")
    
    # Create DB tables if they don't exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified.")
    except Exception as e:
        logger.error(f"✗ Failed to create database tables: {e}")

    # Load machine learning models
    load_model()
    
    yield
    logger.info("Shutting down Fraud Detection API.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ANN-powered real-time transaction scoring and alert system.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept calls from React (5173/3000) or Streamlit (8501)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include endpoint routes
app.include_router(predict.router,      prefix="/api/predict",      tags=["Prediction"])
app.include_router(transactions.router, prefix="/api/transactions",  tags=["Transactions"])
app.include_router(dashboard.router,    prefix="/api/dashboard",     tags=["Dashboard"])
app.include_router(alerts.router,       prefix="/api/alerts",        tags=["Alerts"])


@app.get("/")
def root():
    return {"status": "online", "service": "fraud-detection-api", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}
