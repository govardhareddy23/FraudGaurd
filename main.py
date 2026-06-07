from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import predict, transactions, dashboard, alerts
from database import engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating DB tables...")
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ANN-powered fraud detection with real-time monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,      prefix="/api/predict",      tags=["Prediction"])
app.include_router(transactions.router, prefix="/api/transactions",  tags=["Transactions"])
app.include_router(dashboard.router,    prefix="/api/dashboard",     tags=["Dashboard"])
app.include_router(alerts.router,       prefix="/api/alerts",        tags=["Alerts"])


@app.get("/")
def root():
    return {"status": "online", "service": "fraud-detection-api"}


@app.get("/health")
def health():
    return {"status": "healthy"}
