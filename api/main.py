from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict_slot import router as slot_router
from src.database.db_connection import init_db
from dotenv import load_dotenv
load_dotenv()
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("Starting up — initializing database")
    init_db()
    logger.info("EMR Slot Recommendation Engine - API Mode: MINIMAL")
    logger.info("Active endpoints: POST /recommend-slots, GET /health")
    logger.info("=" * 70)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Healthcare Slot Recommendation ML-powered system",
    version="1.0.0",
    description="ML-powered appointment slot recommendation for EMR systems",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core recommendation router only
app.include_router(slot_router)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "1.0.0"}
