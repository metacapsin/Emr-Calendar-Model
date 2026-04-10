from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.patient_routes import router as patient_router
from api.routes.predict_slot import router as slot_router
from api.routes.provider_routes import router as provider_router
from src.database.db_connection import init_db
from dotenv import load_dotenv
load_dotenv()
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database")
    init_db()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Healthcare Slot Recommendation ML-powered system",
    version="1.0.0",
    description="ML-powered appointment slot recommendation for EMR systems",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(slot_router)
app.include_router(patient_router)
app.include_router(provider_router)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "1.0.0"}
