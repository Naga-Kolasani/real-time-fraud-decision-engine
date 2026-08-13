"""
FastAPI app for the Real-Time AI Fraud Decision Engine.

Just /health for now. /score_transaction comes next.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import HealthResponse
from src.models.infer import MODEL_VERSION, load_model_artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once when the app starts, not on every request.
    app.state.artifact = load_model_artifact()
    yield


app = FastAPI(
    title="Real-Time AI Fraud Decision Engine",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Just checks the model loaded and the service is up.
    return HealthResponse(
        status="ok",
        model_name=app.state.artifact["model_name"],
        model_version=MODEL_VERSION,
    )
