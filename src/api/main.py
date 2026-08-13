"""
FastAPI app for the Real-Time AI Fraud Decision Engine.

/health and /score_transaction.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import HealthResponse, ScoreResponse, TransactionRequest
from src.models.infer import MODEL_VERSION, load_model_artifact, score_transaction


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


@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction_endpoint(
    transaction: TransactionRequest,
) -> ScoreResponse:
    # Run one validated transaction through the saved model.
    return score_transaction(transaction.model_dump())
