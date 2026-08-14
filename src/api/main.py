"""
FastAPI app for the Real-Time AI Fraud Decision Engine.

/health and /score_transaction.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import HealthResponse, ScoreResponse, TransactionRequest
from src.models.infer import MODEL_VERSION, load_model_artifact, score_transaction
from src.monitoring.log_predictions import log_prediction


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
def score_transaction_endpoint(transaction: TransactionRequest) -> ScoreResponse:
    """ Score one validated transaction and log the successful prediction. """
    transaction_data = transaction.model_dump()
    prediction = score_transaction(transaction_data)

    log_prediction(transaction=transaction_data, prediction=prediction)

    return prediction