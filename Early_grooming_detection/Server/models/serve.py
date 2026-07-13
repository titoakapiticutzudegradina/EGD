import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from runtime.conversation_progress import resolve_conversation_progress
from runtime.conversation_text import messages_to_text
from runtime.message_attribution import attributable_messages, resolve_threshold
from core.predictors import get_predictor, threshold_for_progress
from runtime.transport_crypto import create_handshake, decrypt_json, encrypt_json

# init model name, max messages, text characters and top flagged messages
MODEL_NAME = "roberta_goemotions"
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "5000"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "500000"))
TOP_FLAGGED_MESSAGES = int(os.environ.get("TOP_FLAGGED_MESSAGES", "5"))

# init message model
class Message(BaseModel):
    text: str
    author: str | None = None

# init predict request model
class PredictRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Ordered chat messages in the conversation so far.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional client id for logging or tracing.",
    )
    block_message_counts: list[int] | None = Field(
        default=None,
        description=(
            "Non-empty message count per UI block (ordered). "
            "Used to map blocks to 10%/20%/40% early-detection windows."
        ),
    )

# init flagged message model
class FlaggedMessage(BaseModel):
    index: int = Field(description="0-based index in the request messages list.")
    text: str
    contribution: float = Field(
        description="How much removing this message lowers the predatory score."
    )

# init handshake request model
class HandshakeRequest(BaseModel):
    client_public_key: str = Field(
        ..., description="Base64-encoded SPKI bytes of client ECDH P-256 public key."
    )

# init encrypted envelope model
class EncryptedEnvelope(BaseModel):
    session_id: str
    iv: str
    ciphertext: str

# init predict response model
class PredictResponse(BaseModel):
    label: int = Field(description="0 = not predatory, 1 = predatory.")
    flagged: bool = Field(
        description="True when the model classifies the conversation as predatory."
    )
    predatory: bool = Field(description="Same as label == 1.")
    message_count: int = Field(
        description="Number of non-empty messages used for inference."
    )
    model: str = Field(description="Model name used for this prediction.")
    score: float = Field(description="Predatory probability in [0, 1].")
    window_strategy: str = Field(
        default="full",
        description="Window used for inference (cumulative full history).",
    )
    progress: float | None = Field(
        default=None,
        description="Estimated fraction of the conversation analyzed (0–1).",
    )
    threshold: float | None = Field(
        default=None,
        description="Decision threshold applied for this progress checkpoint.",
    )
    flagged_messages: list[FlaggedMessage] = Field(
        default_factory=list,
        description="When label is 1, top messages by contribution score (highest first).",
    )

#async context manager for lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_name = MODEL_NAME
    #loads the predictor model
    app.state.predictor = get_predictor(MODEL_NAME)
    yield

#init fastapi app
app = FastAPI(
    title="Predatory conversation detection API",
    description="Evaluates a chunk of chat messages and returns a predatory label.",
    lifespan=lifespan,
)

#add cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


CLIENT_DIR = Path(__file__).resolve().parents[2] / "Client"

#root endpoint
@app.get("/")
def root():
    port = os.environ.get("API_PORT", "8000")
    return {
        "service": "predatory-conversation-detection",
        "model": app.state.model_name,
        "endpoints": {
            "handshake": "POST /crypto/handshake",
            "predict_secure": "POST /predict/secure (AES-256-GCM)",
            "predict": "POST /predict (plaintext; dev only)",
            "client_app": "GET /app/",
            "health": "GET /health",
            "docs": "GET /docs",
        },
        "client_url": (
            f"https://127.0.0.1:{port}/app/"
            if os.environ.get("USE_TLS", "0") == "1"
            else f"http://127.0.0.1:{port}/app/"
        ),
        "transport_encryption": "ECDH P-256 + AES-256-GCM on /predict/secure",
        "postman_url": f"http://127.0.0.1:{port}/predict",
        "note": "Open the web UI at /app/ (do not open index.html via file://).",
    }

#redirect to app index
@app.get("/app", include_in_schema=False)
def app_index_redirect():
    return RedirectResponse(url="/app/", status_code=307)

#health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.model_name}

#client config endpoint
@app.get("/config")
def client_config():
    port = int(os.environ.get("API_PORT", "8000"))
    use_tls = os.environ.get("USE_TLS", "0") == "1"
    scheme = "https" if use_tls else "http"
    base = f"{scheme}://127.0.0.1:{port}"
    return {
        "api_base": base,
        "use_tls": use_tls,
        "client_app": f"{base}/app/",
    }

#run predict
def _run_predict(request: PredictRequest) -> PredictResponse:
    if len(request.messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages (max {MAX_MESSAGES}).",
        )

    payload = [m.model_dump() for m in request.messages]
    model_name = app.state.model_name

    text = messages_to_text(payload)

    if not text:
        raise HTTPException(
            status_code=400,
            detail="No non-empty message text to evaluate.",
        )

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation text too long (max {MAX_TEXT_CHARS} characters).",
        )

    predictor = app.state.predictor
    used = sum(1 for m in request.messages if (m.text or "").strip())

    block_counts = request.block_message_counts
    if block_counts is not None:
        if len(block_counts) == 0 or any(c < 1 for c in block_counts):
            raise HTTPException(
                status_code=400,
                detail="block_message_counts must list at least one positive count per block.",
            )
        if sum(block_counts) != used:
            raise HTTPException(
                status_code=400,
                detail=(
                    "block_message_counts must sum to the number of non-empty messages."
                ),
            )
    progress = resolve_conversation_progress(used, block_counts)

    default_threshold = resolve_threshold(predictor)
    progress_thresholds = getattr(predictor, "progress_thresholds", None) or {}
    threshold = threshold_for_progress(
        progress, progress_thresholds, default_threshold
    )

    score = float(predictor.predict_proba([text], progress=[progress])[0])
    label = int(score >= threshold)
    predatory = label == 1

    flagged_messages: list[FlaggedMessage] = []
    if predatory:
        for item in attributable_messages(
            predictor,
            payload,
            top_k=TOP_FLAGGED_MESSAGES,
            progress=progress,
        ):
            flagged_messages.append(FlaggedMessage(**item))

    return PredictResponse(
        label=label,
        flagged=predatory,
        predatory=predatory,
        message_count=used,
        model=model_name,
        score=round(score, 4),
        window_strategy="full",
        progress=round(progress, 4),
        threshold=round(threshold, 4),
        flagged_messages=flagged_messages,
    )

#crypto handshake endpoint
@app.post("/crypto/handshake")
def crypto_handshake(body: HandshakeRequest):
    try:
        return create_handshake(body.client_public_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

#encrypt predict secure endpoint
@app.post("/predict/secure")
def predict_secure(envelope: EncryptedEnvelope):
    try:
        payload = decrypt_json(envelope.model_dump())
        request = PredictRequest.model_validate(payload)
        result = _run_predict(request)
        return encrypt_json(envelope.session_id, result.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

#plain predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if os.environ.get("REQUIRE_ENCRYPTED", "1") == "1":
        raise HTTPException(
            status_code=403,
            detail="Plain /predict is disabled. Use POST /predict/secure after /crypto/handshake.",
        )
    return _run_predict(request)

#predict root endpoint for clients that post to the base URL
@app.post("/", response_model=PredictResponse)
def predict_root(request: PredictRequest):
    if os.environ.get("REQUIRE_ENCRYPTED", "1") == "1":
        raise HTTPException(
            status_code=403,
            detail="Plain POST / is disabled. Use POST /predict/secure.",
        )
    return _run_predict(request)


if CLIENT_DIR.is_dir():
    app.mount(
        "/app",
        StaticFiles(directory=str(CLIENT_DIR), html=True),
        name="client",
    )
