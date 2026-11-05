import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import ALLOWED_AUDIO, MAX_UPLOAD_BYTES
from inference import ModelNotAvailableError, run_inference
from schemas import PredictionResponse

logger = logging.getLogger("neuroecho")

app = FastAPI(title="NeuroEcho API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "NeuroEcho"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext or 'unknown'}. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO))}",
        )

    raw_data = await file.read()
    if not raw_data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(raw_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Uploaded file is too large")

    # The audio is analysed from a temporary file that is always removed;
    # recordings are never kept on disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(raw_data)
        tmp_path = tmp.name

    try:
        result = await run_in_threadpool(run_inference, tmp_path)
    except ModelNotAvailableError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(503, "Screening model is not available on the server")
    except Exception:
        logger.exception("Inference failed for %s", file.filename)
        raise HTTPException(422, "Could not analyse the audio; please record again")
    finally:
        os.unlink(tmp_path)

    logger.info("Prediction: probability=%s band=%s", result["probability"], result["band"])
    return result


# Mount the static web directory at the end so API routes are not shadowed
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
