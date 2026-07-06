import tempfile
import os
import shutil
import wave
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schemas import PredictionResponse
from inference import run_inference
from config import ALLOWED_AUDIO

app = FastAPI(title="NeuroEcho API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Debug directory to save uploaded audio files
DEBUG_DIR = Path(__file__).parent / "debug_uploads"
DEBUG_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "NeuroEcho"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    raw_data = await file.read()
    file_size = len(raw_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(raw_data)
        tmp_path = tmp.name

    # Debug: save a copy and log WAV metadata
    timestamp = int(time.time())
    debug_copy = DEBUG_DIR / f"upload_{timestamp}{ext}"
    shutil.copy2(tmp_path, debug_copy)
    print(f"\n{'='*60}")
    print(f"[DEBUG] Received file: {file.filename}")
    print(f"[DEBUG] File size: {file_size} bytes")
    print(f"[DEBUG] Saved debug copy: {debug_copy}")

    if ext == ".wav":
        try:
            with wave.open(tmp_path, 'rb') as wf:
                print(f"[DEBUG] WAV channels: {wf.getnchannels()}")
                print(f"[DEBUG] WAV sample rate: {wf.getframerate()} Hz")
                print(f"[DEBUG] WAV sample width: {wf.getsampwidth()} bytes")
                print(f"[DEBUG] WAV num frames: {wf.getnframes()}")
                duration = wf.getnframes() / wf.getframerate()
                print(f"[DEBUG] WAV duration: {duration:.2f} seconds")
        except Exception as e:
            print(f"[DEBUG] Could not read WAV metadata: {e}")

    try:
        result = run_inference(tmp_path)
        print(f"[DEBUG] Result: probability={result['probability']}, band={result['band']}")
        print(f"[DEBUG] Biomarkers: {result['biomarkers']}")
        print(f"{'='*60}\n")
    finally:
        os.unlink(tmp_path)

    return result

# Mount the static web directory at the end so API routes are not shadowed
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

