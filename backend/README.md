# NeuroEcho — Backend (FastAPI)

Serves the screening API and the static web client from `../web`.

## Run
From the repository root, install the dependencies once, then start the server from `backend/`:
```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload
```

Docs at http://localhost:8000/docs

## Endpoints
- `GET /health` — service check
- `POST /predict` — upload audio (`.wav`, `.flac`, `.mp3`; max 25 MB), returns risk band + biomarkers

## Configuration
- `MODEL_DIR` — directory holding `biomarker_model.pkl` (and optionally `cnn_model.pt`); defaults to `ml/models`

## Tests
```bash
pytest backend/tests
```

## Docker
Build from the repository root so the image can include `ml/` and `web/`:
```bash
docker build -f backend/Dockerfile -t neuroecho .
docker run -p 8000:8000 neuroecho
```
or simply `docker compose up --build`.
