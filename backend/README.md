# NeuroEcho — Backend (FastAPI)

## Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Docs at http://localhost:8000/docs

## Endpoints
- `GET /health` — service check
- `POST /predict` — upload audio (wav/m4a/mp3), returns risk band + biomarkers

## Docker
```bash
docker build -t neuroecho-api .
docker run -p 8000:8000 -v $(pwd)/../ml/models:/app/models neuroecho-api
```
