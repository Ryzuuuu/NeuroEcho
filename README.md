# NeuroEcho

NeuroEcho is a voice-first screening platform for early detection of neurological decline using a short smartphone audio sample. The system combines speech biomarkers such as jitter, shimmer, harmonic-to-noise ratio, and pause patterns with lightweight machine learning models to produce a risk band for follow-up care.

> NeuroEcho is a screening and triage aid, not a medical diagnosis. Final assessment should always be made by a qualified clinician.

## Why it matters
- Supports early detection for stroke, Parkinson’s-related changes, and cognitive decline
- Works with a simple 90-second voice test on a low-cost mobile device
- Designed for low-resource or underserved settings where specialist access is limited

## Project structure
- [ml](ml) — training, evaluation, and model artifacts for the biomarker and CNN pipelines
- [backend](backend) — FastAPI API for audio upload and inference
- [web](web) — lightweight frontend that serves the screening experience
- [neuroecho_prototype.py](neuroecho_prototype.py) — prototype inference workflow
- [test_predict.py](test_predict.py) — quick local prediction smoke test

## Tech stack
- Python
- FastAPI
- Uvicorn
- PyTorch
- Librosa
- Parselmouth / Praat-based analysis
- Docker Compose

## Quick start

### 1) Clone and set up the environment
```bash
git clone https://github.com/Ryzuuuu/NeuroEcho.git
cd NeuroEcho
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000 to use the app.

### 3) Run with Docker
```bash
docker compose up --build
```
The API will be available at http://localhost:8000.

## Model training
```bash
cd ml
pip install -r requirements.txt
python src/train_biomarker.py
python src/train_cnn.py
```

## API overview
- GET /health — service health check
- POST /predict — upload an audio file and receive a risk band plus biomarkers
- Swagger docs are available at /docs

## Data and evaluation
The project is designed to work with public voice datasets such as:
- UCI Parkinson’s dataset
- mPower
- DementiaBank / Pitt Corpus

## License
This project is licensed under the MIT License.

## Contact
For collaboration or questions, please open an issue or reach out through the GitHub repository.
