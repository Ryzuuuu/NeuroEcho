from pydantic import BaseModel


class Biomarkers(BaseModel):
    fo_hz: float
    fhi_hz: float
    flo_hz: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    pause_ratio: float
    articulation_rate: float


class PredictionResponse(BaseModel):
    probability: float
    band: str            # GREEN | YELLOW | RED
    message: str
    biomarkers: Biomarkers
