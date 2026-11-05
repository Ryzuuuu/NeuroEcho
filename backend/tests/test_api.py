import os

import pytest
from fastapi.testclient import TestClient

import main
from inference import ModelNotAvailableError

client = TestClient(main.app)

FAKE_RESULT = {
    "probability": 0.123,
    "band": "GREEN",
    "message": "ok",
    "biomarkers": {
        "fo_hz": 150.0,
        "fhi_hz": 160.0,
        "flo_hz": 140.0,
        "jitter_pct": 0.5,
        "shimmer_pct": 2.0,
        "hnr_db": 20.0,
        "pause_ratio": 0.1,
        "articulation_rate": 3.0,
    },
}


def _upload(name="voice.wav", data=b"RIFFfakewav"):
    return client.post("/predict", files={"file": (name, data, "audio/wav")})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_bad_type():
    r = client.post("/predict", files={"file": ("x.txt", b"abc", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_empty_file():
    assert _upload(data=b"").status_code == 400


def test_predict_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    assert _upload(data=b"12345").status_code == 413


def test_predict_returns_result_and_cleans_up(monkeypatch):
    seen = {}

    def fake_inference(path):
        seen["path"] = path
        assert os.path.exists(path)
        return FAKE_RESULT

    monkeypatch.setattr(main, "run_inference", fake_inference)
    r = _upload()
    assert r.status_code == 200
    body = r.json()
    assert body["band"] == "GREEN"
    assert body["probability"] == pytest.approx(0.123)
    assert body["biomarkers"]["hnr_db"] == 20.0
    assert not os.path.exists(seen["path"])  # temp file removed


def test_predict_reports_unavailable_model(monkeypatch):
    def boom(path):
        raise ModelNotAvailableError("missing")

    monkeypatch.setattr(main, "run_inference", boom)
    assert _upload().status_code == 503


def test_predict_reports_unreadable_audio(monkeypatch):
    def boom(path):
        raise ValueError("not audio")

    monkeypatch.setattr(main, "run_inference", boom)
    assert _upload().status_code == 422
