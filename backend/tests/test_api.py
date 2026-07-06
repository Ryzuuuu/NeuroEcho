from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_rejects_bad_type():
    r = client.post("/predict", files={"file": ("x.txt", b"abc", "text/plain")})
    assert r.status_code == 400
