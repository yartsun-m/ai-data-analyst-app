import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.validation_service import validate_dataframe
from app.services.ml_service import train_models
from app.ml.explainability import _scores_to_feature_list
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_llm(client):
    r = client.get("/health/llm")
    assert r.status_code == 200
    assert "configured" in r.json()


def test_validation_detects_empty():
    df = pd.DataFrame()
    report = validate_dataframe(df, {})
    assert report["passed"] is False


def test_validation_numeric_issues():
    df = pd.DataFrame({"price": ["a", "b", "10"]})
    report = validate_dataframe(df, {"price": "numeric"})
    assert report["passed"] is False
    assert any("non-numeric" in i for i in report["issues"])


def test_feature_importance_aggregation():
    names = ["cat__Category_A", "cat__Category_B", "freq__Color_freq"]
    scores = np.array([0.3, 0.2, 0.5])
    result = _scores_to_feature_list(names, scores)
    features = {item["feature"]: item["importance"] for item in result}
    assert "Category" in features
    assert "Color" in features


def test_train_regression_synthetic():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "category": rng.choice(["A", "B", "C"], size=n),
        "target": rng.normal(size=n),
    })
    df["target"] = df["target"] + df["x1"] * 2
    result = train_models(df, "target", random_state=42)
    assert result["task_type"] == "regression"
    assert result["best_model"] is not None
    assert "cross_validation" in result
    assert "_best_pipeline" in result


def test_upload_and_dataset_pagination(client, tmp_path):
    csv_content = "a,b,target\n1,2,10\n3,4,20\n5,6,30\n"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    upload = client.post("/upload", files=files)
    assert upload.status_code == 200, upload.text
    session_id = upload.json()["session_id"]

    page = client.get(f"/dataset?session_id={session_id}&page=1&page_size=2")
    assert page.status_code == 200
    data = page.json()
    assert data["total_rows"] == 3
    assert len(data["rows"]) == 2


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests" in r.text or "python" in r.text.lower()


def test_custom_eda_and_clustering(client):
    csv_content = "x,y,cat,target\n1,2,A,10\n2,3,A,11\n3,4,B,12\n4,5,B,13\n5,6,C,14\n"
    files = {"file": ("eda.csv", csv_content, "text/csv")}
    session_id = client.post("/upload", files=files).json()["session_id"]
    client.post("/clean", json={"session_id": session_id})

    eda = client.post(
        "/eda/custom",
        json={"session_id": session_id, "x_column": "x", "y_column": "y", "chart_type": "scatter"},
    )
    assert eda.status_code == 200
    assert eda.json()["chart"]["type"] == "scatter"

    cluster = client.post("/clustering", json={"session_id": session_id, "n_clusters": 2})
    assert cluster.status_code == 200
    assert cluster.json()["clustering"]["n_clusters"] == 2


def test_anomaly_detection(client):
    rows = ["a,b,c\n"] + [f"{i},{i*2},{i*3}\n" for i in range(30)]
    files = {"file": ("anom.csv", "".join(rows), "text/csv")}
    session_id = client.post("/upload", files=files).json()["session_id"]
    result = client.post("/anomaly", json={"session_id": session_id, "contamination": 0.05})
    assert result.status_code == 200
    assert "anomaly_count" in result.json()["anomaly"]


def test_mean_encoder_fit():
    from app.ml.encoders import MeanEncoder
    import pandas as pd
    import numpy as np

    X = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    y = np.array([10.0, 20.0, 12.0, 30.0])
    enc = MeanEncoder()
    enc.fit(X, y)
    out = enc.transform(X)
    assert out.shape == (4, 1)
    assert out[0, 0] > 0

