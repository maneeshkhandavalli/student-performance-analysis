"""Automated tests.  Run with:  python -m pytest -v"""
import io, os, sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as webapp
from explain import explain_one
from train_model import CLASSES, FEATURES

GOOD = dict(name="A", roll="1", attendance=95, study_hours=20, ca_avg=85, assignments_done=95,
            midterm_marks=85, parental_education=3, extracurricular=1, internet_access=1)
WEAK = dict(name="B", roll="2", attendance=50, study_hours=3, ca_avg=40, assignments_done=40,
            midterm_marks=35, parental_education=0, extracurricular=0, internet_access=0)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "DB", str(tmp_path / "test.db"))   # separate test database
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


# ---------- input validation ----------
def test_valid_input_parses():
    assert webapp.parse(GOOD)["attendance"] == 95

@pytest.mark.parametrize("value", [0, 100])
def test_attendance_boundaries_accepted(value):
    assert webapp.parse({**GOOD, "attendance": value})["attendance"] == value

@pytest.mark.parametrize("value", [-1, 101, 150])
def test_attendance_out_of_range_rejected(value):
    with pytest.raises(ValueError):
        webapp.parse({**GOOD, "attendance": value})

def test_missing_field_rejected():
    d = dict(GOOD); d.pop("midterm_marks")
    with pytest.raises(ValueError):
        webapp.parse(d)

def test_text_in_number_field_rejected():
    with pytest.raises(ValueError):
        webapp.parse({**GOOD, "study_hours": "abc"})


# ---------- model ----------
def test_model_outputs_valid_class():
    X = pd.DataFrame([webapp.parse(GOOD)])[FEATURES]
    assert webapp.CLASSES[webapp.model.predict(X)[0]] in CLASSES

def test_probabilities_sum_to_one():
    X = pd.DataFrame([webapp.parse(WEAK)])[FEATURES]
    assert abs(webapp.model.predict_proba(X)[0].sum() - 1) < 1e-6

def test_strong_student_is_low_risk():
    X = pd.DataFrame([webapp.parse(GOOD)])[FEATURES]
    assert CLASSES[webapp.model.predict(X)[0]] == "Low-risk"

def test_weak_student_is_at_risk():
    X = pd.DataFrame([webapp.parse(WEAK)])[FEATURES]
    assert CLASSES[webapp.model.predict(X)[0]] == "At-risk"

def test_shap_returns_all_features():
    X = pd.DataFrame([webapp.parse(GOOD)])[FEATURES]
    assert len(explain_one(webapp.xgb_model, X, 2, FEATURES)) == len(FEATURES)

def test_shap_sorted_by_impact():
    X = pd.DataFrame([webapp.parse(WEAK)])[FEATURES]
    vals = [abs(v) for _, v in explain_one(webapp.xgb_model, X, 0, FEATURES)]
    assert vals == sorted(vals, reverse=True)

def test_model_meets_accuracy_target():
    assert webapp.metrics["models"][webapp.metrics["selected_model"]]["accuracy"] >= 85


# ---------- web pages / integration ----------
@pytest.mark.parametrize("page", ["/", "/batch", "/history", "/dashboard", "/sample.csv"])
def test_pages_load(client, page):
    assert client.get(page).status_code == 200

def test_predict_page_and_history_saved(client):
    r = client.post("/predict", data=GOOD)
    assert r.status_code == 200 and b"Low-risk" in r.data
    assert b"Low-risk" in client.get("/history").data

def test_invalid_form_shows_error(client):
    r = client.post("/predict", data={**GOOD, "attendance": 200})
    assert b"must be between" in r.data

def test_batch_upload_csv(client):
    csv = client.get("/sample.csv").data
    r = client.post("/batch", data={"file": (io.BytesIO(csv), "s.csv")}, content_type="multipart/form-data")
    assert b"25 students" in r.data
    assert client.get("/batch/download").data.startswith(b"student_id")

def test_batch_with_missing_values_is_imputed(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES]; df.loc[0, "study_hours"] = None
    r = client.post("/batch", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "m.csv")},
                    content_type="multipart/form-data")
    assert b"2 students" in r.data

def test_batch_missing_column_error(client):
    r = client.post("/batch", data={"file": (io.BytesIO(b"attendance\n80\n"), "bad.csv")},
                    content_type="multipart/form-data")
    assert b"missing columns" in r.data

def test_history_filter_by_risk(client):
    client.post("/predict", data=GOOD); client.post("/predict", data=WEAK)
    r = client.get("/history?risk=At-risk").data
    assert b"badge bg-danger" in r and b"badge bg-success" not in r

def test_html_is_escaped(client):
    r = client.post("/predict", data={**GOOD, "name": "<script>alert(1)</script>"})
    assert b"<script>alert(1)</script>" not in r.data

def test_sql_injection_is_harmless(client):
    client.post("/predict", data={**GOOD, "name": "x'); DROP TABLE predictions;--"})
    assert client.get("/history").status_code == 200
