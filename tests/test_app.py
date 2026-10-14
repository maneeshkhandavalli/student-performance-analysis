"""Automated tests.  Run with:  python -m pytest -v"""
import io, os, re, sys
from html import unescape
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
    assert b"25 students" in r.data and b'name="csv"' in r.data

def test_batch_results_csv_adds_predicted_risk(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES]; df.loc[0, "study_hours"] = None
    html = _upload(client, df).data.decode()
    csv = re.search(r'name="csv" value="([^"]*)"', html).group(1)
    out = pd.read_csv(io.StringIO(unescape(csv)))
    assert list(out.columns) == FEATURES + ["predicted_risk"]
    assert list(out.predicted_risk) == ["Low-risk", "At-risk"]
    assert pd.isna(out.study_hours[0])                      # uploaded blanks are not overwritten

def test_batch_download_returns_csv_attachment(client):
    r = client.post("/batch/download", data={"csv": "attendance,predicted_risk\n90,Low-risk\n"})
    assert r.mimetype == "text/csv" and "attachment" in r.headers["Content-Disposition"]
    assert r.data.startswith(b"attendance,predicted_risk")

def test_batch_with_missing_values_is_imputed(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES]; df.loc[0, "study_hours"] = None
    r = client.post("/batch", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "m.csv")},
                    content_type="multipart/form-data")
    assert b"2 students" in r.data

def test_batch_missing_column_error(client):
    r = client.post("/batch", data={"file": (io.BytesIO(b"attendance\n80\n"), "bad.csv")},
                    content_type="multipart/form-data")
    assert b"missing columns" in r.data

def _upload(client, df):
    return client.post("/batch", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "b.csv")},
                       content_type="multipart/form-data")

def test_batch_out_of_range_rejected(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES]; df.loc[1, "attendance"] = 500
    assert b"Row 2: Attendance must be between 0 and 100" in _upload(client, df).data

def test_batch_text_value_rejected(client):
    df = pd.DataFrame([GOOD])[FEATURES].astype(object); df.loc[0, "ca_avg"] = "ninety"
    assert b"must be a number" in _upload(client, df).data

@pytest.mark.parametrize("data, message", [({}, b"Please choose a CSV file"),
                                           ({"file": (io.BytesIO(b""), "e.csv")}, b"The CSV file is empty")])
def test_batch_no_or_empty_file(client, data, message):
    assert message in client.post("/batch", data=data, content_type="multipart/form-data").data

def _upload_bytes(client, raw):
    return client.post("/batch", data={"file": (io.BytesIO(raw), "b.csv")}, content_type="multipart/form-data").data

def test_batch_blank_excel_rows_are_ignored(client):
    csv = pd.DataFrame([GOOD, WEAK])[FEATURES].to_csv(index=False) + ",,,,,,,\n,,,,,,,\n"
    assert b"2 students" in _upload_bytes(client, csv.encode())

def test_batch_row_without_features_rejected(client):
    df = pd.DataFrame([GOOD, WEAK])[["name"] + FEATURES].astype(object); df.loc[1, FEATURES] = None
    assert b"Row 2: none of the 8 feature columns" in _upload(client, df).data

def test_batch_fully_blank_column_rejected(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES].astype(object); df["study_hours"] = None
    assert b"Study hours per week column is empty" in _upload(client, df).data

def test_fractional_category_rejected(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES].astype(object); df.loc[1, "parental_education"] = 2.5
    assert b"Row 2: Parents&#39; education must be a whole number" in _upload(client, df).data
    with pytest.raises(ValueError):
        webapp.parse({**GOOD, "extracurricular": 0.5})

@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_batch_excel_encodings_accepted(client, encoding):
    df = pd.DataFrame([{**GOOD, "name": "José"}, WEAK])[["name"] + FEATURES]
    assert b"2 students" in _upload_bytes(client, df.to_csv(index=False).encode(encoding))

def test_batch_headers_are_case_and_space_insensitive(client):
    df = pd.DataFrame([GOOD, WEAK])[FEATURES].rename(columns=lambda c: f" {c.title()} ")
    assert b"2 students" in _upload(client, df).data

def test_batch_xlsx_gets_helpful_error(client):
    assert b"looks like an Excel file" in _upload_bytes(client, b"PK\x03\x04rest-of-zip")

def test_batch_download_get_redirects(client):
    assert client.get("/batch/download").status_code == 302

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
