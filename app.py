"""Student Performance Prediction System - Flask web app.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import json, os, sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for

from explain import explain_one
from train_model import FEATURES, train

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
DB = "history.db"
MODEL_VERSION = 2

def load_bundle():
    try:
        b = joblib.load("model.pkl")
        if isinstance(b, dict) and b.get("version") == MODEL_VERSION and os.path.exists("metrics.json"):
            return b
    except Exception:
        pass
    print("No up-to-date model found - training now (takes about a minute)...")
    train()
    return joblib.load("model.pkl")


bundle = load_bundle()
model, xgb_model, CLASSES = bundle["model"], bundle["xgb"], bundle["classes"]
metrics = json.load(open("metrics.json"))

app = Flask(__name__)

# field: (label, min, max, unit / options)
FIELDS = {
    "attendance":         ("Attendance", 0, 100, "%"),
    "study_hours":        ("Study hours per week", 0, 40, "hrs"),
    "ca_avg":             ("Continuous assessment average", 0, 100, "/ 100"),
    "assignments_done":   ("Assignments completed", 0, 100, "%"),
    "midterm_marks":      ("Mid-term marks", 0, 100, "/ 100"),
    "parental_education": ("Parents' education", 0, 4,
                           ["None", "School", "Intermediate", "Graduate", "Post-graduate"]),
    "extracurricular":    ("Extracurricular activities", 0, 1, ["No", "Yes"]),
    "internet_access":    ("Internet access at home", 0, 1, ["No", "Yes"]),
}
SHORT = {"attendance": "Attendance", "study_hours": "Study hours", "ca_avg": "CA average",
         "assignments_done": "Assignments", "midterm_marks": "Mid-term",
         "parental_education": "Parents' education", "extracurricular": "Extracurricular",
         "internet_access": "Internet access"}
BADGE = {"Low-risk": "success", "Medium-risk": "warning", "At-risk": "danger"}


@app.context_processor
def globals_():
    return {"BADGE": BADGE, "SHORT": SHORT}


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, roll TEXT, data TEXT,
        result TEXT, confidence REAL, created TEXT)""")
    return con


def suggestions(d):
    tips = []
    if d["attendance"] < 75:       tips.append("Attendance is below 75% - risk of detention. Attend classes regularly.")
    if d["study_hours"] < 10:      tips.append("Less than 10 study hours a week - plan a daily study schedule.")
    if d["assignments_done"] < 70: tips.append("Several assignments are pending - complete them before the deadline.")
    if d["midterm_marks"] < 50:    tips.append("Mid-term score is low - attend remedial / doubt-clearing classes.")
    if d["ca_avg"] < 55:           tips.append("Continuous assessment marks are weak - revise regularly and take practice tests.")
    if not d["internet_access"]:   tips.append("No internet at home - make use of the college library and labs.")
    return tips or ["Doing well - keep it up and stay consistent."]


def parse(form):
    d = {}
    for f, (label, lo, hi, _) in FIELDS.items():
        try:
            v = float(form.get(f, ""))
        except ValueError:
            raise ValueError(f"{label} is required")
        if not lo <= v <= hi:
            raise ValueError(f"{label} must be between {lo} and {hi}")
        d[f] = v
    return d


@app.route("/")
def index():
    return render_template("index.html", fields=FIELDS, error=None, form={})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = parse(request.form)
    except ValueError as e:
        return render_template("index.html", fields=FIELDS, error=str(e), form=request.form)

    X = pd.DataFrame([d])[FEATURES]
    probs = model.predict_proba(X)[0]
    idx = int(probs.argmax())
    result = CLASSES[idx]
    prob = {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)}
    shap = [(SHORT[f], v) for f, v in explain_one(xgb_model, X, idx, FEATURES)]

    name, roll = request.form.get("name", "").strip(), request.form.get("roll", "").strip()
    with db() as con:
        con.execute("INSERT INTO predictions (name, roll, data, result, confidence, created) VALUES (?,?,?,?,?,?)",
                    (name, roll, json.dumps(d), result, prob[result], datetime.now().strftime("%Y-%m-%d %H:%M")))
    return render_template("result.html", name=name, roll=roll, d=d, fields=FIELDS, result=result,
                           prob=prob, shap=shap, tips=suggestions(d), model_name=bundle["name"],
                           now=datetime.now().strftime("%d %b %Y, %I:%M %p"))


@app.route("/batch", methods=["GET", "POST"])
def batch():
    rows, error, summary = None, None, None
    if request.method == "POST":
        try:
            df = pd.read_csv(request.files["file"])
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                raise ValueError("CSV is missing columns: " + ", ".join(missing))
            df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())     # median imputation
            probs = model.predict_proba(df[FEATURES])
            df["prediction"] = [CLASSES[i] for i in probs.argmax(axis=1)]
            df["confidence"] = (probs.max(axis=1) * 100).round(1)
            rows = df.to_dict("records")
            summary = df["prediction"].value_counts().to_dict()
            app.config["LAST_BATCH"] = df.to_csv(index=False)
        except Exception as e:
            error = str(e)
    return render_template("batch.html", rows=rows, error=error, summary=summary, features=FEATURES)


@app.route("/batch/download")
def batch_download():
    return Response(app.config.get("LAST_BATCH", ""), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.route("/sample.csv")
def sample_csv():
    df = pd.read_csv("students.csv").sample(25, random_state=1)[["student_id"] + FEATURES]
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_students.csv"})


@app.route("/history")
def history():
    risk, day = request.args.get("risk", ""), request.args.get("date", "")
    q, args = "SELECT id, name, roll, result, confidence, created FROM predictions WHERE 1=1", []
    if risk:
        q += " AND result = ?"; args.append(risk)
    if day:
        q += " AND created LIKE ?"; args.append(day + "%")
    with db() as con:
        rows = con.execute(q + " ORDER BY id DESC", args).fetchall()
    return render_template("history.html", rows=rows, risk=risk, day=day, classes=CLASSES)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with db() as con:
        con.execute("DELETE FROM predictions")
    return redirect(url_for("history"))


@app.route("/dashboard")
def dashboard():
    df = pd.read_csv("students.csv")
    with db() as con:
        hist = con.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result").fetchall()
        trend = con.execute("SELECT substr(created,1,10) d, result, COUNT(*) FROM predictions "
                            "GROUP BY d, result ORDER BY d").fetchall()
    days = sorted({t[0] for t in trend})
    trend_series = {c: [next((t[2] for t in trend if t[0] == d and t[1] == c), 0) for d in days] for c in CLASSES}
    shap_imp = [(SHORT[k], v) for k, v in metrics["shap_importance"].items()]
    return render_template("dashboard.html", m=metrics, shap_imp=shap_imp, total=len(df),
                           dataset_counts={c: int((df.risk_class == c).sum()) for c in CLASSES},
                           hist=dict(hist), total_preds=sum(h[1] for h in hist),
                           days=days, trend=trend_series, classes=CLASSES)


if __name__ == "__main__":
    app.run(debug=True)
