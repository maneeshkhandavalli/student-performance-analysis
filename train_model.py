"""Trains and compares 4 models (Logistic Regression, SVM, Random Forest, XGBoost).

Steps:
  1. Load students.csv (generated automatically if missing)
  2. 70 / 15 / 15 split into train / validation / test (stratified)
  3. Each model is an imbalanced-learn pipeline: [StandardScaler] -> SMOTE -> classifier
     (SMOTE only runs on training folds, so there is no data leakage)
  4. GridSearchCV with 5-fold stratified cross-validation, optimising macro F1
  5. Best model (by cross-validated macro F1) is saved to model.pkl, all results to metrics.json
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

FEATURES = ["attendance", "study_hours", "ca_avg", "assignments_done",
            "midterm_marks", "parental_education", "extracurricular", "internet_access"]
TARGET = "risk_class"
CLASSES = ["At-risk", "Medium-risk", "Low-risk"]   # fixed order -> labels 0, 1, 2


def build_models():
    smote = lambda: SMOTE(random_state=42)
    return {
        "Logistic Regression": (
            ImbPipeline([("scaler", StandardScaler()), ("smote", smote()),
                         ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": [0.1, 1, 10]}),
        "SVM (RBF kernel)": (
            ImbPipeline([("scaler", StandardScaler()), ("smote", smote()),
                         ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
            {"clf__C": [1, 10], "clf__gamma": ["scale", 0.05]}),
        "Random Forest": (
            ImbPipeline([("smote", smote()),
                         ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 12]}),
        "XGBoost": (
            ImbPipeline([("smote", smote()),
                         ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1]}),
    }


def train(data_path="students.csv"):
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    if df is None or TARGET not in df.columns:          # missing or old-format dataset
        from generate_data import generate
        df = generate(path=data_path)

    le = LabelEncoder().fit(CLASSES)
    le.classes_ = np.array(CLASSES)            # keep logical order At-risk, Medium, Low
    X, y = df[FEATURES], le.transform(df[TARGET])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, fitted = {}, {}
    for name, (pipe, grid) in build_models().items():
        t = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
        m = gs.best_estimator_
        p = m.predict(X_te)
        results[name] = {
            "accuracy":  round(accuracy_score(y_te, p) * 100, 1),
            "precision": round(precision_score(y_te, p, average="macro") * 100, 1),
            "recall":    round(recall_score(y_te, p, average="macro") * 100, 1),
            "f1":        round(f1_score(y_te, p, average="macro") * 100, 1),
            "cv_f1":     round(gs.best_score_ * 100, 1),
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        }
        fitted[name] = m
        r = results[name]
        print(f"{name:20s} acc {r['accuracy']:5}%  F1 {r['f1']:5}%  (cv F1 {r['cv_f1']}%)  {time.time()-t:.1f}s")

    best = max(results, key=lambda k: results[k]["cv_f1"])   # choose on CV, not on test set
    model = fitted[best]
    p_te = model.predict(X_te)
    report = classification_report(y_te, p_te, target_names=CLASSES, output_dict=True)
    print(f"\nSelected model: {best}")
    print(classification_report(y_te, p_te, target_names=CLASSES))

    # Global SHAP importance = mean |SHAP value| on the test set (XGBoost TreeSHAP)
    from explain import shap_values
    xgb = fitted["XGBoost"]
    sv = shap_values(xgb, X_te)                         # (n, classes, features)
    mean_abs = np.abs(sv).mean(axis=(0, 1))
    shap_importance = dict(sorted(zip(FEATURES, [round(float(v), 1) for v in mean_abs / mean_abs.sum() * 100]),
                                  key=lambda kv: -kv[1]))

    smote_counts = np.bincount(SMOTE(random_state=42).fit_resample(X_tr, y_tr)[1])
    metrics = {
        "selected_model": best,
        "models": results,
        "split": {"train": len(X_tr), "validation": len(X_val), "test": len(X_te)},
        "validation_accuracy": round(accuracy_score(y_val, model.predict(X_val)) * 100, 1),
        "classification_report": {c: {k: round(v * 100, 1) if k != "support" else int(v)
                                      for k, v in report[c].items()} for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_te, p_te).tolist(),
        "shap_importance": shap_importance,
        "class_counts_before_smote": dict(zip(CLASSES, np.bincount(y_tr).tolist())),
        "class_counts_after_smote": dict(zip(CLASSES, smote_counts.tolist())),
    }
    joblib.dump({"model": model, "xgb": xgb, "classes": CLASSES, "features": FEATURES,
                 "name": best, "version": 2}, "model.pkl")
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print("Saved model.pkl and metrics.json")
    return metrics


if __name__ == "__main__":
    train()
