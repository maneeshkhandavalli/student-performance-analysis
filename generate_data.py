"""Creates the student dataset (students.csv) used for training.

The data is synthetic but realistic: features are correlated through a hidden
"ability" value, attendance below 65% has a sharp penalty, and the classes are
imbalanced (~14% At-risk, ~62% Medium-risk, ~24% Low-risk) - which is why SMOTE
is used during training.
"""
import numpy as np
import pandas as pd

def generate(n=1200, seed=42, path="students.csv"):
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    clip = lambda x, lo, hi: np.clip(np.round(x, 1), lo, hi)

    df = pd.DataFrame({
        "student_id":         [f"STU{i:04d}" for i in range(1, n + 1)],
        "attendance":         clip(78 + 9 * ability + rng.normal(0, 9, n), 30, 100),
        "study_hours":        clip(12 + 4 * ability + rng.normal(0, 5, n), 0, 40),
        "ca_avg":             clip(65 + 11 * ability + rng.normal(0, 8, n), 0, 100),
        "assignments_done":   clip(80 + 8 * ability + rng.normal(0, 10, n), 0, 100),
        "midterm_marks":      clip(60 + 13 * ability + rng.normal(0, 10, n), 0, 100),
        "parental_education": rng.integers(0, 5, n),   # 0 none .. 4 postgraduate
        "extracurricular":    rng.integers(0, 2, n),
        "internet_access":    rng.choice([0, 1], n, p=[0.25, 0.75]),
    })

    # Rule-like (threshold) effects, as seen in real academic data
    score = (
        0.25 * df.ca_avg
        + 0.25 * df.midterm_marks
        + np.where(df.assignments_done >= 70, 6, -4)                       # pending assignments
        + np.where(df.study_hours >= 10, 5, 0) * (0.5 + 0.5 * df.internet_access)
        + np.where(df.parental_education >= 3, 2.5, 0)
        + 1.5 * df.extracurricular
        - np.where(df.attendance < 65, 14, np.where(df.attendance < 75, 5, 0))  # attendance shortage
        - np.where((df.midterm_marks < 50) & (df.ca_avg < 55), 8, 0)       # weak in both exams
        + rng.normal(0, 2.5, n)
    )
    lo, hi = np.percentile(score, [14, 76])
    df["risk_class"] = np.where(score >= hi, "Low-risk",
                        np.where(score >= lo, "Medium-risk", "At-risk"))
    df.to_csv(path, index=False)
    print(f"Saved {n} rows to {path}")
    print(df.risk_class.value_counts())
    return df

if __name__ == "__main__":
    generate()
