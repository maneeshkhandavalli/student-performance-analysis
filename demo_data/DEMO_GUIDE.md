# Demo guide

All names and numbers here are made up. Start the app (`run.bat`), open http://127.0.0.1:5000 and
follow the steps in order. History and Dashboard fill up as you go.

## 1. Predict page: type in one student at a time

| Student (Name / Roll) | Attendance | Study hrs | CA avg | Assignments | Mid-term | Parents' edu | Extracurricular | Internet | Expected result |
|---|---|---|---|---|---|---|---|---|---|
| Ananya Reddy / 24CSE7101 | 96 | 25 | 92 | 100 | 90 | Post-graduate | Yes | Yes | **Low-risk** (~100%) |
| Rohit Verma / 24CSE7102 | 80 | 12 | 65 | 82 | 60 | Intermediate | No | Yes | **Medium-risk** (~83%) |
| Sai Kiran / 24CSE7103 | 55 | 4 | 40 | 45 | 35 | School | No | No | **At-risk** (~100%), shows all 6 suggestions |
| Sneha Patel / 24CSE7104 | 58 | 15 | 78 | 85 | 75 | Graduate | Yes | Yes | **Medium-risk** (~66%) |
| Vikram Singh / 24CSE7105 | 74 | 9 | 56 | 68 | 51 | Intermediate | No | Yes | **Medium-risk** (~98%), 3 suggestions |

**Best one to explain:** Sneha. Her marks are good, but attendance is 58%, so the model does
**not** call her Low-risk. In the SHAP chart, **Attendance** is the biggest bar. This shows the
model learned the attendance-shortage rule and can explain why it gave that result.

**Show validation:** enter attendance `150` or type `abc` in a number box. You get a clear error
message and no prediction.

## 2. Batch Upload page

| File | What it shows |
|---|---|
| `demo_class_20_students.csv` | A class of 20 students: 6 At-risk, 8 Medium-risk, 6 Low-risk. Click **Download results CSV** afterwards. |
| `demo_with_missing_values.csv` | 8 students with 3 blank cells. Blanks are filled with the column median and the predictions still work. |
| `demo_invalid_attendance_120.csv` | Error: *Row 3: Attendance must be between 0 and 100 (got 120)* |
| `demo_wrong_columns.csv` | Error: *CSV is missing columns: ...* |

## 3. History page
Every student from step 1 appears here. Filter by **At-risk** to show only students who need help.

## 4. Dashboard
Model comparison (XGBoost selected, ~89% accuracy), SHAP feature importance (CA average, mid-term
and attendance matter most), class balance before and after SMOTE, and charts of the predictions
you just made.

To start the next demo with an empty history, click **Clear history** on the History page.
