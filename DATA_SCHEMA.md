# 📊 Data Schema

This document explains the structure of datasets used in the project.

---

## 📁 data.csv (Training Data)

Each row represents a completed requirement.

---

### Input Features

| Column                | Description                        | Range                        |
| --------------------- | ---------------------------------- | ---------------------------- |
| story_points          | Size of work                       | 1–20                         |
| dependencies          | Number of blockers or dependencies | 0–10                         |
| teams_involved        | Number of teams required           | 1–5                          |
| team_experience       | Familiarity with work              | 0–10                         |
| complexity            | Technical difficulty               | 1=Low, 2=Medium, 3=High      |
| deadline_days         | Time available                     | 1–30                         |
| external_integrations | APIs or external systems           | 0–10                         |
| production_impact     | Affects live system                | 0=No, 1=Yes                  |
| requirement_clarity   | Clarity of requirement             | 1=Vague, 2=Moderate, 3=Clear |
| test_scope            | Testing required                   | 1=Small, 2=Medium, 3=Large   |
| past_delay_rate       | Historical delay pattern           | 0.0–1.0                      |

---

### Target Variables

| Column                | Description                |
| --------------------- | -------------------------- |
| risk_level            | Low / Medium / High        |
| spillover_probability | Probability of delay (0–1) |

---

## 📁 feedback.csv (Raw Feedback)

Contains user feedback after prediction.

---

### Additional Columns

| Column                          | Description            |
| ------------------------------- | ---------------------- |
| predicted_risk                  | Model prediction       |
| predicted_spillover_probability | Model output           |
| actual_risk                     | What actually happened |
| actual_spillover_days           | Delay in days          |
| feedback_notes                  | Optional comments      |

---

## 📁 approved_data.csv

Cleaned and validated feedback data.

Used for:

* retraining model
* improving prediction accuracy

---

## 🔁 Data Flow

```text
Prediction → Feedback → Approved Data → Retraining → Improved Model
```

---

## 🧠 Key Principles

### 1. Input = Known before execution

All features must be known at planning stage.

---

### 2. Output = Known after execution

Targets represent real outcomes.

---

### 3. Feedback = Ground truth

Model improves only from real-world outcomes.

---

## ⚠️ Important Notes

* Do not mix predicted data with training data
* Only use validated feedback for retraining
* Keep datasets clean and consistent

---

## 🚀 Future Extensions

* Add `spillover_days` prediction
* Add team-specific features
* Add sprint capacity
* Add priority/criticality levels

---
