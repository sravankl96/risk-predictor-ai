# ⚠️ Requirement Risk Predictor

## 📌 Overview

This project is an AI + Machine Learning powered application that predicts **delivery risk** and **spillover probability** from plain-English software requirements.

It combines:

* 🤖 **LLM (Groq)** → understands requirement text
* 📊 **Machine Learning (Random Forest)** → predicts risk from structured data
* 🧠 **Explainability Layer** → explains *why* the risk exists using real-world principles

---

## 🚀 What it does

1. User pastes a requirement (example: feature, alert, system change)
2. AI extracts structured signals like:

   * story points
   * dependencies
   * teams involved
   * complexity
   * deadline
   * integrations
   * production impact
3. ML model predicts:

   * Risk Level (Low / Medium / High)
   * Spillover Probability (%)
4. App explains:

   * reasoning
   * rationale
   * universal delivery principles

---

## 🧠 Architecture

```
Requirement Text
        ↓
Groq (LLM)
        ↓
Structured Features
        ↓
Random Forest Model
        ↓
Prediction (Risk + Probability)
        ↓
Explanation Engine
```

---

## 🧩 Features

* ✅ AI-powered requirement understanding
* ✅ ML-based risk prediction
* ✅ Real-time feature extraction
* ✅ Explainable outputs (not black-box)
* ✅ Manual override for tuning inputs
* ✅ Feedback collection system
* ✅ Retraining capability

---

## 📂 Project Structure

```
risk_predictor_app/
│
├── app.py                # Main Streamlit app
├── data.csv              # Base training dataset
├── approved_data.csv     # Clean feedback data
├── feedback.csv          # Raw user feedback
├── model.pkl             # (optional) saved classifier
├── reg.pkl               # (optional) saved regressor
├── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone / open project

```bash
cd risk_predictor_app
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn groq
```

---

## 🔑 Setup API Key (Groq)

In terminal:

```bash
$env:GROQ_API_KEY="your_api_key_here"
```

---

## ▶️ Run the App

```bash
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Input Features Explained

| Feature               | Meaning                           |
| --------------------- | --------------------------------- |
| Story Points          | Size of the work                  |
| Dependencies          | External blockers or dependencies |
| Teams Involved        | Number of teams coordinating      |
| Team Experience       | Familiarity with work             |
| Complexity            | Technical difficulty              |
| Deadline Days         | Time available                    |
| External Integrations | APIs / systems involved           |
| Production Impact     | Affects live system or not        |
| Requirement Clarity   | How clear the requirement is      |
| Test Scope            | Amount of testing needed          |
| Past Delay Rate       | Historical delay pattern          |

---

## 🔁 Feedback & Retraining

The app supports continuous improvement:

### Feedback Flow

1. User reviews prediction
2. Provides actual outcome
3. Stored in `feedback.csv`
4. Approved rows → `approved_data.csv`
5. Retrain model using updated data

---

## 🧠 Key Design Principle

> LLM understands the requirement
> ML predicts based on structured patterns

This avoids inconsistency between:

* human-readable input
* model-based prediction

---

## ⚠️ Limitations

* Initial model trained on synthetic data
* Prediction improves with real feedback
* LLM extraction may vary slightly
* Not a replacement for expert judgment

---

## 🚀 Future Improvements

* Model persistence (joblib)
* Auto-retraining pipeline
* Historical analytics dashboard
* Feature importance visualization
* Team-specific calibration
* Integration with Jira / alerts

---

## 👨‍💻 Author

Built as a practical AI + ML portfolio project
for real-world delivery risk prediction.

---

## 💡 Tip

Use:

* AI section → understand requirement
* ML section → get consistent prediction

---
