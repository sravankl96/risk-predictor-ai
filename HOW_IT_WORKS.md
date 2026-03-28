# 🧠 How It Works

## Overview

This application predicts delivery risk using a combination of:

* Large Language Model (LLM)
* Machine Learning (ML)
* Rule-based explanations

---

## 🔄 End-to-End Flow

```text
User Requirement (Text)
        ↓
Groq LLM (Understanding)
        ↓
Structured Feature Extraction
        ↓
Machine Learning Model
        ↓
Risk Prediction
        ↓
Explanation Engine
```

---

## 1️⃣ Requirement Input

The user provides a requirement in plain English.

Example:

* Build a monitoring system
* Add alerting logic
* Integrate with external APIs

---

## 2️⃣ AI Feature Extraction (Groq)

The LLM converts text into structured fields:

* story_points
* dependencies
* teams_involved
* complexity
* deadline_days
* external_integrations
* production_impact
* requirement_clarity
* test_scope
* past_delay_rate

### Why this step is needed

ML models cannot understand raw text.

So we convert:

```text
"complex system with 3 teams and tight deadline"
```

Into:

```json
{
  "dependencies": 5,
  "teams_involved": 3,
  "deadline_days": 5,
  "complexity": 3
}
```

---

## 3️⃣ Machine Learning Prediction

We use:

* RandomForestClassifier → Risk Level
* RandomForestRegressor → Spillover Probability

### Input:

Structured features

### Output:

* Risk: Low / Medium / High
* Spillover Probability: 0–100%

---

## 4️⃣ Explanation Layer

The system explains the prediction using:

### Factor

Example:

* High dependencies

### Rationale

* Multiple systems are involved

### Universal Principle

* More dependencies increase coordination risk

---

## 5️⃣ Manual Override

Users can:

* adjust extracted values
* simulate scenarios
* test sensitivity

---

## 6️⃣ Feedback Loop

Users can submit:

* actual risk
* actual spillover
* corrections

Stored in:

* `feedback.csv`

Approved data goes into:

* `approved_data.csv`

---

## 7️⃣ Retraining

When retrained:

* new data is merged with training data
* model learns from real outcomes
* prediction improves over time

---

## 🧠 Design Philosophy

### Separation of responsibilities

| Component | Role                |
| --------- | ------------------- |
| LLM       | Understand language |
| ML        | Predict outcomes    |
| Rules     | Explain decisions   |

---

## ⚖️ Why not only LLM?

LLMs:

* are not consistent for numerical prediction
* do not learn from your data

---

## ⚖️ Why not only ML?

ML:

* cannot understand raw text

---

## ✅ Combined approach

Best of both:

* LLM → understanding
* ML → prediction

---

## 🚀 Summary

This system transforms:

👉 Unstructured requirement
➡️ into structured signals
➡️ into predicted delivery risk
➡️ with explainable reasoning

---
