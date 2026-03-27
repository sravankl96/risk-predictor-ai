import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Requirement Risk Intelligence",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Styling
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #172554 100%);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .soft-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 1rem 1rem 0.8rem 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .factor-box {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .factor-title {
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.82rem;
    }

    .mini-stat {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 0.85rem;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 0.15rem;
    }

    .mini-value {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding-left: 14px;
        padding-right: 14px;
        height: 42px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Constants
# =========================================================
BASE_DATA_PATH = "data.csv"
FEEDBACK_PATH = "feedback_log.csv"
APPROVED_DATA_PATH = "approved_data.csv"

FEATURE_COLS = [
    "story_points",
    "dependencies",
    "teams_involved",
    "team_experience",
    "complexity",
    "deadline_days",
    "external_integrations",
    "production_impact",
    "requirement_clarity",
    "test_scope",
    "past_delay_rate"
]

# =========================================================
# UI Header
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">⚠️ Requirement Risk Intelligence</div>
    <p class="hero-subtitle">
        AI reads the requirement, extracts delivery signals, ML predicts risk,
        humans give feedback, approved feedback improves future training.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def clamp(value, low, high):
    return max(low, min(high, value))

def complexity_name(value):
    return {1: "Low", 2: "Medium", 3: "High"}.get(int(value), str(value))

def clarity_name(value):
    return {1: "Vague", 2: "Moderate", 3: "Clear"}.get(int(value), str(value))

def test_scope_name(value):
    return {1: "Small", 2: "Medium", 3: "Large"}.get(int(value), str(value))

def yes_no(value):
    return "Yes" if int(value) == 1 else "No"

def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def make_feedback_id(row_dict):
    raw = f"{row_dict.get('timestamp','')}|{row_dict.get('requirement_text','')}|{row_dict.get('predicted_risk','')}|{row_dict.get('actual_risk','')}"
    return hashlib.md5(raw.encode()).hexdigest()

def normalize_spillover_days_to_probability(days):
    # simple bounded mapping for now
    return float(clamp(days / 10.0, 0.0, 1.0))

def save_feedback(feedback_row, file_path=FEEDBACK_PATH):
    feedback_row = dict(feedback_row)
    feedback_row["feedback_id"] = make_feedback_id(feedback_row)

    new_df = pd.DataFrame([feedback_row])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["feedback_id"], keep="last")
    else:
        combined_df = new_df

    combined_df.to_csv(file_path, index=False)

def append_to_approved_data(row, file_path=APPROVED_DATA_PATH):
    row = dict(row)
    if "approval_source_feedback_id" not in row:
        row["approval_source_feedback_id"] = ""

    new_df = pd.DataFrame([row])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        if (
            "approval_source_feedback_id" in existing_df.columns
            and row["approval_source_feedback_id"]
            and row["approval_source_feedback_id"] in existing_df["approval_source_feedback_id"].astype(str).values
        ):
            return False

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(file_path, index=False)
    return True

def load_training_data():
    base_df = pd.read_csv(BASE_DATA_PATH)

    if os.path.exists(APPROVED_DATA_PATH):
        approved_df = pd.read_csv(APPROVED_DATA_PATH)

        # keep only training columns if extra metadata columns exist
        training_cols = FEATURE_COLS + ["risk_level", "spillover_probability"]
        approved_df = approved_df[[c for c in approved_df.columns if c in training_cols]]

        for col in training_cols:
            if col not in approved_df.columns:
                approved_df[col] = None

        approved_df = approved_df[training_cols]
        base_df = base_df[training_cols]
        combined = pd.concat([base_df, approved_df], ignore_index=True)
        return combined

    return base_df

def train_models():
    training_data = load_training_data()

    X = training_data[FEATURE_COLS]
    y_risk = training_data["risk_level"]
    y_prob = training_data["spillover_probability"]

    label_encoder = LabelEncoder()
    y_risk_encoded = label_encoder.fit_transform(y_risk)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y_risk_encoded)

    reg = RandomForestRegressor(random_state=42)
    reg.fit(X, y_prob)

    return clf, reg, label_encoder, training_data

def extract_features_with_groq(client, requirement_text: str):
    prompt = f"""
You are a software delivery risk analyst.

Read the requirement and return ONLY valid JSON.
Do not return markdown.
Do not add explanation outside JSON.

Return exactly this schema:
{{
  "story_points": integer from 1 to 20,
  "dependencies": integer from 0 to 10,
  "teams_involved": integer from 1 to 5,
  "team_experience": integer from 0 to 10,
  "complexity": integer where 1=Low, 2=Medium, 3=High,
  "deadline_days": integer from 1 to 30,
  "external_integrations": integer from 0 to 10,
  "production_impact": integer where 0=No and 1=Yes,
  "requirement_clarity": integer where 1=Vague, 2=Moderate, 3=Clear,
  "test_scope": integer where 1=Small, 2=Medium, 3=Large,
  "past_delay_rate": float from 0.0 to 1.0,
  "summary": "short summary",
  "reasoning": [
    "reason 1",
    "reason 2",
    "reason 3"
  ]
}}

Guidelines:
- Real-time systems, critical launches, multiple teams, partner APIs, production-facing changes, vague requirements, and short timelines should raise risk-related values.
- If some information is missing, make practical assumptions.
- Be realistic, not dramatic.
- Keep reasoning concise.

Requirement:
{requirement_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You return only strict JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start:end + 1])
        else:
            raise ValueError("Groq did not return valid JSON.")

    parsed["story_points"] = int(clamp(int(parsed.get("story_points", 5)), 1, 20))
    parsed["dependencies"] = int(clamp(int(parsed.get("dependencies", 1)), 0, 10))
    parsed["teams_involved"] = int(clamp(int(parsed.get("teams_involved", 1)), 1, 5))
    parsed["team_experience"] = int(clamp(int(parsed.get("team_experience", 3)), 0, 10))
    parsed["complexity"] = int(clamp(int(parsed.get("complexity", 2)), 1, 3))
    parsed["deadline_days"] = int(clamp(int(parsed.get("deadline_days", 7)), 1, 30))
    parsed["external_integrations"] = int(clamp(int(parsed.get("external_integrations", 0)), 0, 10))
    parsed["production_impact"] = int(clamp(int(parsed.get("production_impact", 0)), 0, 1))
    parsed["requirement_clarity"] = int(clamp(int(parsed.get("requirement_clarity", 2)), 1, 3))
    parsed["test_scope"] = int(clamp(int(parsed.get("test_scope", 2)), 1, 3))
    parsed["past_delay_rate"] = float(clamp(float(parsed.get("past_delay_rate", 0.2)), 0.0, 1.0))

    if "summary" not in parsed:
        parsed["summary"] = "Requirement analyzed."
    if "reasoning" not in parsed or not isinstance(parsed["reasoning"], list):
        parsed["reasoning"] = ["The requirement was converted into structured delivery signals."]

    return parsed

def predict_from_features(clf, reg, label_encoder, features: dict):
    input_df = pd.DataFrame([{
        "story_points": features["story_points"],
        "dependencies": features["dependencies"],
        "teams_involved": features["teams_involved"],
        "team_experience": features["team_experience"],
        "complexity": features["complexity"],
        "deadline_days": features["deadline_days"],
        "external_integrations": features["external_integrations"],
        "production_impact": features["production_impact"],
        "requirement_clarity": features["requirement_clarity"],
        "test_scope": features["test_scope"],
        "past_delay_rate": features["past_delay_rate"]
    }])

    risk_encoded = clf.predict(input_df)[0]
    risk = label_encoder.inverse_transform([risk_encoded])[0]

    prob = float(reg.predict(input_df)[0])
    prob = clamp(prob, 0.0, 1.0)

    return risk, prob

def get_principle_explanations(features: dict):
    explanations = []

    if features["story_points"] >= 8:
        explanations.append({
            "factor": "High story points",
            "rationale": "The work appears large, which increases execution effort and coordination needs.",
            "principle": "Larger work items carry more uncertainty because more tasks, edge cases, and hidden dependencies appear during delivery."
        })

    if features["dependencies"] >= 3:
        explanations.append({
            "factor": "High dependencies",
            "rationale": "This requirement depends on several moving parts or external contributors.",
            "principle": "More dependencies increase coordination overhead and blocking risk because progress depends on others, not just the delivery team."
        })

    if features["teams_involved"] >= 3:
        explanations.append({
            "factor": "Multiple teams involved",
            "rationale": "Delivery depends on coordination across several teams.",
            "principle": "Cross-team work increases communication delay, ownership ambiguity, and sequencing risk."
        })

    if features["team_experience"] <= 2:
        explanations.append({
            "factor": "Low team experience",
            "rationale": "The team may be less familiar with this kind of work or system area.",
            "principle": "Lower experience usually increases estimation error and slows down issue resolution when unexpected problems occur."
        })

    if features["complexity"] == 3:
        explanations.append({
            "factor": "High complexity",
            "rationale": "The implementation likely has more technical depth, edge cases, or architectural impact.",
            "principle": "Complex systems create more unknowns, and unknowns are one of the main drivers of schedule risk."
        })

    if features["deadline_days"] <= 5:
        explanations.append({
            "factor": "Tight deadline",
            "rationale": "There is very little recovery time if anything slips.",
            "principle": "Short deadlines reduce buffer, so even small issues can push work into spillover."
        })

    if features["external_integrations"] >= 2:
        explanations.append({
            "factor": "External integrations",
            "rationale": "The requirement interacts with outside APIs, systems, or vendors.",
            "principle": "External systems create uncertainty because interface changes, access issues, or slow responses are outside direct team control."
        })

    if features["production_impact"] == 1:
        explanations.append({
            "factor": "Production impact",
            "rationale": "This work affects live systems or customer-facing behavior.",
            "principle": "Production-facing work usually demands higher caution, testing, rollout planning, and risk management."
        })

    if features["requirement_clarity"] == 1:
        explanations.append({
            "factor": "Low requirement clarity",
            "rationale": "The requirement appears vague or incomplete.",
            "principle": "Ambiguity increases rework because teams discover missing details only during implementation."
        })

    if features["test_scope"] == 3:
        explanations.append({
            "factor": "Large test scope",
            "rationale": "Validation effort looks extensive.",
            "principle": "Broader testing increases delivery effort and can reveal late-breaking issues that expand scope."
        })

    if features["past_delay_rate"] >= 0.5:
        explanations.append({
            "factor": "High past delay rate",
            "rationale": "Similar work seems to have slipped before.",
            "principle": "Historical patterns matter because recurring delivery conditions often repeat unless the system or process has changed."
        })

    if not explanations:
        explanations.append({
            "factor": "Relatively manageable inputs",
            "rationale": "The extracted signals do not show many strong delivery stressors.",
            "principle": "Work tends to be safer when scope, clarity, coordination, and timeline are all reasonably controlled."
        })

    return explanations

# =========================================================
# Session State
# =========================================================
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "models" not in st.session_state:
    clf, reg, label_encoder, training_data = train_models()
    st.session_state.models = {
        "clf": clf,
        "reg": reg,
        "label_encoder": label_encoder,
        "training_data": training_data
    }

# =========================================================
# Groq client
# =========================================================
groq_api_key = os.getenv("GROQ_API_KEY")
client = None
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except Exception:
        client = None

clf = st.session_state.models["clf"]
reg = st.session_state.models["reg"]
label_encoder = st.session_state.models["label_encoder"]
training_data = st.session_state.models["training_data"]

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Optional manual overrides for extracted signals.")

    manual_story_points = st.slider(
        "Story Points", 1, 20, 5,
        help="Estimated effort size. Bigger work usually carries more uncertainty."
    )
    manual_dependencies = st.slider(
        "Dependencies", 0, 10, 1,
        help="Number of blockers or outside dependencies such as teams, systems, approvals, or vendors."
    )
    manual_teams_involved = st.slider(
        "Teams Involved", 1, 5, 1,
        help="More teams usually means more coordination overhead."
    )
    manual_team_experience = st.slider(
        "Team Experience", 0, 10, 3,
        help="How familiar the delivery team is with this type of work."
    )
    manual_complexity = st.selectbox(
        "Complexity",
        [1, 2, 3],
        format_func=lambda x: ["Low", "Medium", "High"][x - 1],
        help="Technical depth, uncertainty, and number of edge cases."
    )
    manual_deadline_days = st.slider(
        "Deadline (days)", 1, 30, 7,
        help="Shorter timelines reduce schedule buffer."
    )
    manual_external_integrations = st.slider(
        "External Integrations", 0, 10, 0,
        help="How many outside systems, APIs, vendors, or partner services are involved."
    )
    manual_production_impact = st.selectbox(
        "Production Impact", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Whether the change touches live production or customer-facing behavior."
    )
    manual_requirement_clarity = st.selectbox(
        "Requirement Clarity", [1, 2, 3],
        format_func=lambda x: ["Vague", "Moderate", "Clear"][x - 1],
        help="Low clarity often causes rework during implementation."
    )
    manual_test_scope = st.selectbox(
        "Test Scope", [1, 2, 3],
        format_func=lambda x: ["Small", "Medium", "Large"][x - 1],
        help="The broader the validation effort, the higher the schedule pressure."
    )
    manual_past_delay_rate = st.slider(
        "Past Delay Rate", 0.0, 1.0, 0.2, 0.05,
        help="How often similar work slipped in the past."
    )

    use_manual = st.checkbox(
        "Use manual overrides instead of AI extracted values",
        help="Turn this on if you want your own inputs to drive the prediction."
    )

# =========================================================
# Tabs
# =========================================================
tab_analyze, tab_feedback, tab_admin = st.tabs([
    "🔍 Analyze Requirement",
    "🗣️ Feedback & Learning",
    "🛠️ Admin & Retraining"
])

# =========================================================
# Analyze Tab
# =========================================================
with tab_analyze:
    c1, c2 = st.columns([1.35, 0.65])

    with c1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Requirement Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Paste a requirement, feature request, or project scope statement for AI + ML analysis.</div>', unsafe_allow_html=True)

        requirement_text = st.text_area(
            "Requirement",
            height=220,
            placeholder="Example: Build a real-time monitoring system for content distribution with multiple partner integrations, backend changes, and a 5-day deadline.",
            label_visibility="collapsed",
            help="Paste plain English requirement text. The AI will extract structured delivery signals from it."
        )

        analyze_clicked = st.button(
            "Analyze Requirement",
            use_container_width=True,
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How the system works</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="pill">1. AI reads requirement</div>
<div class="pill">2. Extracts delivery signals</div>
<div class="pill">3. ML predicts risk</div>
<div class="pill">4. Human feedback improves system</div>
""", unsafe_allow_html=True)

        st.info("Tip: Use shorter, concrete requirement text for more stable extraction.")
        if client is None:
            st.warning("Groq API key is not available in the terminal where Streamlit is running.")
        else:
            st.success("Groq connection looks ready.")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not requirement_text.strip():
            st.warning("Please paste a requirement first.")
        elif client is None:
            st.error("Groq is not ready. Set GROQ_API_KEY in the same terminal where you run Streamlit, then restart the app.")
        else:
            try:
                with st.spinner("Reading requirement and extracting delivery signals..."):
                    extracted = extract_features_with_groq(client, requirement_text)

                if use_manual:
                    features = {
                        "story_points": manual_story_points,
                        "dependencies": manual_dependencies,
                        "teams_involved": manual_teams_involved,
                        "team_experience": manual_team_experience,
                        "complexity": manual_complexity,
                        "deadline_days": manual_deadline_days,
                        "external_integrations": manual_external_integrations,
                        "production_impact": manual_production_impact,
                        "requirement_clarity": manual_requirement_clarity,
                        "test_scope": manual_test_scope,
                        "past_delay_rate": manual_past_delay_rate,
                        "summary": extracted["summary"],
                        "reasoning": extracted["reasoning"]
                    }
                    feature_source = "Manual override"
                else:
                    features = extracted
                    feature_source = "AI extracted"

                risk, prob = predict_from_features(clf, reg, label_encoder, features)
                principle_explanations = get_principle_explanations(features)

                st.session_state.last_result = {
                    "requirement_text": requirement_text,
                    "features": features,
                    "feature_source": feature_source,
                    "risk": risk,
                    "prob": prob,
                    "principle_explanations": principle_explanations
                }

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        features = result["features"]
        feature_source = result["feature_source"]
        risk = result["risk"]
        prob = result["prob"]
        principle_explanations = result["principle_explanations"]

        top_left, top_right = st.columns([1.05, 0.95])

        with top_left:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🤖 Requirement Understanding</div>', unsafe_allow_html=True)
            st.write(features["summary"])
            st.caption(f"Feature source: {feature_source}")

            st.markdown("**AI reasoning**")
            for item in features["reasoning"]:
                st.write(f"- {item}")
            st.markdown('</div>', unsafe_allow_html=True)

        with top_right:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 ML Final Prediction</div>', unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                if risk == "High":
                    st.error(f"🚨 Risk Level: {risk}")
                elif risk == "Medium":
                    st.warning(f"⚠️ Risk Level: {risk}")
                else:
                    st.success(f"✅ Risk Level: {risk}")

            with mc2:
                st.metric(
                    "Spillover Probability",
                    f"{prob * 100:.1f}%",
                    help="Predicted chance that this requirement will slip or spill over."
                )

            st.progress(float(prob))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧩 Extracted Structured Inputs</div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)
        r3c1, r3c2, r3c3 = st.columns(3)
        r4c1, r4c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="mini-stat"><div class="mini-label">Story Points</div><div class="mini-value">{}</div></div>'.format(features["story_points"]), unsafe_allow_html=True)
        with r1c2:
            st.markdown('<div class="mini-stat"><div class="mini-label">Dependencies</div><div class="mini-value">{}</div></div>'.format(features["dependencies"]), unsafe_allow_html=True)
        with r1c3:
            st.markdown('<div class="mini-stat"><div class="mini-label">Teams Involved</div><div class="mini-value">{}</div></div>'.format(features["teams_involved"]), unsafe_allow_html=True)

        with r2c1:
            st.markdown('<div class="mini-stat"><div class="mini-label">Team Experience</div><div class="mini-value">{}</div></div>'.format(features["team_experience"]), unsafe_allow_html=True)
        with r2c2:
            st.markdown('<div class="mini-stat"><div class="mini-label">Complexity</div><div class="mini-value">{}</div></div>'.format(complexity_name(features["complexity"])), unsafe_allow_html=True)
        with r2c3:
            st.markdown('<div class="mini-stat"><div class="mini-label">Deadline Days</div><div class="mini-value">{}</div></div>'.format(features["deadline_days"]), unsafe_allow_html=True)

        with r3c1:
            st.markdown('<div class="mini-stat"><div class="mini-label">External Integrations</div><div class="mini-value">{}</div></div>'.format(features["external_integrations"]), unsafe_allow_html=True)
        with r3c2:
            st.markdown('<div class="mini-stat"><div class="mini-label">Production Impact</div><div class="mini-value">{}</div></div>'.format(yes_no(features["production_impact"])), unsafe_allow_html=True)
        with r3c3:
            st.markdown('<div class="mini-stat"><div class="mini-label">Requirement Clarity</div><div class="mini-value">{}</div></div>'.format(clarity_name(features["requirement_clarity"])), unsafe_allow_html=True)

        with r4c1:
            st.markdown('<div class="mini-stat"><div class="mini-label">Test Scope</div><div class="mini-value">{}</div></div>'.format(test_scope_name(features["test_scope"])), unsafe_allow_html=True)
        with r4c2:
            st.markdown('<div class="mini-stat"><div class="mini-label">Past Delay Rate</div><div class="mini-value">{:.2f}</div></div>'.format(features["past_delay_rate"]), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Why this risk?</div>', unsafe_allow_html=True)

        for item in principle_explanations:
            st.markdown(f"""
            <div class="factor-box">
                <div class="factor-title">{item['factor']}</div>
                <div><strong>Rationale:</strong> {item['rationale']}</div>
                <div style="margin-top:0.35rem;"><strong>Universal principle:</strong> {item['principle']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Feedback Tab
# =========================================================
with tab_feedback:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feedback Loop</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Capture real outcomes so the system can improve over time.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.last_result is None:
        st.info("Run an analysis first. The latest result will appear here for feedback.")
    else:
        result = st.session_state.last_result
        features = result["features"]
        feature_source = result["feature_source"]
        risk = result["risk"]
        prob = result["prob"]
        saved_requirement_text = result["requirement_text"]

        f1, f2 = st.columns([1, 1])

        with f1:
            feedback_quality = st.selectbox(
                "How good was this prediction?",
                ["Correct", "Partially correct", "Wrong"],
                help="Use this to indicate whether the model judgment matched reality."
            )

            actual_risk = st.selectbox(
                "What was the actual risk level?",
                ["Low", "Medium", "High"],
                help="Choose the real-world risk level after the work progressed or completed."
            )

            actual_spillover_happened = st.selectbox(
                "Did it actually spill over?",
                ["No", "Yes"],
                help="Did the requirement slip beyond its planned delivery window?"
            )

        with f2:
            actual_spillover_days = st.number_input(
                "Actual spillover days",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
                help="How many days it actually slipped."
            )

            feedback_comments = st.text_area(
                "Comments",
                placeholder="Example: dependencies were underestimated, requirement clarity changed later, partner API was delayed...",
                help="Optional qualitative feedback to help future review."
            )

        if st.button("Save Feedback", use_container_width=True, type="primary"):
            feedback_row = {
                "timestamp": datetime.now().isoformat(),
                "requirement_text": saved_requirement_text,
                "feature_source": feature_source,
                "summary": features["summary"],
                "story_points": features["story_points"],
                "dependencies": features["dependencies"],
                "teams_involved": features["teams_involved"],
                "team_experience": features["team_experience"],
                "complexity": features["complexity"],
                "deadline_days": features["deadline_days"],
                "external_integrations": features["external_integrations"],
                "production_impact": features["production_impact"],
                "requirement_clarity": features["requirement_clarity"],
                "test_scope": features["test_scope"],
                "past_delay_rate": features["past_delay_rate"],
                "predicted_risk": risk,
                "predicted_spillover_probability": prob,
                "feedback_quality": feedback_quality,
                "actual_risk": actual_risk,
                "actual_spillover_happened": actual_spillover_happened,
                "actual_spillover_days": actual_spillover_days,
                "comments": feedback_comments
            }

            save_feedback(feedback_row)
            st.success("Feedback saved successfully.")

    feedback_df = safe_read_csv(FEEDBACK_PATH)
    if not feedback_df.empty:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Feedback History</div>', unsafe_allow_html=True)

        cols_to_show = [
            c for c in [
                "timestamp", "predicted_risk", "actual_risk",
                "feedback_quality", "actual_spillover_happened",
                "actual_spillover_days", "comments"
            ] if c in feedback_df.columns
        ]

        st.dataframe(
            feedback_df[cols_to_show].sort_values("timestamp", ascending=False).head(10),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Admin Tab
# =========================================================
with tab_admin:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Admin Control Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Approve trustworthy feedback, append it to approved training data, and retrain the model.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    base_count = len(pd.read_csv(BASE_DATA_PATH)) if os.path.exists(BASE_DATA_PATH) else 0
    approved_count = len(safe_read_csv(APPROVED_DATA_PATH))
    feedback_count = len(safe_read_csv(FEEDBACK_PATH))
    total_training_count = len(training_data)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Base rows", base_count, help="Original curated training dataset.")
    with s2:
        st.metric("Feedback rows", feedback_count, help="Raw user feedback rows collected so far.")
    with s3:
        st.metric("Approved rows", approved_count, help="Feedback rows promoted into approved training data.")
    with s4:
        st.metric("Training rows in use", total_training_count, help="Current total rows used by the model.")

    feedback_df = safe_read_csv(FEEDBACK_PATH)

    if feedback_df.empty:
        st.info("No feedback data available yet.")
    else:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Approve Feedback for Training</div>', unsafe_allow_html=True)

        review_options = []
        for idx, row in feedback_df.sort_values("timestamp", ascending=False).iterrows():
            label = f"{idx} | {row.get('timestamp', '')} | predicted={row.get('predicted_risk', '')} | actual={row.get('actual_risk', '')} | quality={row.get('feedback_quality', '')}"
            review_options.append((label, idx))

        selected_label = st.selectbox(
            "Select a feedback row",
            options=[label for label, _ in review_options],
            help="Pick a feedback row for detailed review before adding it to approved training data."
        )
        selected_idx = dict(review_options)[selected_label]
        selected_row = feedback_df.loc[selected_idx]

        rc1, rc2 = st.columns([1.1, 0.9])

        with rc1:
            st.markdown("**Selected feedback**")
            st.json(selected_row.to_dict())

        with rc2:
            st.markdown("**Approval guidance**")
            st.write("- Approve rows that reflect trustworthy real-world outcomes.")
            st.write("- Avoid approving noisy or contradictory rows.")
            st.write("- You can retrain after accumulating several approved rows.")

        if st.button("✅ Approve Selected Feedback", type="primary", use_container_width=True):
            approved_row = {
                "story_points": selected_row["story_points"],
                "dependencies": selected_row["dependencies"],
                "teams_involved": selected_row["teams_involved"],
                "team_experience": selected_row["team_experience"],
                "complexity": selected_row["complexity"],
                "deadline_days": selected_row["deadline_days"],
                "external_integrations": selected_row["external_integrations"],
                "production_impact": selected_row["production_impact"],
                "requirement_clarity": selected_row["requirement_clarity"],
                "test_scope": selected_row["test_scope"],
                "past_delay_rate": selected_row["past_delay_rate"],
                "risk_level": selected_row["actual_risk"],
                "spillover_probability": normalize_spillover_days_to_probability(selected_row["actual_spillover_days"]),
                "approval_source_feedback_id": str(selected_row.get("feedback_id", "")),
                "approved_at": datetime.now().isoformat()
            }

            added = append_to_approved_data(approved_row)
            if added:
                st.success("Approved feedback added to approved training data.")
            else:
                st.info("This feedback row was already approved earlier.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Retrain Model</div>', unsafe_allow_html=True)
    st.caption("Retraining uses: base data.csv + approved_data.csv")

    if st.button("🔁 Retrain Model with Approved Data", use_container_width=True):
        clf, reg, label_encoder, training_data = train_models()
        st.session_state.models = {
            "clf": clf,
            "reg": reg,
            "label_encoder": label_encoder,
            "training_data": training_data
        }
        st.success("Model retrained successfully with the latest approved data.")

    approved_df = safe_read_csv(APPROVED_DATA_PATH)
    if not approved_df.empty:
        st.markdown("**Approved training rows**")
        st.dataframe(approved_df.tail(10), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption(
    "Flow: Requirement text → Groq extracts structured features → ML predicts risk → human feedback is collected → admin approves trusted rows → model retrains on approved data."
)