import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Risk App", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Risk Prediction System")
st.markdown("AI-powered cardiovascular risk prediction with Explainable AI")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("🩺 Enter Patient Data")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", [0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

# ---------------- DATAFRAME ----------------
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]], columns=feature_names)

scaled = scaler.transform(input_data)

# ---------------- PDF FUNCTION ----------------
def generate_pdf(patient_data, prediction_text, probability):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Heart Disease Risk Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    for key, value in patient_data.items():
        story.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction: {prediction_text}", styles["Heading2"]))
    story.append(Paragraph(f"Risk Probability: {probability:.2%}", styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))

    doc.build(story)
    return temp_file.name

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk"):
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    prediction_text = "⚠️ High Risk of Heart Disease" if prediction == 1 else "✅ Low Risk of Heart Disease"

    st.subheader("🩺 Prediction Result")
    st.success(prediction_text)
    st.metric("Risk Probability", f"{probability:.2%}")

    # ---------------- GAUGE CHART ----------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Risk Level (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(fig, width="stretch")

    # ---------------- SHAP EXPLAINABILITY ----------------
    st.subheader("🧠 Explainable AI Insights")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Fix for different SHAP output structures
    if isinstance(shap_values, list):
        impact_values = np.array(shap_values[1]).flatten()
    else:
        impact_values = np.array(shap_values).flatten()

    # Make sure lengths match
    impact_values = impact_values[:len(feature_names)]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": impact_values
    }).sort_values("Impact", ascending=False)

    st.bar_chart(shap_df.set_index("Feature"), width="stretch")

    # ---------------- SAVE TO CSV ----------------
    history_entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "BP": trestbps,
        "Cholesterol": chol,
        "Heart Rate": thalach,
        "Prediction": prediction_text,
        "Probability": f"{probability:.2%}"
    }

    csv_file = "patient_history.csv"

    if os.path.exists(csv_file):
        history_df = pd.read_csv(csv_file)
        history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
    else:
        history_df = pd.DataFrame([history_entry])

    history_df.to_csv(csv_file, index=False)

    # ---------------- PDF DOWNLOAD ----------------
    patient_data = input_data.iloc[0].to_dict()
    pdf_file = generate_pdf(patient_data, prediction_text, probability)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Medical Report",
            f,
            file_name="heart_risk_report.pdf",
            mime="application/pdf"
        )

# ---------------- HISTORY TABLE ----------------
st.subheader("📁 Patient Prediction History")

if "history" in st.session_state and st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, width="stretch")