import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ---------------- PDF FUNCTION ----------------
def generate_pdf(patient_data, prediction_text, probability):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 800, "Heart Disease Risk Prediction Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 770, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 730
    for key, value in patient_data.items():
        pdf.drawString(100, y, f"{key}: {value}")
        y -= 20

    pdf.drawString(100, y - 10, f"Prediction: {prediction_text}")
    pdf.drawString(100, y - 30, f"Risk Probability: {probability:.2%}")

    pdf.save()
    buffer.seek(0)
    return buffer

# ---------------- HEADER ----------------
st.title("❤️ Explainable Heart Disease Risk Prediction Dashboard")
st.markdown("### AI + XAI powered cardiovascular clinical decision support")

col1, col2 = st.columns([1, 2])

# ---------------- INPUT PANEL ----------------
with col1:
    st.subheader("🩺 Patient Inputs")

    age = st.slider("Age", 20, 100, 45)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# ---------------- DASHBOARD ----------------
with col2:
    st.subheader("📊 Risk Dashboard")

    if st.button("🔍 Predict + Explain"):
        input_df = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]], columns=feature_names)

        scaled = scaler.transform(input_df)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        prediction_text = "High Risk" if prediction == 1 else "Low Risk"

        if prediction == 1:
            st.error(f"⚠️ High Risk ({probability:.2%})")
        else:
            st.success(f"✅ Low Risk ({probability:.2%})")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Probability %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig_gauge, width="stretch")

        # Clinical chart
        chart_df = pd.DataFrame({
            "Feature": ["Age", "BP", "Cholesterol", "Heart Rate"],
            "Value": [age, trestbps, chol, thalach]
        })
        fig = px.bar(chart_df, x="Feature", y="Value", title="Clinical Indicators")
        st.plotly_chart(fig, width="stretch")

        # SHAP
        st.subheader("🧠 Explainable AI Insights")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled)

        if isinstance(shap_values, list):
            impact_values = np.ravel(shap_values[1])
        else:
            impact_values = np.ravel(shap_values)

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": impact_values[:len(feature_names)]
        })

        shap_df["Absolute Impact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values("Absolute Impact", ascending=False)

        fig_shap = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            title="Feature Contribution to Prediction"
        )
        st.plotly_chart(fig_shap, width="stretch")

        # PDF Download
        patient_data = {
            "Age": age,
            "Sex": sex,
            "Chest Pain": cp,
            "Blood Pressure": trestbps,
            "Cholesterol": chol,
            "Heart Rate": thalach
        }

        pdf_file = generate_pdf(patient_data, prediction_text, probability)

        st.download_button(
            label="📄 Download Patient Report PDF",
            data=pdf_file,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )