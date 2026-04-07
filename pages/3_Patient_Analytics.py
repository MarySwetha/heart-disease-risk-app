import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Patient Analytics", page_icon="📊", layout="wide")

st.title("📊 Patient Analytics Dashboard")
st.markdown("Clinical insights from stored patient risk assessments")

csv_file = "patient_history.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)

    # ---------------- KPIs ----------------
    total_patients = len(df)
    high_risk = df["Prediction"].str.contains("High").sum()
    low_risk = total_patients - high_risk

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_patients)
    col2.metric("High Risk Cases", high_risk)
    col3.metric("Low Risk Cases", low_risk)

    # ---------------- RISK PIE ----------------
    st.subheader("🥧 Risk Distribution")
    risk_counts = df["Prediction"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]

    fig_pie = px.pie(risk_counts, names="Risk", values="Count")
    st.plotly_chart(fig_pie, width="stretch")

    # ---------------- AGE HISTOGRAM ----------------
    st.subheader("📈 Age Distribution")
    fig_age = px.histogram(df, x="Age")
    st.plotly_chart(fig_age, width="stretch")

    # ---------------- CHOLESTEROL TREND ----------------
    st.subheader("🩺 Cholesterol Trend")
    fig_chol = px.line(df, y="Cholesterol")
    st.plotly_chart(fig_chol, width="stretch")

else:
    st.warning("No patient history found yet.")