import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation Dashboard")
st.markdown("### Performance analysis of heart disease prediction model")

# Metrics
accuracy = 0.89
precision = 0.88
recall = 0.90
f1 = 0.89
roc_auc = 0.92

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.0%}")
col2.metric("Precision", f"{precision:.0%}")
col3.metric("Recall", f"{recall:.0%}")
col4.metric("F1 Score", f"{f1:.0%}")
col5.metric("ROC AUC", f"{roc_auc:.0%}")

# Confusion Matrix
st.subheader("📊 Confusion Matrix")

cm_df = pd.DataFrame(
    [[45, 5],
     [4, 46]],
    columns=["Pred No Disease", "Pred Disease"],
    index=["Actual No Disease", "Actual Disease"]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    title="Confusion Matrix"
)
st.plotly_chart(fig_cm, width="stretch")

# ROC Curve
st.subheader("📈 ROC Curve")

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=[0, 0.1, 0.2, 1],
    y=[0, 0.85, 0.95, 1],
    mode="lines",
    name="ROC Curve"
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    name="Baseline",
    line=dict(dash="dash")
))
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    title="Receiver Operating Characteristic"
)
st.plotly_chart(fig_roc, width="stretch")