import streamlit as st

st.set_page_config(page_title="Secure Login", page_icon="🔐")

st.title("🔐 Clinical Access Login")
st.markdown("Secure login for authorized healthcare staff")

# default credentials
USERNAME = "doctor"
PASSWORD = "1234"

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == USERNAME and password == PASSWORD:
        st.success("✅ Login successful")
        st.markdown("👉 Open the main prediction system from sidebar / multipage menu")
    else:
        st.error("❌ Invalid credentials")