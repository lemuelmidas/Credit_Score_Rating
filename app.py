import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Load model & dataset
model = pickle.load(open("model/credit_model.pkl", "rb"))
df = pd.read_csv("data/sample_credit_data.csv")

st.set_page_config(page_title="Credit Score Dashboard", layout="wide")

# Title
st.title("📊 Credit Score Dashboard")
st.write("Analyze trends and predict credit score using ML model.")

# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.header("🧾 Enter Customer Details")

age = st.sidebar.number_input("Age", 18, 80, 30)
income = st.sidebar.number_input("Monthly Income ($)", 1000, 500000, 50000)
loan = st.sidebar.number_input("Loan Amount ($)", 1000, 500000, 10000)
loan_term = st.sidebar.slider("Loan Term (months)", 6, 60, 12)
credit_history = st.sidebar.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
dependents = st.sidebar.number_input("Dependents", 0, 10, 0)

# ----------------------------
# PREDICTION
# ----------------------------
if st.sidebar.button("Predict Credit Score"):
    features = np.array([[age, income, loan, loan_term, credit_history, dependents]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.sidebar.success("🟢 GOOD Credit Score – Eligible")
    else:
        st.sidebar.error("🔴 BAD Credit Score – Not Eligible")


# ----------------------------
# DATA ANALYSIS SECTION
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Credit Score Distribution")
    fig = px.histogram(df, x="CreditScore", color="CreditScore")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Income vs Loan Analysis")
    fig2 = px.scatter(df, x="Income", y="LoanAmount", color="CreditScore")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# DATA TABLE
# ----------------------------
st.subheader("📋 Raw Dataset Preview")
st.dataframe(df)

st.write("---")
st.write("💡 **Tip:** Add real dataset + more features for better accuracy.")
