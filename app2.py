import streamlit as st
import joblib
import numpy as np

model = joblib.load("rf.joblib")

st.title("Customer Churn Prediction App")

tenure = st.number_input("Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
internet = st.radio("Internet Service",["Fiber Optic", "No"])
internet_fiber = 1 if internet == "Fiber Optic" else 0
internet_no = 1 if internet == "No" else 0
contract = st.radio("Contract Type",["Month-to-Month", "One Year", "Two Year"])
contract_one = 1 if contract == "One Year" else 0
contract_two = 1 if contract == "Two Year" else 0
payment = st.radio("Payment Method",["Credit Card (Automatic)", "Electronic Check", "Mailed Check"])
pay_credit = 1 if payment == "Credit Card (Automatic)" else 0
pay_elec = 1 if payment == "Electronic Check" else 0
pay_mail = 1 if payment == "Mailed Check" else 0

if st.button("Predict"):
    input_data = np.array([[
        tenure,
        monthly,
        total,
        internet_fiber,
        internet_no,
        contract_one,
        contract_two,
        pay_credit,
        pay_elec,
        pay_mail
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.write("🔍 Prediction confidence:", prob)

    if prediction == 1:
        st.success("Customer will STAY ✅")
    else:
        st.error("Customer will CHURN ❌")
