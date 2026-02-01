import streamlit as st
import joblib
import numpy as np

model = joblib.load("rf.joblib")

st.title("Customer Churn Prediction App")

a= st.number_input("Tenure (months)")
b = st.number_input("Monthly Charges")
c=st.number_input("Total charges")

d=st.selectbox("Internet services_fiberoptics",[True,False])
e=st.selectbox("InternetSerice_No",[True,False])
f=st.selectbox("contract_one year",[True,False])
g=st.selectbox("Contract Two year",[True,False])

h=st.selectbox("PaymentMethod_Creditcard(automatic)",[True,False])
i=st.selectbox("PaymentMethod Electronic check",[True,False])
j=st.selectbox("Payment method mailed check",[True,False])





if st.button("Predict"):
    prediction = model.predict([[a,b,c,d,e,f,g,h,i,j]])
    #print(st.title(f"Customer will stay if 1 :  {prediction[0]}"))
    if prediction[0]==0:
        st.subheader(f"Customer will churn ❌ ")
    else:
        st.subheader(f"Customer will stay ✅")