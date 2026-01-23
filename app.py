import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5', compile=False)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Card-like container */
    .input-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        max-width: 600px;
        margin: auto;
    }

    /* Buttons */
    div.stButton > button {
        background: #4CAF50;
        color: white;
        padding: 10px 25px;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📌 Customer Churn Prediction", anchor=None)


geography = st.selectbox('🌍 Geography', encoder.categories_[0])
gender = st.selectbox('👤 Gender', encoder.categories_[1])
age = st.slider('🎂 Age', 18, 95)
balance = st.number_input('💰 Balance')
credit_score = st.number_input('📊 Credit Score')
estimated_salary = st.number_input('💵 Estimated Salary')
tenure = st.slider('📅 Tenure', 0, 10)
num_of_products = st.slider('🛍️ Number of Products', 1, 4)

has_cr_card = st.radio(
    'Does the customer have a credit card?',
    options=['No', 'Yes']
)
has_cr_card = 1 if has_cr_card == 'Yes' else 0

is_active = st.radio(
    'Is the customer an active member?',
    options=['No', 'Yes']
)
is_active = 1 if is_active == 'Yes' else 0


if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Balance': balance,
        'CreditScore': credit_score,
        'EstimatedSalary': estimated_salary,
        'Tenure': tenure,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active
    }])

    cat = ["Geography", "Gender"]
    encoded_cat = encoder.transform(
        input_data[['Geography', 'Gender']]
    )

    numeric_features = [
        'CreditScore', 'Age', 'Tenure','Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary'     
    ]
    scaled_num = scaler.transform(
        input_data[numeric_features]
    )

    final_input = np.hstack((scaled_num, encoded_cat))

    prediction = model.predict(final_input)

    churn_prob = prediction[0][0]

    if churn_prob > 0.5:
        st.markdown(f"""
            <div style="
                background-color: #ffe6e6;
                color: #b30000;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            ">
                Customer is likely to churn!<br>
                <span style="font-size:16px;">Probability: {churn_prob:.2f}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="
                background-color: #e6ffe6;
                color: #006600;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            ">
                Customer is not likely to churn!<br>
                <span style="font-size:16px;">Probability: {churn_prob:.2f}</span>
            </div>
        """, unsafe_allow_html=True)
