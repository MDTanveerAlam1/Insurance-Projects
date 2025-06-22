# app.py
import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the trained model
model = joblib.load('medical_cost_model.pkl')

# Page Configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for day/night mode and colors
def local_css():
    st.markdown("""
        <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-size: 16px;
        }
        footer {
            font-size: 12px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# Sidebar
st.sidebar.title("🔧 Settings")
theme = st.sidebar.radio("Choose Theme", ["🌞 Light", "🌜 Dark"])
if theme == "🌜 Dark":
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #00b4d8;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Main Title
st.title("💼 Medical Insurance Cost Predictor")
st.subheader("🧠 Powered by Machine Learning")

st.markdown("---")
st.markdown("#### 🔍 Enter Patient Details")

# Input Form
with st.form("prediction_form"):
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Sex", ['male', 'female'])
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.selectbox("Smoker Status", ['yes', 'no'])
    region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

    submitted = st.form_submit_button("🚀 Predict")

# Prepare data and predict
if submitted:
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Medical Expense: **${prediction:.2f}**")

    st.markdown("---")
    st.markdown("📌 *This prediction is based on a multiple linear regression model.*")

# Footer with founder info
st.markdown("---")
st.markdown(
    f"""
    <footer>
        Developed by <b>MD Tanveer Alam</b> | Founder - Life With AI<br>
        📅 {datetime.datetime.now().strftime('%B %d, %Y')} | 🕒 {datetime.datetime.now().strftime('%I:%M %p')}
    </footer>
    """,
    unsafe_allow_html=True
)

