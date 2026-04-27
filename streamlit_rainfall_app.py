import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️", layout="wide")

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        obj = joblib.load("model.pkl")

        if hasattr(obj, "predict"):
            return obj

        if isinstance(obj, dict):
            for k in obj:
                if hasattr(obj[k], "predict"):
                    return obj[k]
    return None

model = load_model()

st.title("🌧️ Rainfall Prediction App")

if model:
    st.success("✅ Model Loaded")
else:
    st.error("❌ model.pkl not loaded")

# --------------------------
# Get exact training columns
# --------------------------
if model and hasattr(model, "feature_names_in_"):
    MODEL_COLUMNS = list(model.feature_names_in_)
else:
    MODEL_COLUMNS = []

st.write("### Required Features")
st.write(MODEL_COLUMNS)

# --------------------------
# Input UI
# --------------------------
data = {}

cols = st.columns(2)

for i, col in enumerate(MODEL_COLUMNS):
    with cols[i % 2]:
        if "humidity" in col.lower():
            data[col] = st.slider(col, 0, 100, 70)
        elif "direction" in col.lower():
            data[col] = st.slider(col, 0, 360, 180)
        else:
            data[col] = st.number_input(col, value=0.0)

# --------------------------
# Predict
# --------------------------
if st.button("🔍 Predict Rainfall"):

    df = pd.DataFrame([data])

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            prob = float(model.predict(df)[0])

        pred = int(prob >= 0.5)

        st.metric("Rain Probability", f"{prob:.2f}")

        if pred == 1:
            st.success("🌧️ Rain Expected")
        else:
            st.info("☀️ No Rain Expected")

    except Exception as e:
        st.error(e)