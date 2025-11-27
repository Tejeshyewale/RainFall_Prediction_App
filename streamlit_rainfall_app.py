"""
Advanced Streamlit UI for Rainfall Prediction (Rain / No Rain)
File: streamlit_rainfall_app.py

How to run:
1. pip install -r requirements.txt
   (requirements: streamlit pandas numpy scikit-learn joblib shap plotly matplotlib altair)
2. Put your trained classifier pipeline as `model.pkl` in the same folder OR upload a model through the sidebar.
3. streamlit run streamlit_rainfall_app.py

Features:
- Sidebar to load model or upload a saved model (joblib/pickle)
- Choose between single-record (form) prediction and bulk CSV prediction
- Auto-generated input widgets based on feature schema or uploaded sample CSV
- Prediction probability, threshold slider, and final class
- Model explainability: SHAP force plot / summary (falls back if SHAP not available)
- Feature importance plot (if model supports `feature_importances_`)
- Batch predictions with downloadable CSV
- Visualisations: histogram / time-series / map preview (if lat/lon present)
- Custom CSS for a clean UI and layout with columns

Notes:
- The app assumes your model expects a DataFrame with numeric columns matching `feature_columns`.
- If your model includes preprocessing in a sklearn `Pipeline`, predictions will be correct directly.

"""

import io
import os
import base64
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ML libs
import joblib
from sklearn.base import BaseEstimator

# Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Optional: shap for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# Utility helpers
# ---------------------------

@st.cache_resource
def load_model_from_path(path: str) -> BaseEstimator:
    return joblib.load(path)

@st.cache_resource
def load_model_from_bytes(b: bytes) -> BaseEstimator:
    return joblib.load(io.BytesIO(b))


def download_link(df: pd.DataFrame, filename: str = "predictions.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ---------------------------
# Default feature schema (edit to match your model)
# ---------------------------
DEFAULT_FEATURES = [
    'temperature',      # in deg C
    'humidity',         # in %
    'pressure',         # in hPa
    'wind_speed',       # m/s
    'wind_direction',   # degrees
    'cloud_cover',      # oktas or %
    'dew_point',        # deg C
    # optionally: 'latitude', 'longitude', 'datetime'
]

# ---------------------------
# Streamlit layout
# ---------------------------

st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .reportview-container .main header {display: none}
    .stApp { background: linear-gradient(180deg,#e9f0fb, #ffffff);}    
    .logo {font-size:30px; font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("# 🌧️ Rainfall Prediction — Interactive App")
    st.write("Upload your trained model or use the demo, enter features, and get predictions with explainability.")
with col2:
    st.image("https://images.unsplash.com/photo-1501973801540-537f08ccae7b?w=600&q=80", width=120)

# Sidebar: model, options
st.sidebar.header("Model & Data")
model_source = st.sidebar.radio("Load model", options=["Use demo model (none)", "Upload model (.pkl/.joblib)", "Load from file model.pkl"]) 

model = None
model_bytes = None

if model_source == "Upload model (.pkl/.joblib)":
    uploaded_model = st.sidebar.file_uploader("Upload your sklearn Pipeline or classifier", type=["pkl","joblib"], accept_multiple_files=False)
    if uploaded_model is not None:
        try:
            model = load_model_from_bytes(uploaded_model.read())
            st.sidebar.success("Model loaded from upload ✅")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

elif model_source == "Load from file model.pkl":
    if os.path.exists('model.pkl'):
        try:
            model = load_model_from_path('model.pkl')
            st.sidebar.success("model.pkl loaded ✅")
        except Exception as e:
            st.sidebar.error(f"Failed to load model.pkl: {e}")
    else:
        st.sidebar.info("model.pkl not found in working directory")

else:
    st.sidebar.info("Using demo mode: No model loaded. You can still explore the UI and CSV upload for batch predictions.")

# Input mode
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Mode", options=["Single record (form)", "Batch CSV prediction"]) 

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Decision threshold (probability for 'Rain')", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Allow user to provide a sample CSV to auto-detect features
sample_csv = st.sidebar.file_uploader("Optional: Upload a sample CSV (to detect feature columns)", type=['csv'])
sample_df = None
if sample_csv is not None:
    try:
        sample_df = pd.read_csv(sample_csv)
        st.sidebar.success("Sample CSV loaded — feature columns will be detected")
    except Exception as e:
        st.sidebar.error(f"Failed to read sample CSV: {e}")

# Let user set feature columns (editable)
st.sidebar.header("Feature columns")
if sample_df is not None:
    feature_columns = list(sample_df.columns)
else:
    feature_columns = DEFAULT_FEATURES.copy()

feature_text = st.sidebar.text_area("Columns (comma separated)", value=", ".join(feature_columns), height=120)
feature_columns = [c.strip() for c in feature_text.split(',') if c.strip()]

# ---------------------------
# Main: single or batch UI
# ---------------------------

if mode == "Single record (form)":
    st.subheader("Single record prediction")
    with st.form(key='single-form'):
        cols = st.columns(2)
        input_data = {}
        for i, feat in enumerate(feature_columns):
            col = cols[i % 2]
            # choose widget by name heuristics
            if 'time' in feat.lower() or 'date' in feat.lower():
                input_data[feat] = col.text_input(feat, value="2025-01-01 12:00:00")
            elif 'lat' in feat.lower() or 'lon' in feat.lower():
                input_data[feat] = col.number_input(feat, value=0.0)
            elif 'wind_direction' in feat.lower() or 'direction' in feat.lower():
                input_data[feat] = col.slider(feat, 0, 360, 180)
            else:
                input_data[feat] = col.number_input(feat, value=0.0, format="%.4f")

        submit = st.form_submit_button("Predict")

    if submit:
        df_input = pd.DataFrame([input_data])
        df_input = ensure_numeric(df_input, feature_columns)

        if model is None:
            st.warning("No model loaded — returning a demo prediction using heuristic: humidity > 70 -> Rain")
            hum = float(df_input.get('humidity', 0))
            prob = min(max((hum - 30) / 70, 0), 1)
            pred = int(prob >= threshold)
            st.metric(label="Rain probability", value=f"{prob:.2f}")
            st.write("Prediction:", "🌧️ Rain" if pred == 1 else "☀️ No rain")
        else:
            try:
                probs = model.predict_proba(df_input)[:, 1]
                pred = (probs >= threshold).astype(int)
                st.metric(label="Rain probability", value=f"{probs[0]:.3f}")
                st.write("Prediction:", "🌧️ Rain" if pred[0] == 1 else "☀️ No rain")

                # Show probabilities bar
                fig = px.bar(x=['No Rain', 'Rain'], y=[1-probs[0], probs[0]], labels={'x':'Class','y':'Probability'})
                st.plotly_chart(fig, use_container_width=True)

                # Explainability
                if SHAP_AVAILABLE:
                    try:
                        explainer = shap.Explainer(model, df_input)
                        shap_values = explainer(df_input)
                        st.subheader("SHAP explanation")
                        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
                    except Exception as e:
                        st.info(f"SHAP explanation failed: {e}")
                else:
                    st.info("SHAP not installed — install shap for feature-level explanations (pip install shap)")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.subheader("Batch CSV prediction")
    st.write("Upload a CSV with the feature columns. The app will return predictions and a downloadable CSV file.")
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=['csv'])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data")
            st.dataframe(df.head())

            # ensure features exist
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                st.warning(f"The following expected columns are missing from the uploaded CSV: {missing}")

            # keep only feature columns available
            use_cols = [c for c in feature_columns if c in df.columns]
            df_features = ensure_numeric(df.copy(), use_cols)

            if model is None:
                st.warning("No model loaded — demo heuristics will be applied (humidity-based)")
                probs = df_features.get('humidity', pd.Series(0)).apply(lambda h: min(max((float(h) - 30) / 70, 0), 1) if pd.notna(h) else 0)
                preds = (probs >= threshold).astype(int)
                out = df.copy()
                out['rain_probability'] = probs
                out['rain_pred'] = preds
            else:
                try:
                    probs = model.predict_proba(df_features.fillna(0))[:,1]
                    preds = (probs >= threshold).astype(int)
                    out = df.copy()
                    out['rain_probability'] = probs
                    out['rain_pred'] = preds
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    out = None

            if out is not None:
                st.success("Predictions ready")
                st.dataframe(out.head())
                href = download_link(out)
                st.markdown(f"[Download predictions]({href})")

                # Show basic plots
                st.subheader("Batch Visuals")
                if 'datetime' in df.columns:
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        fig = px.line(out, x='datetime', y='rain_probability', title='Rain probability over time')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

                if 'latitude' in df.columns and 'longitude' in df.columns:
                    st.map(df.rename(columns={'latitude':'lat','longitude':'lon'})[['lat','lon']].dropna())

        except Exception as e:
            st.error(f"Failed to process uploaded CSV: {e}")

# ---------------------------
# Extras: model insights
# ---------------------------

st.markdown("---")
st.header("Model insights & diagnostics")

if model is None:
    st.info("No model loaded — insights unavailable. Upload or load a model to see feature importance and diagnostics.")
else:
    try:
        st.subheader("Model type")
        st.write(type(model))

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            try:
                fi = model.feature_importances_
                cols = feature_columns[:len(fi)]
                df_fi = pd.DataFrame({'feature': cols, 'importance': fi})
                df_fi = df_fi.sort_values('importance', ascending=False)
                fig = px.bar(df_fi, x='importance', y='feature', orientation='h', title='Feature importance')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute feature importances: {e}")
        else:
            st.info("Model has no `feature_importances_`. If it's a pipeline, try loading the underlying estimator.")

        # Simple test on sample data if provided
        if sample_df is not None:
            st.subheader("Quick test on sample CSV")
            try:
                use_cols = [c for c in feature_columns if c in sample_df.columns]
                X = ensure_numeric(sample_df.copy(), use_cols)[use_cols].fillna(0)
                probs = model.predict_proba(X)[:,1]
                st.write("Pred probability (sample):", probs[:10])
                fig = px.histogram(probs, nbins=30, title='Predicted rain probability distribution')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Quick test failed: {e}")

        if SHAP_AVAILABLE:
            st.subheader("Global SHAP summary (first 200 rows)")
            try:
                explainer = shap.Explainer(model, sample_df[feature_columns].fillna(0) if sample_df is not None else None)
                sample_for_shap = None
                if sample_df is not None:
                    sample_for_shap = ensure_numeric(sample_df.copy(), feature_columns).fillna(0)[feature_columns].head(200)
                else:
                    # create a random sample placeholder
                    sample_for_shap = pd.DataFrame(np.random.normal(size=(100, len(feature_columns))), columns=feature_columns)

                shap_values = explainer(sample_for_shap)
                st.pyplot(shap.plots.beeswarm(shap_values, show=False))
            except Exception as e:
                st.info(f"SHAP global explanation failed: {e}")

    except Exception as e:
        st.error(f"Error analyzing model: {e}")

# Footer with tips
st.markdown("---")
st.caption("Tips: 1) If your pipeline contains scaling/encoding inside a sklearn Pipeline or ColumnTransformer, predictions will 'just work'. 2) For best explainability, keep a copy of the training feature names and preprocessing steps.")


# End of app
