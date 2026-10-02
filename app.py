import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("🩺 Breast Cancer Prediction System")
st.write("Enter tumor feature values below to predict whether a tumor is **Malignant** or **Benign**.")

# Safely load model from joblib file
@st.cache_resource
def load_model():
    artifact = joblib.load("models/breast_cancer_model.joblib")
    
    # Extract estimator if saved inside a dictionary
    if isinstance(artifact, dict):
        if 'model' in artifact:
            return artifact['model']
        elif 'best_model' in artifact:
            return artifact['best_model']
        else:
            for val in artifact.values():
                if hasattr(val, 'predict'):
                    return val
    return artifact

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load dataset for baseline reference values
@st.cache_data
def get_dataset():
    try:
        return pd.read_csv("full_dataset.csv")
    except Exception:
        return None

df = get_dataset()

# Extract exact features expected by the trained model
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif df is not None:
    feature_names = [col for col in df.columns if col not in ['target', 'diagnosis', 'Unnamed: 0']]
else:
    st.error("Could not determine model feature names.")
    st.stop()

st.subheader("Input Patient Tumor Details")

# Generate form inputs based on model's exact expected features
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    
    # Calculate default initial value
    default_val = 0.0
    if df is not None and feature in df.columns:
        default_val = float(df[feature].mean())
    elif df is not None:
        clean_feature = feature.replace("_", " ").strip()
        matching = [c for c in df.columns if c.replace("_", " ").strip() == clean_feature]
        if matching:
            default_val = float(df[matching[0]].mean())

    input_data[feature] = col.number_input(
        label=feature.replace("_", " ").title(),
        value=default_val,
        format="%.4f"
    )

# Prediction button
if st.button("🔍 Predict Diagnosis", use_container_width=True):
    # Pass DataFrame with exact column names expected by model
    input_df = pd.DataFrame([input_data])
    
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(input_df)[0]
            confidence = prob[1] if prediction == 1 else prob[0]
        except Exception:
            pass

    st.markdown("---")
    if prediction == 1:
        st.error("### Result: **Malignant (Cancerous)**")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.success("### Result: **Benign (Non-Cancerous)**")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence * 100:.2f}%")