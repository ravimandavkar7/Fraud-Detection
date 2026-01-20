import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")
st.title("💳 Credit Card Fraud Detection (Demo)")

st.write(
    "This demo uses anonymized PCA features (V1–V28). "
    "Instead of manual input, select a transaction to simulate real-time fraud prediction."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    	
    model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "fraud_scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "fraud_feature_names.pkl"))
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ---------- Prepare input ----------

st.subheader("Enter Transaction Details")

V1 = st.number_input("V1", value=0.0)
V2 = st.number_input("V2", value=0.0)
V3 = st.number_input("V3", value=0.0)
V4 = st.number_input("V4", value=0.0)
V5 = st.number_input("V5", value=0.0)
Amount = st.number_input("Transaction Amount", value=0.0)

input_data = pd.DataFrame([{
    "V1": V1,
    "V2": V2,
    "V3": V3,
    "V4": V4,
    "V5": V5,
    "Amount": Amount
}])


# Reindex to match training features
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Scale
# X_scaled = scaler.transform(input_data)

# ---------- Predict ----------
if st.button("Check Fraud Risk"):
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"## Fraud Probability: **{prob:.2f}**")

    # Risk bands
    if prob >= 0.6:
        st.error("🚨 High Risk Transaction")
    elif prob >= 0.35:
        st.warning("⚠️ Medium Risk Transaction")
    else:
        st.success("✅ Low Risk Transaction")

    # Actual label (if exists)
    if "Class" in row.index:
        actual = "Fraud" if int(row["Class"]) == 1 else "Normal"
        st.caption(f"Actual label in dataset: **{actual}**")
