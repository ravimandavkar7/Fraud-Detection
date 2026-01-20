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

# ---------- Load data ----------
#@st.cache_data
#def load_data():
#    df = pd.read_csv("creditcard.csv")
#    return df
#
#df = load_data()*/
df = pd.DataFrame([
    {"V1": -1.2, "V2": 0.3, "V3": 1.5, "Amount": 50},
    {"V1": 0.1, "V2": -0.8, "V3": -1.1, "Amount": 200},
    {"V1": -2.3, "V2": 1.9, "V3": 0.4, "Amount": 120}
])

# ---------- UI ----------
st.subheader("Select a transaction")
max_idx = len(df) - 1
idx = st.slider("Transaction index", 0, max_idx, 10)

row = df.iloc[idx]

st.write("### Transaction snapshot")
st.dataframe(row.to_frame(name="Value"))

# ---------- Prepare input ----------
# Drop target column if present
if "Class" in row.index:
    X_row = row.drop("Class")
else:
    X_row = row.copy()

# Convert to DataFrame
X_df = pd.DataFrame([X_row])

# Reindex to match training features
X_df = X_df.reindex(columns=feature_names, fill_value=0)

# Scale
# X_scaled = scaler.transform(X_df)

# ---------- Predict ----------
if st.button("Predict Fraud Risk"):
    prob = model.predict_proba(X_df)[0][1]

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
