# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: kurmann
#     language: python
#     name: python3
# ---

# %%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import joblib

st.set_page_config(layout="wide", page_title="EENO Predictor for MDVO")

#Warning/Disclaimer
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h4 style='color: #dc2626; font-size: 18px; margin: 0 0 8px 0; line-height: 1.2;'>
            Research and Education Use Only.<br>
            This tool provides research predictions. Consult professional healthcare professionals and treatment guidelines for individual patient care.
        </h4>
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    clf = joblib.load('no_dominant_m2_24h_nihss_cpu.pkl')
    return clf

clf = load_model()

#Title
st.markdown("""
    <h1 style='font-size: 36px; color: #e2e8f0; text-align: center; 
                margin-bottom: 20px; font-weight: bold;'>
        TabPFN to Predict Treatment Response to EVT in MDVO
    </h1>
""", unsafe_allow_html=True)

#Sidebar for predictors
st.sidebar.header("Enter Predictors")
age = st.sidebar.number_input("Age", 18, 100, 72, 1, format="%d")

sex = st.sidebar.selectbox("Sex", ["Male", "Female"],index=0)
sex_numeric = 0 if sex == "Male" else 1 

onset_to_img = st.sidebar.number_input("Time from onset to imaging (min)", 0, 2000, 210, 1, format="%d")

nihss = st.sidebar.number_input("NIHSS at admission", 0, 42, 6, 1, format="%d")

prestroke_mrs = st.sidebar.number_input("Prestroke mRS", 0, 6, 0, 1, format="%d")

antiplatelets = st.sidebar.selectbox("Antiplatelets", ["No", "Yes"], index=0)
antiplatelets_numeric = 0 if antiplatelets == "No" else 1

anticoagulants = st.sidebar.selectbox("Anticoagulants", ["No", "Yes"], index=0)
anticoagulants_numeric = 0 if anticoagulants == "No" else 1

ivt = st.sidebar.selectbox("IVT", ["No", "Yes"], index=0)
ivt_numeric = 0 if ivt == "No" else 1

hist_stroke = st.sidebar.selectbox("History of stroke", ["No", "Yes"], index=0)
hist_stroke_numeric = 0 if hist_stroke == "No" else 1

hist_tia = st.sidebar.selectbox("History of TIA", ["No", "Yes"], index=0)
hist_tia_numeric = 0 if hist_tia == "No" else 1

aht = st.sidebar.selectbox("Arterial Hypertension", ["No", "Yes"], index=0)
aht_numeric = 0 if aht == "No" else 1

diabetes = st.sidebar.selectbox("Diabetes Mellitus", ["No", "Yes"], index=0)
diabetes_numeric = 0 if diabetes == "No" else 1

af = st.sidebar.selectbox("Atrial Fibrillation", ["No", "Yes"], index=0)
af_numeric = 0 if af == "No" else 1

glucose = st.sidebar.number_input("Blood Glucose at admission (mmol/L)", 0.0, 40.0, 6.6, 0.1)

vessel_options = {
    "Non-/Co-dominant M2": 4,
    "M3 and more distal": 5,
    "A1": 6,
    "A2 and more distal": 7,
    "P1": 10,
    "P2 and more distal": 11
}
occluded_vessel = st.sidebar.selectbox(
    "Occluded Vessel", 
    options=list(vessel_options.keys()),
    index=0
)
vessel_numeric = vessel_options[occluded_vessel]  # Now works!

tissue_at_risk = st.sidebar.number_input("Tissue at risk (Tmax>6s, ml)", 0.0, 500.0, 30.0, 0.1)

# Match your model's expected input shape/order
input_data = np.array([[
    age,
    sex_numeric,
    onset_to_img,
    nihss,
    prestroke_mrs, 
    antiplatelets_numeric,
    anticoagulants_numeric,
    ivt_numeric,
    hist_stroke_numeric,
    hist_tia_numeric,
    aht_numeric, 
    diabetes_numeric,
    af_numeric,
    glucose,
    vessel_numeric,
    tissue_at_risk
]])

if st.sidebar.button("Predict Outcome", type="primary"):
    probs = clf.predict_proba(input_data)[0]
    
    # Probability display
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='font-size: 28px; color: #e2e8f0; margin-bottom: 2px;'>
                Predicted Probability of Excellent Early Neurological Outcome (BMT alone):
            </h2>
            <h1 style='font-size: 38px; color: #e2e8f0; margin: 0;'>
                <strong>{probs[1]:.1%}</strong>
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Recommendation boxes
    if probs[1] < 0.23:
        st.markdown(f"""
            <div style='
                background-color: #f8fafc; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #64748b; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <h2 style='font-size: 32px; color: #334155; margin: 0; font-weight: bold;'>
                    Consider EVT, Model Predicts Neutral Treatment Effect
                </h2>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='
                background-color: #fee2e2; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #dc2626; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <h2 style='font-size: 32px; color: #dc2626; margin: 0; font-weight: bold;'>
                    EVT Not Recommended
                </h2>
                <p style='color: #991b1b; font-size: 16px; margin-top: 8px;'>
                    Higher risk of poor outcome (90-days mRS >2)
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Centered plot with vertical line at probs[1]
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Load PNG and overlay horizontal line at probs[1]
        img = mpimg.imread("Fig2_probabilites_good_outcome.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, aspect='auto')
        ax.axvline(x=110 + probs[1]*800, color='red', linewidth=1, linestyle='--', ymin=0.12)  # 
        ax.axis('off')
        st.pyplot(fig)

