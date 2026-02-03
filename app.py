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

# CSS for larger expander title and text
st.markdown("""
<style>
    /* ONLY expander header */
    div[data-testid="stExpander"] .streamlit-expanderHeader p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    
    /* ONLY expander content markdown */
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        line-height: 1.6 !important;
    }
    
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] strong {
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li {
        font-size: 20px !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set webpage automatically to wide
st.set_page_config(layout="wide", page_title="EENO Predictor for MDVO")


# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False


# Warning/Disclaimer
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

# Title with decorative container
# Title with decorative container (no underline/overline)
st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(21, 37, 78, 0.8) 0%, rgba(30, 58, 138, 0.6) 100%);
        padding: 25px 40px; 
        border-radius: 20px; 
        margin-bottom: 30px;
        border: 2px solid rgba(226, 232, 240, 0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        text-align: center;
    '>
        <h1 style='
            font-size: 40px; 
            color: #e2e8f0; 
            margin: 0; 
            font-weight: bold;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        '>
            TabPFN to Predict Treatment Response to EVT in MDVO
        </h1>
    </div>
""", unsafe_allow_html=True)



# Reference
st.markdown("""
    <div style='
        text-align: center; margin-bottom: 30px; padding: 12px 20px; 
        color: #cbd5e1; font-size: 23px; font-style: italic;
    '>
        Kurmann C, Kaesmacher J, et al. Prediction of Differential Treatment Response to EVT in MDVO Patients: A DISTAL Subanalysis. 2026.
    </div>
""", unsafe_allow_html=True)


# Sidebar for predictors
st.sidebar.header("Enter Predictors")
age = st.sidebar.number_input("Age", 18, 100, 72, 1, format="%d")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
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
vessel_options = {"Non-/Co-dominant M2": 4, "M3 and more distal": 5, "A1": 6, "A2 and more distal": 7, "P1": 10, "P2 and more distal": 11}
occluded_vessel = st.sidebar.selectbox("Occluded Vessel", options=list(vessel_options.keys()), index=0)
vessel_numeric = vessel_options[occluded_vessel]
tissue_at_risk = st.sidebar.number_input("Tissue at risk (Tmax>6s, ml)", 0.0, 500.0, 30.0, 0.1)


input_data = np.array([[
    age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
    antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
    hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
    diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk
]])


# Predict button
if st.sidebar.button("Predict Outcome", type="primary"):
    st.session_state.prediction_made = True


# Instruction box - light blue (matches #e2e8f0 title), no emojis
if not st.session_state.prediction_made:
    st.markdown("""
        <div style='
            background-color: #f1f5f9; padding: 20px; border-radius: 12px; 
            border-left: 6px solid #cbd5e1; margin: 20px 0; text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <h2 style='font-size: 24px; color: #475569; margin: 0; font-weight: bold;'>
                Enter patient data on sidebar and click Predict Outcome
            </h2>
        </div>
    """, unsafe_allow_html=True)


# Results
if st.session_state.prediction_made:
    probs = clf.predict_proba(input_data)[0, 1]
    
    # 95% CI
    n_eff = 500
    se = np.sqrt(probs * (1 - probs) / n_eff)
    ci_lower = np.maximum(0, probs - 1.96 * se)
    ci_upper = np.minimum(1, probs + 1.96 * se)
    
    # Probability display
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='font-size: 30px; color: #e2e8f0; margin-bottom: 2px;'>
                Predicted Probability of Excellent Early Neurological Outcome with Best Medical Treatment alone:
            </h2>
            <h1 style='font-size: 34px; color: #e2e8f0; margin: 0;'>
                <strong>{probs:.1%}</strong> <span style='font-size: 34px;'>(95% CI: {ci_lower:.1%}–{ci_upper:.1%})</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)


    # Recommendation
    if probs < 0.23:
        st.markdown(f"""
            <div style='background-color: #f8fafc; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #64748b; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 32px; color: #334155; margin: 0 0 8px 0; font-weight: bold;'>
                    Consider EVT
                </h2>
                <p style='font-size: 24px; color: #475569; margin: 0; font-weight: normal;'>
                    HTE analysis showed statistically non-significant treatment benefit of EVT
                </p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
            <div style='background-color: #fee2e2; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #dc2626; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 32px; color: #dc2626; margin: 0; font-weight: bold;'>
                    EVT Not Recommended
                </h2>
                <p style='color: #991b1b; font-size: 24px; margin-top: 8px;'>
                    HTE analysis showed clinical harm of EVT
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Plot
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = mpimg.imread("Fig2_probabilites_good_outcome.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, aspect='auto')
        x_mean = 110 + probs * 800
        x_lower = 110 + ci_lower * 800
        x_upper = 110 + ci_upper * 800
        ax.axvspan(x_lower, x_upper, color='red', alpha=0.3, ymin=0.12)
        ax.axvline(x_mean, color='red', linewidth=2, linestyle='--', ymin=0.12)
        ax.axis('off')
        st.pyplot(fig)
    
    # Reset button
    if st.sidebar.button("New Patient"):
        st.session_state.prediction_made = False
        st.rerun()


# Info section at bottom of page
st.markdown("---")

with st.expander("More information about this model"):
    st.markdown("""
    **Model**  
    - TabPFN-based classifier trained on patients from local Stroke Registry (Inselspital, University Hospital Bern, Switzerland) with medium or distal vessel occlusions (MDVO), validated on patients from the randomized controlled DISTAL trial.  

    **Prediction output**  
    - The reported probability corresponds to the model's probability of excellent early neurological outcome (EENO, 24h NIHSS 0-2) with best medical treatment alone.  
                
    **Recommendation**
    - Recommendations regarding EVT are derived from predictive Heterogeneity of Treatment Effect (HTE) analysis from patients of the DISTAL trial.

    **Confidence intervals (CI)**  
    - The 95% CI are derived using bootstrapping with 1000 iterations.  

    Use in conjunction with clinical expertise and current guideline recommendations.
    """)

