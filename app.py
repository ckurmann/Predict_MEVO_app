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


# CSS for larger expander title, text, and sidebar elements
st.markdown("""
<style>
    /* SIDEBAR STYLING */
    /* font size for sidebar labels (Age, Sex, etc.) */
    section[data-testid="stSidebar"] label p {
        font-size: 16px !important;
    }

    /* font size for sidebar subheaders */
    section[data-testid="stSidebar"] h3 {
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    /* font size for radio buttons and checkboxes text */
    section[data-testid="stSidebar"] .st-bt div, 
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 16px !important;
    }

    /* MAIN PAGE EXPANDER STYLING */
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
            
    /* Unify font size of the actual values inside all sidebar inputs (numbers and dropdowns) */
    section[data-testid="stSidebar"] input {
        font-size: 16px !important; /* Matches your labels */
    }

    /* Target the dropdown text (Male/Female) specifically */
    section[data-testid="stSidebar"] div[data-baseweb="select"] div {
        font-size: 16px !important;
    }
            
    /* Target the button AND the paragraph tag inside it for the font size */
    section[data-testid="stSidebar"] div.stButton > button p {
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    /* Style the button container itself */
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        border-radius: 12px !important;
        height: 4em !important; /* Increased height to fit larger text */
        background-color: #1e40af !important;
        color: white !important;
        border: none !important;
        margin-top: 20px !important;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #1d4ed8 !important;
        border: 1px solid white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* NEW: Mobile auto-collapse after prediction */
    @media (max-width: 768px) {
        /* Smooth transition for sidebar collapse */
        section[data-testid="stSidebar"] {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Ensure collapse button is always visible on mobile */
        [data-testid="collapsedControl"] {
            z-index: 9999 !important;
            opacity: 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False


# Warning/Disclaimer
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h4 style='color: #f59e0b; font-size: 18px; margin: 0 0 8px 0; line-height: 1.2;'>
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

# Title and Reference Styling
st.html(f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.05);
        padding: 35px 25px; 
        border-radius: 15px; 
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 25px;
        border: 1px solid rgba(226, 232, 240, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
    ">
        <h1 style="
            font-size: 36px; 
            color: #f8fafc; 
            margin: 0px 0px 15px 0px !important; 
            font-weight: 800;
            letter-spacing: -0.5px;
            line-height: 1.2;
            text-align: center;
            display: block;
            width: 100%;
        ">
            TabPFN Model to Predict Treatment Response to EVT in MDVO
        </h1>
        <p style="
            color: #f8fafc; 
            font-size: 20px; 
            font-style: italic;
            margin: 0px !important;
            font-weight: 400;
            text-align: center;
            display: block;
            width: 100%;
        ">
            Kurmann CC et al. Prediction of Differential Treatment Response to EVT in MDVO Patients: A DISTAL Subanalysis. 2026.
        </p>
    </div>
""")

# Set webpage configurations
st.set_page_config(
    page_title="MDVO Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)


#--Predictors--
# 1. Age
st.sidebar.subheader("Baseline ")
age = st.sidebar.number_input("Age", 18, 100, 72, 1, format="%d")

# 2. Sex
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
sex_numeric = 0 if sex == "Male" else 1 

# 3. NIHSS
nihss = st.sidebar.number_input("NIHSS at admission", 0, 42, 6, 1, format="%d")

# 4. Prestroke mRS
prestroke_mrs = st.sidebar.number_input("Prestroke mRS", 0, 6, 0, 1, format="%d")

# 5. Blood Glucose
glucose = st.sidebar.number_input("Blood Glucose at admission (mmol/L)", 0.0, 40.0, 6.6, 0.1)

# 6. Occluded Vessel
st.sidebar.markdown("---")
st.sidebar.subheader("Imaging")
vessel_options = {"Non-/Co-dominant M2": 4, "M3 and more distal": 5, "A1": 6, "A2 and more distal": 7, "P1": 10, "P2 and more distal": 11}
occluded_vessel = st.sidebar.selectbox("Occluded Vessel", options=list(vessel_options.keys()), index=0)
vessel_numeric = vessel_options[occluded_vessel]

# 7. Tissue at risk
tissue_at_risk = st.sidebar.number_input("Tissue at risk (Tmax>6s, ml)", 0.0, 500.0, 30.0, 0.1)

# 8. Time from onset to imaging
onset_to_img = st.sidebar.number_input("Time from onset to imaging (min)", 0, 2000, 210, 1, format="%d")

# 9. IVT Section
st.sidebar.markdown("---")
st.sidebar.subheader("Intravenous Thrombolysis")
ivt_selection = st.sidebar.radio(
    label="",               # This must be a string
    options=["No", "Yes"],  # This is your list
    index=0, 
    horizontal=True,
    label_visibility="collapsed" # This hides the extra empty space
)
ivt_numeric = 1 if ivt_selection == "Yes" else 0

# 10. All other clickable boxes
st.sidebar.markdown("---")
st.sidebar.subheader("Medical History & Medication")
antiplatelets_numeric = 1 if st.sidebar.checkbox("Antiplatelets") else 0
anticoagulants_numeric = 1 if st.sidebar.checkbox("Anticoagulants") else 0
hist_stroke_numeric = 1 if st.sidebar.checkbox("History of stroke") else 0
hist_tia_numeric = 1 if st.sidebar.checkbox("History of TIA") else 0
aht_numeric = 1 if st.sidebar.checkbox("Arterial Hypertension") else 0
diabetes_numeric = 1 if st.sidebar.checkbox("Diabetes Mellitus") else 0
af_numeric = 1 if st.sidebar.checkbox("Atrial Fibrillation") else 0

# The array below keeps the exact sequence your model expects
input_data = np.array([[
    age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
    antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
    hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
    diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk
]])


# Predict button
if st.sidebar.button("Predict Outcome", use_container_width=True):
    st.session_state.prediction_made = True
    
    # Auto-collapse sidebar on mobile only
    st.components.v1.html("""
    <script>
    if (window.innerWidth <= 768) {
        setTimeout(function() {
            const btn = parent.document.querySelector('[data-testid="collapsedControl"]');
            if (btn && !btn.matches(':disabled')) {
                btn.click();
            }
        }, 150);
    }
    </script>
    """, height=0)


# Instruction box
if not st.session_state.prediction_made:
    st.markdown("""
        <div style='
            padding: 20px; margin: 20px 0; text-align: center;
        '>
            <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>
                Enter patient data on sidebar and click Predict Outcome
            </p>
            <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>
                Tap <span style='font-size: 32px; '>»</span> (top-left) to open sidebar
            </p>
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
            <p style='font-size: 26px; color: #e2e8f0; margin-bottom: 2px;'>
                Predicted Probability of Excellent Early Neurological Outcome (24h NIHSS 0-2 ) with Best Medical Treatment alone:
            </p>
            <h1 style='font-size: 34px; color: #e2e8f0; margin: 0;'>
                <strong>{probs:.1%}</strong> <span style='font-size: 34px;'>(95% CI: {ci_lower:.1%}–{ci_upper:.1%})</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)


    # Recommendation
    if ci_lower > 0.23:
        st.markdown(f"""
            <div style='background-color: #fee2e2; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #dc2626; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 28px; color: #dc2626; margin: 0; font-weight: bold;'>
                    EVT Not Recommended
                </h2>
                <p style='color: #991b1b; font-size: 20px; margin-top: 8px;'>
                    HTE analysis showed clinical harm of EVT
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
            <div style='background-color: #f8fafc; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #64748b; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 28px; color: #334155; margin: 0 0 8px 0; font-weight: bold;'>
                    Consider EVT
                </h2>
                <p style='font-size: 20px; color: #475569; margin: 0; font-weight: normal;'>
                    HTE analysis showed statistically non-significant treatment benefit of EVT
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    # Plot
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
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
        except:
            st.warning("Prediction visualization image not found.")


# Info section at bottom of page
st.markdown("---")

with st.expander("More information about this model"):
    st.markdown("""
    **Model** 
    - TabPFN-based classifier trained on patients from a local Stroke Registry (Inselspital, University Hospital Bern, Switzerland) with medium or distal vessel occlusions (MDVO), validated on patients from the randomized controlled DISTAL trial.  
                
    **Recommendation**
    - Recommendations regarding EVT are derived from predictive Heterogeneity of Treatment Effect (HTE) analysis from patients of the DISTAL trial.

    **Confidence intervals (CI)**
    - The 95% CI are derived using bootstrapping with 1000 iterations.  

    Use in conjunction with clinical expertise and current guideline recommendations.
    """)
