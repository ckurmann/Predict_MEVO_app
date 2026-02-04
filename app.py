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
import gc

# CSS only once
if not st.session_state.get('css_loaded', False):
    st.markdown("""
    <style>
        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] label p {
            font-size: 16px !important;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 18px !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] .st-bt div, 
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-size: 16px !important;
        }
        div[data-testid="stExpander"] .streamlit-expanderHeader p {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
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
        section[data-testid="stSidebar"] input {
            font-size: 16px !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] div {
            font-size: 16px !important;
        }
        section[data-testid="stSidebar"] div.stButton > button p {
            font-size: 18px !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100% !important;
            border-radius: 12px !important;
            height: 4em !important;
            background-color: #1e40af !important;
            color: white !important;
            border: none !important;
            margin-top: 20px !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #1d4ed8 !important;
            border: 1px solid white !important;
        }
        @media (max-width: 768px) {
            section[data-testid="stSidebar"].mobile-hide {
                transform: translateX(-100%) !important;
                transition: transform 0.3s ease !important;
            }
            [data-testid="collapsedControl"] {
                z-index: 9999 !important;
                opacity: 1 !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    st.session_state.css_loaded = True

st.set_page_config(
    page_title="MDVO Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize ALL session state variables
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = True
if 'last_input_hash' not in st.session_state:
    st.session_state.last_input_hash = None
if 'probs' not in st.session_state:
    st.session_state.probs = None
if 'ci_lower' not in st.session_state:
    st.session_state.ci_lower = None
if 'ci_upper' not in st.session_state:
    st.session_state.ci_upper = None
if 'plot_fig' not in st.session_state:
    st.session_state.plot_fig = None
if 'last_computed_hash' not in st.session_state:
    st.session_state.last_computed_hash = None

# Force sidebar render
st.sidebar.markdown("#")

# CACHED FUNCTIONS WITH MEMORY LIMITS
@st.cache_resource
def load_model():
    clf = joblib.load('no_dominant_m2_24h_nihss_cpu.pkl')
    return clf

@st.cache_data(ttl=3600, max_entries=50)
def create_input_data(age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
                     antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
                     hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
                     diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk):
    return np.array([[
        age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
        antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
        hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
        diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk
    ]])

@st.cache_data(ttl=3600, max_entries=100)
def calculate_probs_ci(probs):
    n_eff = 500
    se = np.sqrt(probs * (1 - probs) / n_eff)
    ci_lower = np.maximum(0, probs - 1.96 * se)
    ci_upper = np.minimum(1, probs + 1.96 * se)
    return ci_lower, ci_upper

@st.cache_data(ttl=3600, max_entries=10)
def create_plot(probs, ci_lower, ci_upper, _image_path="Fig2_probabilites_good_outcome.png"):
    try:
        img = mpimg.imread(_image_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, aspect='auto')
        x_mean = 110 + probs * 800
        x_lower = 110 + ci_lower * 800
        x_upper = 110 + ci_upper * 800
        ax.axvspan(x_lower, x_upper, color='red', alpha=0.3, ymin=0.12)
        ax.axvline(x_mean, color='red', linewidth=2, linestyle='--', ymin=0.12)
        ax.axis('off')
        plt.close('all')
        return fig
    except:
        plt.close('all')
        return None

# Load model
clf = load_model()

# Warning/Disclaimer
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h4 style='color: #f59e0b; font-size: 18px; margin: 0 0 8px 0; line-height: 1.2;'>
            Research and Education Use Only.<br>
            This tool provides research predictions. Consult healthcare professionals and treatment guidelines for individual patient care.
        </h4>
    </div>
""", unsafe_allow_html=True)

# Title
st.markdown(f"""
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
""", unsafe_allow_html=True)

#--Predictors--
st.sidebar.subheader("Baseline ")
age = st.sidebar.number_input("Age", 18, 100, 72, 1, format="%d")

sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
sex_numeric = 0 if sex == "Male" else 1 

nihss = st.sidebar.number_input("NIHSS at admission", 0, 42, 6, 1, format="%d")
prestroke_mrs = st.sidebar.number_input("Prestroke mRS", 0, 6, 0, 1, format="%d")
glucose = st.sidebar.number_input("Blood Glucose at admission (mmol/L)", 0.0, 40.0, 6.6, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Imaging")
vessel_options = {"Non-/Co-dominant M2": 4, "M3 and more distal": 5, "A1": 6, "A2 and more distal": 7, "P1": 10, "P2 and more distal": 11}
occluded_vessel = st.sidebar.selectbox("Occluded Vessel", options=list(vessel_options.keys()), index=0)
vessel_numeric = vessel_options[occluded_vessel]
tissue_at_risk = st.sidebar.number_input("Tissue at risk (Tmax>6s, ml)", 0.0, 500.0, 30.0, 0.1)
onset_to_img = st.sidebar.number_input("Time from onset to imaging (min)", 0, 2000, 210, 1, format="%d")

st.sidebar.markdown("---")
st.sidebar.subheader("Intravenous Thrombolysis")
ivt_selection = st.sidebar.radio(
    label="", options=["No", "Yes"], index=0, horizontal=True, label_visibility="collapsed"
)
ivt_numeric = 1 if ivt_selection == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.subheader("Medical History & Medication")
antiplatelets_numeric = 1 if st.sidebar.checkbox("Antiplatelets") else 0
anticoagulants_numeric = 1 if st.sidebar.checkbox("Anticoagulants") else 0
hist_stroke_numeric = 1 if st.sidebar.checkbox("History of stroke") else 0
hist_tia_numeric = 1 if st.sidebar.checkbox("History of TIA") else 0
aht_numeric = 1 if st.sidebar.checkbox("Arterial Hypertension") else 0
diabetes_numeric = 1 if st.sidebar.checkbox("Diabetes Mellitus") else 0
af_numeric = 1 if st.sidebar.checkbox("Atrial Fibrillation") else 0

# CACHED input data + change detection
input_data = create_input_data(age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
                              antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
                              hist_stroke_numeric, hist_tia_numeric, aht_numeric,
                              diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk)

current_hash = hash(str(input_data))
if st.session_state.last_input_hash != current_hash:
    st.session_state.last_input_hash = current_hash

# Predict button
if st.sidebar.button("Predict Outcome", use_container_width=True):
    st.session_state.prediction_made = True
    if st.session_state.show_sidebar:
        st.session_state.show_sidebar = False
        st.components.v1.html("""
        <script>
            setTimeout(() => {
                if (window.innerWidth <= 768) {
                    const sidebar = parent.document.querySelector('section[data-testid="stSidebar"]') || 
                                   window.parent.document.querySelector('section[data-testid="stSidebar"]') ||
                                   document.querySelector('section[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.classList.add('mobile-hide');
                    }
                    const collapseBtn = parent.document.querySelector('[data-testid="collapsedControl"]') ||
                                       window.parent.document.querySelector('[data-testid="collapsedControl"]');
                    if (collapseBtn) {
                        collapseBtn.click();
                    }
                }
            }, 200);
        </script>
        """, height=0)
    st.rerun()

# Instruction box
if not st.session_state.prediction_made:
    st.markdown("""
        <div style='padding: 20px; margin: 20px 0; text-align: center;'>
            <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>Enter patient data on sidebar and click Predict Outcome</p>
            <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>Tap » (top-left) to open sidebar</p>
        </div>
    """, unsafe_allow_html=True)

# Results - FIXED comparison logic
if st.session_state.prediction_made:
    if st.session_state.last_input_hash != st.session_state.last_computed_hash:
        st.session_state.probs = clf.predict_proba(input_data)[0, 1]
        st.session_state.ci_lower, st.session_state.ci_upper = calculate_probs_ci(st.session_state.probs)
        st.session_state.plot_fig = create_plot(st.session_state.probs, st.session_state.ci_lower, st.session_state.ci_upper)
        st.session_state.last_computed_hash = st.session_state.last_input_hash
    
    probs = st.session_state.probs
    ci_lower = st.session_state.ci_lower
    ci_upper = st.session_state.ci_upper
    
    # FIXED color back to original #e2e8f0
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <p style='font-size: 26px; color: #e2e8f0; margin-bottom: 2px;'>Predicted Probability of Excellent Early Neurological Outcome (24h NIHSS 0-2 ) with Best Medical Treatment alone:</p>
            <h1 style='font-size: 34px; color: #e2e8f0; margin: 0;'><strong>{probs:.1%}</strong> <span style='font-size: 34px;'>(95% CI: {ci_lower:.1%}–{ci_upper:.1%})</span></h1>
        </div>
    """, unsafe_allow_html=True)

    # Recommendation
    if ci_lower > 0.23:
        st.markdown(f"""
            <div style='background-color: #fee2e2; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #dc2626; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 28px; color: #dc2626; margin: 0; font-weight: bold;'>EVT Not Recommended</h2>
                <p style='color: #991b1b; font-size: 20px; margin-top: 8px;'>HTE analysis showed clinical harm of EVT</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: #f8fafc; padding: 20px; border-radius: 12px; 
                border-left: 6px solid #64748b; margin: 20px 0; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='font-size: 28px; color: #334155; margin: 0 0 8px 0; font-weight: bold;'>Consider EVT</h2>
                <p style='font-size: 20px; color: #475569; margin: 0; font-weight: normal;'>HTE analysis showed statistically non-significant treatment benefit of EVT</p>
            </div>
        """, unsafe_allow_html=True)
        
    # Lazy plot from session state + memory cleanup
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.plot_fig:
            st.pyplot(st.session_state.plot_fig)
        else:
            st.warning("Prediction visualization image not found.")
    
    gc.collect()

# Info section
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

gc.collect()


# %%
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import streamlit as st
# import numpy as np
# import joblib

# # CSS and page config FIRST (static content)
# st.markdown("""
# <style>
#     /* SIDEBAR STYLING */
#     section[data-testid="stSidebar"] label p {
#         font-size: 16px !important;
#     }
#     section[data-testid="stSidebar"] h3 {
#         font-size: 18px !important;
#         font-weight: 700 !important;
#     }
#     section[data-testid="stSidebar"] .st-bt div, 
#     section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
#         font-size: 16px !important;
#     }
#     div[data-testid="stExpander"] .streamlit-expanderHeader p {
#         font-size: 20px !important;
#         font-weight: 600 !important;
#     }
#     div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {
#         font-size: 20px !important;
#         line-height: 1.6 !important;
#     }
#     div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] strong {
#         font-size: 22px !important;
#         font-weight: 700 !important;
#     }
#     div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li {
#         font-size: 20px !important;
#         line-height: 1.6 !important;
#     }
#     section[data-testid="stSidebar"] input {
#         font-size: 16px !important;
#     }
#     section[data-testid="stSidebar"] div[data-baseweb="select"] div {
#         font-size: 16px !important;
#     }
#     section[data-testid="stSidebar"] div.stButton > button p {
#         font-size: 18px !important;
#         font-weight: 700 !important;
#     }
#     section[data-testid="stSidebar"] div.stButton > button {
#         width: 100% !important;
#         border-radius: 12px !important;
#         height: 4em !important;
#         background-color: #1e40af !important;
#         color: white !important;
#         border: none !important;
#         margin-top: 20px !important;
#     }
#     section[data-testid="stSidebar"] div.stButton > button:hover {
#         background-color: #1d4ed8 !important;
#         border: 1px solid white !important;
#     }
#     @media (max-width: 768px) {
#         section[data-testid="stSidebar"].mobile-hide {
#             transform: translateX(-100%) !important;
#             transition: transform 0.3s ease !important;
#         }
#         [data-testid="collapsedControl"] {
#             z-index: 9999 !important;
#             opacity: 1 !important;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# st.set_page_config(
#     page_title="MDVO Predictor", 
#     layout="wide", 
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'prediction_made' not in st.session_state:
#     st.session_state.prediction_made = False
# if 'show_sidebar' not in st.session_state:
#     st.session_state.show_sidebar = True

# # Force sidebar render
# st.sidebar.markdown("#")

# # CACHED FUNCTIONS
# @st.cache_resource
# def load_model():
#     clf = joblib.load('no_dominant_m2_24h_nihss_cpu.pkl')
#     return clf

# @st.cache_data
# def create_input_data(age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
#                      antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
#                      hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
#                      diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk):
#     return np.array([[
#         age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
#         antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
#         hist_stroke_numeric, hist_tia_numeric, aht_numeric, 
#         diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk
#     ]])

# @st.cache_data
# def create_plot(probs, ci_lower, ci_upper, _image_path="Fig2_probabilites_good_outcome.png"):
#     try:
#         img = mpimg.imread(_image_path)
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.imshow(img, aspect='auto')
#         x_mean = 110 + probs * 800
#         x_lower = 110 + ci_lower * 800
#         x_upper = 110 + ci_upper * 800
#         ax.axvspan(x_lower, x_upper, color='red', alpha=0.3, ymin=0.12)
#         ax.axvline(x_mean, color='red', linewidth=2, linestyle='--', ymin=0.12)
#         ax.axis('off')
#         return fig
#     except:
#         return None

# # Load model
# clf = load_model()

# # Warning/Disclaimer
# st.markdown("""
#     <div style='text-align: center; margin-bottom: 20px;'>
#         <h4 style='color: #f59e0b; font-size: 18px; margin: 0 0 8px 0; line-height: 1.2;'>
#             Research and Education Use Only.<br>
#             This tool provides research predictions. Consult healthcare professionals and treatment guidelines for individual patient care.
#         </h4>
#     </div>
# """, unsafe_allow_html=True)

# # Title
# st.markdown(f"""
#     <div style="
#         background-color: rgba(255, 255, 255, 0.05);
#         padding: 35px 25px; 
#         border-radius: 15px; 
#         margin-left: auto;
#         margin-right: auto;
#         margin-bottom: 25px;
#         border: 1px solid rgba(226, 232, 240, 0.2);
#         display: flex;
#         flex-direction: column;
#         align-items: center;
#         justify-content: center;
#         text-align: center;
#         width: 100%;
#     ">
#         <h1 style="
#             font-size: 36px; 
#             color: #f8fafc; 
#             margin: 0px 0px 15px 0px !important; 
#             font-weight: 800;
#             letter-spacing: -0.5px;
#             line-height: 1.2;
#             text-align: center;
#             display: block;
#             width: 100%;
#         ">
#             TabPFN Model to Predict Treatment Response to EVT in MDVO
#         </h1>
#         <p style="
#             color: #f8fafc; 
#             font-size: 20px; 
#             font-style: italic;
#             margin: 0px !important;
#             font-weight: 400;
#             text-align: center;
#             display: block;
#             width: 100%;
#         ">
#             Kurmann CC et al. Prediction of Differential Treatment Response to EVT in MDVO Patients: A DISTAL Subanalysis. 2026.
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# #--Predictors--
# st.sidebar.subheader("Baseline ")
# age = st.sidebar.number_input("Age", 18, 100, 72, 1, format="%d")

# sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
# sex_numeric = 0 if sex == "Male" else 1 

# nihss = st.sidebar.number_input("NIHSS at admission", 0, 42, 6, 1, format="%d")
# prestroke_mrs = st.sidebar.number_input("Prestroke mRS", 0, 6, 0, 1, format="%d")
# glucose = st.sidebar.number_input("Blood Glucose at admission (mmol/L)", 0.0, 40.0, 6.6, 0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Imaging")
# vessel_options = {"Non-/Co-dominant M2": 4, "M3 and more distal": 5, "A1": 6, "A2 and more distal": 7, "P1": 10, "P2 and more distal": 11}
# occluded_vessel = st.sidebar.selectbox("Occluded Vessel", options=list(vessel_options.keys()), index=0)
# vessel_numeric = vessel_options[occluded_vessel]
# tissue_at_risk = st.sidebar.number_input("Tissue at risk (Tmax>6s, ml)", 0.0, 500.0, 30.0, 0.1)
# onset_to_img = st.sidebar.number_input("Time from onset to imaging (min)", 0, 2000, 210, 1, format="%d")

# st.sidebar.markdown("---")
# st.sidebar.subheader("Intravenous Thrombolysis")
# ivt_selection = st.sidebar.radio(
#     label="", options=["No", "Yes"], index=0, horizontal=True, label_visibility="collapsed"
# )
# ivt_numeric = 1 if ivt_selection == "Yes" else 0

# st.sidebar.markdown("---")
# st.sidebar.subheader("Medical History & Medication")
# antiplatelets_numeric = 1 if st.sidebar.checkbox("Antiplatelets") else 0
# anticoagulants_numeric = 1 if st.sidebar.checkbox("Anticoagulants") else 0
# hist_stroke_numeric = 1 if st.sidebar.checkbox("History of stroke") else 0
# hist_tia_numeric = 1 if st.sidebar.checkbox("History of TIA") else 0
# aht_numeric = 1 if st.sidebar.checkbox("Arterial Hypertension") else 0
# diabetes_numeric = 1 if st.sidebar.checkbox("Diabetes Mellitus") else 0
# af_numeric = 1 if st.sidebar.checkbox("Atrial Fibrillation") else 0

# # CACHED input data
# input_data = create_input_data(age, sex_numeric, onset_to_img, nihss, prestroke_mrs, 
#                               antiplatelets_numeric, anticoagulants_numeric, ivt_numeric,
#                               hist_stroke_numeric, hist_tia_numeric, aht_numeric,
#                               diabetes_numeric, af_numeric, glucose, vessel_numeric, tissue_at_risk)

# # Predict button
# if st.sidebar.button("Predict Outcome", use_container_width=True):
#     st.session_state.prediction_made = True
#     if st.session_state.show_sidebar:  # Only collapse once
#         st.session_state.show_sidebar = False
#         st.components.v1.html("""
#         <script>
#             setTimeout(() => {
#                 if (window.innerWidth <= 768) {
#                     const sidebar = parent.document.querySelector('section[data-testid="stSidebar"]') || 
#                                    window.parent.document.querySelector('section[data-testid="stSidebar"]') ||
#                                    document.querySelector('section[data-testid="stSidebar"]');
#                     if (sidebar) {
#                         sidebar.classList.add('mobile-hide');
#                     }
#                     const collapseBtn = parent.document.querySelector('[data-testid="collapsedControl"]') ||
#                                        window.parent.document.querySelector('[data-testid="collapsedControl"]');
#                     if (collapseBtn) {
#                         collapseBtn.click();
#                     }
#                 }
#             }, 200);
#         </script>
#         """, height=0)
#     st.rerun()

# # Instruction box
# if not st.session_state.prediction_made:
#     st.markdown("""
#         <div style='padding: 20px; margin: 20px 0; text-align: center;'>
#             <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>Enter patient data on sidebar and click Predict Outcome</p>
#             <p style='color: #e2e8f0; margin: 10px 0 0 0; font-size: 22px;'>Tap » (top-left) to open sidebar</p>
#         </div>
#     """, unsafe_allow_html=True)

# # Results
# if st.session_state.prediction_made:
#     probs = clf.predict_proba(input_data)[0, 1]
#     n_eff = 500
#     se = np.sqrt(probs * (1 - probs) / n_eff)
#     ci_lower = np.maximum(0, probs - 1.96 * se)
#     ci_upper = np.minimum(1, probs + 1.96 * se)
    
#     # Probability display
#     st.markdown(f"""
#         <div style='text-align: center; padding: 20px;'>
#             <p style='font-size: 26px; color: #e2e8f0; margin-bottom: 2px;'>Predicted Probability of Excellent Early Neurological Outcome (24h NIHSS 0-2 ) with Best Medical Treatment alone:</p>
#             <h1 style='font-size: 34px; color: #e2e8f0; margin: 0;'><strong>{probs:.1%}</strong> <span style='font-size: 34px;'>(95% CI: {ci_lower:.1%}–{ci_upper:.1%})</span></h1>
#         </div>
#     """, unsafe_allow_html=True)

#     # Recommendation
#     if ci_lower > 0.23:
#         st.markdown(f"""
#             <div style='background-color: #fee2e2; padding: 20px; border-radius: 12px; 
#                 border-left: 6px solid #dc2626; margin: 20px 0; text-align: center;
#                 box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
#                 <h2 style='font-size: 28px; color: #dc2626; margin: 0; font-weight: bold;'>EVT Not Recommended</h2>
#                 <p style='color: #991b1b; font-size: 20px; margin-top: 8px;'>HTE analysis showed clinical harm of EVT</p>
#             </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#             <div style='background-color: #f8fafc; padding: 20px; border-radius: 12px; 
#                 border-left: 6px solid #64748b; margin: 20px 0; text-align: center;
#                 box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
#                 <h2 style='font-size: 28px; color: #334155; margin: 0 0 8px 0; font-weight: bold;'>Consider EVT</h2>
#                 <p style='font-size: 20px; color: #475569; margin: 0; font-weight: normal;'>HTE analysis showed statistically non-significant treatment benefit of EVT</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#     # Cached plot
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         plot_fig = create_plot(probs, ci_lower, ci_upper)
#         if plot_fig:
#             st.pyplot(plot_fig)
#         else:
#             st.warning("Prediction visualization image not found.")

# # Info section
# st.markdown("---")
# with st.expander("More information about this model"):
#     st.markdown("""
#     **Model** 
#     - TabPFN-based classifier trained on patients from a local Stroke Registry (Inselspital, University Hospital Bern, Switzerland) with medium or distal vessel occlusions (MDVO), validated on patients from the randomized controlled DISTAL trial.  
                
#     **Recommendation**
#     - Recommendations regarding EVT are derived from predictive Heterogeneity of Treatment Effect (HTE) analysis from patients of the DISTAL trial.

#     **Confidence intervals (CI)**
#     - The 95% CI are derived using bootstrapping with 1000 iterations.  

#     Use in conjunction with clinical expertise and current guideline recommendations.
#     """)

