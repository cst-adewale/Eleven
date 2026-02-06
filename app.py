import streamlit as st
import pandas as pd
import plotly.express as px
from model import perform_inference

# Page configuration
st.set_page_config(
    page_title="Eleven | Advanced Diagnosis",
    page_icon="⏸️",
    layout="wide"
)

# Custom CSS for Premium Look & UI Refinements
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* iOS Style Toggle */
    .stCheckbox > label {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Custom Switch Container */
    .switch-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }

    /* Pause Logo Styling */
    .pause-logo {
        font-size: 3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #00d4ff;
    }
    
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #3d4156;
    }
    
    .stExpander {
        border-radius: 10px !important;
        border: 1px solid #3d4156 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header with "Pause" Symbol
st.markdown('<div class="pause-logo">⏸️ Eleven</div>', unsafe_allow_html=True)
st.markdown("---")

# iOS style toggle for mode (since streamlit doesn't support full theme swap via code easily, 
# we use this as a UI element or placeholder for user preference)
col_header, col_toggle = st.columns([8, 1])
with col_toggle:
    # A mock iOS switch using native streamlit checkbox with custom label
    # In a real app, this would trigger session state changes for CSS variables
    dark_mode = st.checkbox("Mode", value=True, help="Toggle Light/Dark Mode (Simulated)")

# Layout Swapping: Results on the Left, Symptoms on the Right
col_results, col_symptoms = st.columns([1.2, 1], gap="large")

with col_symptoms:
    st.subheader("📋 Symptom Assessment")
    st.info("Select all that apply. The model utilizes 31-point physiological data.")
    
    symptoms_input = {}
    
    # Categorized Symptoms
    with st.expander("🌡️ Systemic Indicators", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            symptoms_input["Fever"] = st.checkbox("Fever")
            symptoms_input["Fatigue"] = st.checkbox("Fatigue")
            symptoms_input["Chills"] = st.checkbox("Chills")
            symptoms_input["BodyAches"] = st.checkbox("Body Aches")
        with c2:
            symptoms_input["LossOfAppetite"] = st.checkbox("Loss of Appetite")
            symptoms_input["NightSweats"] = st.checkbox("Night Sweats")
            symptoms_input["SwollenLymphNodes"] = st.checkbox("Swollen Lymph Nodes")

    with st.expander("🫁 Respiratory Symptoms", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            symptoms_input["Cough"] = st.checkbox("Cough")
            symptoms_input["ShortnessOfBreath"] = st.checkbox("Shortness of Breath")
            symptoms_input["SoreThroat"] = st.checkbox("Sore Throat")
        with c2:
            symptoms_input["Hoarseness"] = st.checkbox("Hoarseness")
            symptoms_input["Wheezing"] = st.checkbox("Wheezing")
            symptoms_input["ChestPain"] = st.checkbox("Chest Pain")

    with st.expander("👃 Nasal & Sinus", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            symptoms_input["Sneeze"] = st.checkbox("Sneezing")
            symptoms_input["RunnyNose"] = st.checkbox("Runny Nose")
            symptoms_input["NasalCongestion"] = st.checkbox("Nasal Congestion")
        with c2:
            symptoms_input["SinusPressure"] = st.checkbox("Sinus Pressure")
            symptoms_input["EarAche"] = st.checkbox("Ear Ache")

    with st.expander("🧠 Neuro & Sensory", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            symptoms_input["Headache"] = st.checkbox("Headache")
            symptoms_input["Dizziness"] = st.checkbox("Dizziness")
            symptoms_input["Confusion"] = st.checkbox("Confusion")
            symptoms_input["LossOfTaste"] = st.checkbox("Loss of Taste")
        with c2:
            symptoms_input["LossOfSmell"] = st.checkbox("Loss of Smell")
            symptoms_input["JointPain"] = st.checkbox("Joint Pain")
            symptoms_input["MuscleStiffness"] = st.checkbox("Muscle Stiffness")

    with st.expander("🤢 Digestive & Other", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            symptoms_input["Nausea"] = st.checkbox("Nausea")
            symptoms_input["Vomiting"] = st.checkbox("Vomiting")
            symptoms_input["Diarrhea"] = st.checkbox("Diarrhea")
        with c2:
            symptoms_input["ItchyEyes"] = st.checkbox("Itchy Eyes")
            symptoms_input["WateryEyes"] = st.checkbox("Watery Eyes")
            symptoms_input["Rashes"] = st.checkbox("Rashes")

with col_results:
    # Perform inference
    with st.spinner("Analyzing physiological patterns..."):
        probabilities = perform_inference(symptoms_input)
    
    df = pd.DataFrame({
        "Ailment": list(probabilities.keys()),
        "Probability (%)": [p * 100 for p in probabilities.values()]
    }).sort_values(by="Probability (%)", ascending=False)
    
    top_ailment = df.iloc[0]["Ailment"]
    top_prob = df.iloc[0]["Probability (%)"]

    st.subheader("🔍 Analysis Results")
    st.metric(label="Primary Diagnosis Confidence", value=top_ailment, delta=f"{top_prob:.1f}% Match")

    # Probability Chart - Showing all ailments again as requested
    fig = px.bar(
        df, 
        x="Probability (%)", 
        y="Ailment", 
        orientation='h',
        color="Probability (%)",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        text=df["Probability (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        font=dict(family="Poppins")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📢 Clinical Recommendation")
    if top_ailment == "Healthy":
        st.success("Your primary indicators align with a healthy state.")
    elif top_prob > 60:
        st.error(f"High correlation with **{top_ailment}**. Immediate professional medical advice is strongly recommended.")
    elif top_prob > 30:
        st.warning(f"Moderate indicators for **{top_ailment}**. Monitor your symptoms and consult a healthcare provider.")
    else:
        st.info("The current symptom profile shows low specificity. Please provide more details or consult a professional.")

st.markdown("---")
st.caption("**Disclaimer:** This tool uses a probabilistic Bayesian Network (15-Ailment Model) for educational simulation. It is not a substitute for professional medical diagnosis.")

# Developer trace
with st.expander("🔬 Inference Trace"):
    st.write("Evidence Vector:")
    st.json({k: v for k, v in symptoms_input.items() if v})
    st.write("Calculated Probabilities:")
    st.json(probabilities)
