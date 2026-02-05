import streamlit as st
import pandas as pd
import plotly.express as px
from model import perform_inference

# Page configuration
st.set_page_config(
    page_title="Bayesian Medical Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d4156;
    }
    .sidebar .sidebar-content {
        background-color: #1e2130;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Bayesian Network Medical Diagnosis")
st.markdown("---")

# Sidebar for symptom selection
st.sidebar.header("Select Observed Symptoms")
st.sidebar.info("Tick the symptoms you are currently experiencing.")

symptoms = {
    "Fever": st.sidebar.checkbox("Fever"),
    "Cough": st.sidebar.checkbox("Cough"),
    "Fatigue": st.sidebar.checkbox("Fatigue"),
    "Sneeze": st.sidebar.checkbox("Sneezing"),
    "LossOfTaste": st.sidebar.checkbox("Loss of Taste/Smell")
}

# Convert symptoms to evidence format (0 for Yes, 1 for No)
# Note: In our model, we only add evidence for symptoms that are "Yes" 
# or we can add both. Here, let's add evidence for all selected ones.
evidence = {}
for name, value in symptoms.items():
    if value:
        evidence[name] = 0  # 0 is 'Yes' in our CPD
    else:
         # We could also set evidence[name] = 1 (No), 
         # but often in diagnosis we only provide 'present' symptoms.
         # However, for a full Bayesian update, 'No' is also informative.
         evidence[name] = 1

# Perform inference
with st.spinner("Calculating probabilities..."):
    probabilities = perform_inference(evidence)

# Main Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Probability Distribution")
    
    df = pd.DataFrame({
        "Disease": list(probabilities.keys()),
        "Probability (%)": [p * 100 for p in probabilities.values()]
    })
    
    # Sort by probability
    df = df.sort_values(by="Probability (%)", ascending=False)
    
    fig = px.bar(
        df, 
        x="Probability (%)", 
        y="Disease", 
        orientation='h',
        color="Probability (%)",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        text=df["Probability (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Diagnosis")
    
    top_disease = df.iloc[0]["Disease"]
    top_prob = df.iloc[0]["Probability (%)"]
    
    st.metric(label="Most Likely Result", value=top_disease, delta=f"{top_prob:.1f}% Confidence")
    
    st.markdown("### Risk Assessment")
    if top_disease == "Healthy":
        st.success("You appear to be healthy according to the current symptoms.")
    elif top_prob > 60:
        st.error(f"High probability of {top_disease}. Please consult a medical professional.")
    else:
        st.warning(f"Moderate probability of {top_disease}. Monitor your symptoms closely.")

    st.markdown("---")
    st.markdown("**Note:** This is a probabilistic model for educational purposes. It does not replace professional medical advice.")

# Interaction log
with st.expander("View Evidence Logic"):
    st.write("Evidence provided to the model:")
    st.json(evidence)
    st.write("Calculated Raw Probabilities:")
    st.json(probabilities)
