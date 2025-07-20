import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="DepDetect AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create a professional sidebar
st.sidebar.title("Depression Detection Tool")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
This tool uses machine learning to detect signs of depression in text messages.

**Not a medical diagnosis tool.**
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Depression Detection Tool")

# Custom CSS
st.markdown("""
<style>
    /* Force dark theme at the page level */
    .main {
        background-color: #0E1117 !important;
        color: #FFFFFF;
    }
    
    /* Force dark background on all containers */
    .block-container, .css-1544g2n {
        background-color: #0E1117 !important;
    }
    
    /* Change header color */
    .stApp header {
        background-color: #0E1117 !important;
    }
    
    /* Set default text color to white for all elements */
    body, p, li, h1, h2, h3, h4, h5, h6, label, th, td, div, span, pre, code {
        color: #FFFFFF !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF !important;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #FFFFFF !important;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #1E293B !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sentence-card {
        background-color: #262730 !important;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #374151;
    }
    .sentence-depressed {
        border-left-color: #EF4444 !important;
    }
    .sentence-normal {
        border-left-color: #10B981 !important;
    }
    .probability-high {
        font-size: 1.2rem;
        color: #EF4444 !important;
        font-weight: bold;
    }
    .probability-low {
        font-size: 1.2rem;
        color: #10B981 !important;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #9CA3AF !important;
        font-style: italic;
    }
    .info-text {
        font-size: 0.9rem;
        margin-bottom: 1rem;
        color: #FFFFFF !important;
    }
    .score-label {
        font-weight: bold;
        margin-right: 0.5rem;
        color: #9CA3AF !important;
    }
    /* Style for sentence text */
    .sentence-text {
        color: #E5E7EB !important;
        margin-bottom: 0.5rem;
    }
    /* Style for sentence analysis */
    .sentence-analysis {
        color: #9CA3AF !important;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the API URL
API_URL = "http://127.0.0.1:3000/predict"

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1 class='main-header'>🧠 DepDetect AI - Depression Detection Tool</h1>", unsafe_allow_html=True)

# Brief description
st.markdown("<p class='info-text'>This tool uses AI to analyze text for potential signs of depression. Enter your text below and click 'Analyze'.</p>", unsafe_allow_html=True)

# Disclaimer
st.markdown("<p class='disclaimer'>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional mental health diagnosis or treatment.</p>", unsafe_allow_html=True)

# Main content - single column layout
message = st.text_area("Enter your message:", height=150)

# Add model selection toggle
st.markdown("<p class='info-text'>Select model type:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    model_type = st.radio(
        "Model Type",
        options=["LSTM", "GRU"],
        horizontal=True,
        label_visibility="collapsed",
        help="LSTM models are better at capturing long-term dependencies, while GRU models are faster and simpler."
    )

if st.button("Analyze"):
    if message.strip():
        with st.spinner("Analyzing your text..."):
            try:
                # Add a small delay to show the spinner
                time.sleep(0.5)
                
                # Send the request to the API with model type
                response = requests.post(API_URL, json={"message": message, "model_type": model_type.lower()})
                
                if response.status_code == 200:
                    # Parse the response
                    result = response.json()
                    
                    # Add to history
                    st.session_state.history.append(result)
                    
                    # Extract data
                    prediction = result.get("prediction", "Unknown")
                    probability = result.get("probability", 0)
                    average_score = result.get("average_score", 0)
                    confidence = result.get("confidence", "None")
                    sentence_analysis = result.get("sentence_analysis", [])
                    model_used = result.get("model_used", "unknown")
                    
                    # Show result
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    
                    st.markdown(f"<h2 class='subheader'>Overall Analysis Result</h2>", unsafe_allow_html=True)
                    
                    if probability > 0.58:  # Using the same threshold as backend
                        st.markdown(f"<p class='probability-high'>Prediction: {prediction}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='probability-low'>Prediction: {prediction}</p>", unsafe_allow_html=True)
                    
                    # Show both scores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"<p><span class='score-label'>Weighted Score:</span> {probability:.1%}</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p><span class='score-label'>Average Score:</span> {average_score:.1%}</p>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<p><span class='score-label'>Confidence:</span> {confidence}</p>", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"<p><span class='score-label'>Model Used:</span> {model_used.upper()}</p>", unsafe_allow_html=True)
                    
                    # Add gauge visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'suffix': "%", 'font': {'size': 24}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#4338CA"},
                            'steps': [
                                {'range': [0, 58], 'color': "#D1FAE5"},  # Updated threshold to 58%
                                {'range': [58, 100], 'color': "#FEE2E2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 58  # Updated threshold to 58
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sentence-by-sentence analysis
                    if sentence_analysis:
                        st.markdown("<h3 class='subheader'>Sentence Analysis</h3>", unsafe_allow_html=True)
                        for sent in sentence_analysis:
                            sent_prob = sent.get("probability", 0)
                            sent_class = "sentence-depressed" if sent_prob > 0.58 else "sentence-normal"
                            st.markdown(f"""
                                <div class='sentence-card {sent_class}'>
                                    <p class='sentence-text'>{sent.get('sentence')}</p>
                                    <p class='sentence-analysis'><strong>Analysis:</strong> {sent.get('prediction')} ({sent_prob:.1%})</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<p class='disclaimer'>⚠️ <strong>Important Note:</strong> This is an automated analysis and should not be considered as a clinical diagnosis. If you or someone you know is struggling with mental health issues, please consult a qualified healthcare professional.</p>", unsafe_allow_html=True)
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Error connecting to the API: {str(e)}")
    else:
        st.warning("Please enter a message to analyze.")

# Show history
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Previous Analyses")
    
    # Create a DataFrame from the history
    history_data = []
    for item in st.session_state.history:
        history_data.append({
            "Message": item.get("message", "")[:50] + "..." if len(item.get("message", "")) > 50 else item.get("message", ""),
            "Prediction": item.get("prediction", ""),
            "Probability": f"{item.get('probability', 0):.1%}"
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Display the DataFrame
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Message": st.column_config.TextColumn("Message"),
            "Prediction": st.column_config.TextColumn("Prediction"),
            "Probability": st.column_config.TextColumn("Probability")
        }
    )
    
    # Initialize clear_history in session state if it doesn't exist
    if 'clear_history_clicked' not in st.session_state:
        st.session_state.clear_history_clicked = False
    
    # Add a button to clear the history with a unique key
    if st.button("Clear History", key="clear_history_button"):
        # Set the flag to true
        st.session_state.clear_history_clicked = True
        # Clear the history
        st.session_state.history = []
        # Rerun the app - using the current API
        st.rerun()
        
# Add requirements information at the bottom
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer'>To run this app, you need to have the FastAPI backend running at http://127.0.0.1:3000</p>", unsafe_allow_html=True)
