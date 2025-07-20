import streamlit as st
import pandas as pd
import requests
import time
import io
import base64
from math import ceil

# Page configuration
st.set_page_config(
    page_title="Bulk Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add a force dark mode CSS that will work regardless of Streamlit version
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
    
    /* Sidebar background */
    .css-1d391kg, .css-1lcbmhc, .sidebar .sidebar-content {
        background-color: #262730 !important;
    }
    
    /* Set default text color to white for all elements */
    body, p, li, h1, h2, h3, h4, h5, h6, label, th, td, div, span, pre, code {
        color: #FFFFFF !important;
    }
    
    /* Style main header */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF !important;
        margin-bottom: 1rem;
    }
    
    /* Style subheaders */
    .subheader, h2, h3, h4 {
        font-size: 1.5rem;
        color: #FFFFFF !important;
        margin-bottom: 1rem;
    }
    
    /* Style result card with a dark background for better contrast */
    .result-card {
        background-color: #1E293B !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Sentence analysis cards */
    .sentence-card {
        background-color: #262730 !important;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #374151;
    }
    
    .sentence-depressed {
        border-left-color: #EF4444 !important;
    }
    
    .sentence-normal {
        border-left-color: #10B981 !important;
    }
    
    /* Styling for probabilities */
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
    
    /* Style disclaimers */
    .disclaimer {
        font-size: 0.8rem;
        color: #9CA3AF !important;
        font-style: italic;
    }
    
    /* Style info text */
    .info-text {
        font-size: 0.9rem;
        margin-bottom: 1rem;
        color: #FFFFFF !important;
    }
    
    /* Style download button */
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #1E3A8A;
        color: white !important;
        text-decoration: none;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    
    /* Contact button */
    .contact-button {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background-color: #1E3A8A;
        color: white !important;
        text-decoration: none;
        border-radius: 0.3rem;
    }
    
    /* Score labels */
    .score-label {
        font-weight: bold;
        margin-right: 0.5rem;
        color: #9CA3AF !important;
    }
    
    /* Expand/collapse button */
    .expand-button {
        color: #60A5FA !important;
        cursor: pointer;
        text-decoration: underline;
    }
    
    /* Remove the black text rules */
    /* Override Streamlit's default styles */
    .stMarkdown, .stMarkdown p, .stText, .stText p {
        color: #FFFFFF !important;
    }
    
    /* Style for code blocks */
    .stCodeBlock, div.stCodeBlock > div,
    pre, code, .stCodeBlock pre, .stCodeBlock code, .stCodeBlock span {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    
    /* Style for data frames */
    .dataframe, .dataframe * {
        color: #FFFFFF !important;
    }
    
    /* Override Streamlit's default white backgrounds */
    .stApp {
        background-color: #111827 !important;
    }
    
    .css-18e3th9 {
        background-color: #111827 !important;
    }
    
    .css-1d391kg {
        background-color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# Define API URL
API_URL = "http://127.0.0.1:3000/bulk-predict"

# Initialize session state for results and pagination
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'items_per_page' not in st.session_state:
    st.session_state.items_per_page = 7

# Add this function definition after the imports and before the main code

def display_results(df):
    """Display the results with pagination and download option."""
    st.subheader("Analysis Results")
    
    # Add download button for full results
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="depression_analysis_results.csv" class="download-button">Download Full Results CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Filter for depressed users (using the 0.58 threshold)
    depressed_df = df[df['is_depressed'] == True].copy()
    depressed_count = len(depressed_df)
    
    if depressed_count > 0:
        st.markdown(f"<p><strong>Found {depressed_count} potentially depressed users</strong></p>", unsafe_allow_html=True)
        
        # Pagination setup
        items_per_page = st.session_state.items_per_page
        total_pages = ceil(depressed_count / items_per_page)
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("Previous", disabled=st.session_state.current_page <= 1):
                st.session_state.current_page -= 1
                st.rerun()
                
        with col2:
            st.markdown(f"<p style='text-align: center;'>Page {st.session_state.current_page} of {total_pages}</p>", unsafe_allow_html=True)
            
        with col3:
            if st.button("Next", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page += 1
                st.rerun()
        
        # Get current page data
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_df = depressed_df.iloc[start_idx:end_idx].copy()
        
        # Display results with sentence analysis
        for _, row in page_df.iterrows():
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                # User info and overall scores
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                col1.markdown(f"**@{row['username']}**")
                col2.markdown(f"<span class='score-label'>Weighted Score:</span> {row['depression_probability']:.1%}", unsafe_allow_html=True)
                col3.markdown(f"<span class='score-label'>Average Score:</span> {row.get('average_score', 0):.1%}", unsafe_allow_html=True)
                
                # Twitter link
                twitter_link = f'https://twitter.com/{row["username"]}'
                col4.markdown(f'<a href="{twitter_link}" target="_blank" class="contact-button">Contact</a>', unsafe_allow_html=True)
                
                # Message and prediction
                st.markdown(f"<p><strong>Message:</strong> {row['message']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Overall:</strong> {row['prediction']}</p>", unsafe_allow_html=True)
                
                # Sentence analysis (if available)
                if 'sentence_analysis' in row and row['sentence_analysis']:
                    with st.expander("View Sentence Analysis"):
                        for sent in row['sentence_analysis']:
                            sent_prob = sent.get('probability', 0)
                            sent_class = 'sentence-depressed' if sent_prob > 0.58 else 'sentence-normal'
                            st.markdown(f"""
                                <div class='sentence-card {sent_class}'>
                                    <p>{sent.get('sentence')}</p>
                                    <p><strong>Analysis:</strong> {sent.get('prediction')} ({sent_prob:.1%})</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No potentially depressed users found in the dataset.")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    total = len(df)
    depressed = depressed_count
    not_depressed = total - depressed
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Analyzed", total)
        st.metric("Potentially Depressed", depressed)
    
    with col2:
        st.metric("Not Depressed", not_depressed)
        if total > 0:
            st.metric("Depression Rate", f"{depressed/total:.1%}")

# Header
st.markdown("<h1 class='main-header'>🧠 Bulk Depression Detection</h1>", unsafe_allow_html=True)

# Description
st.markdown("<p class='info-text'>Upload a CSV file with Twitter usernames and messages to analyze for signs of depression in bulk.</p>", unsafe_allow_html=True)

# Disclaimer
st.markdown("<p class='disclaimer'>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional mental health diagnosis or treatment.</p>", unsafe_allow_html=True)

# Create a container for the CSV requirements
with st.container():
    st.subheader("CSV Format Requirements")
    
    # Use explicit syntax that works regardless of theme
    st.write("Your CSV file must have these columns:")
    
    # Create a colored box with white text - this avoids any theme issues
    st.markdown("""
    <div style="padding: 1rem; margin-bottom: 1rem;">
        <p>• <strong>username</strong>: Twitter username</p>
        <p>• <strong>message</strong>: Text content to analyze</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a very explicit example with forced colors
    st.markdown("""
    <p style="margin-bottom: 0.5rem; font-weight: bold;">CSV Format Example:</p>
    <div style="background-color: #1E293B; color: #FFFFFF; padding: 1rem; border-radius: 0.5rem; 
        font-family: monospace; margin-bottom: 1rem; overflow: auto;">
        <code style="color: #FFFFFF !important; white-space: pre; display: block;">
username,message
user123,"I'm feeling so down lately, nothing seems to help"
user456,"Had a great day at the beach with friends"
        </code>
    </div>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# Add model selection option
model_options = {
    "LSTM": "lstm",
    "GRU": "gru"
}
selected_model = st.selectbox(
    "Select Model", 
    options=list(model_options.keys()),
    format_func=lambda x: x,
    index=0
)
model_type = model_options[selected_model]

if uploaded_file is not None:
    try:
        # Load the CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['username', 'message']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        else:
            # Show preview of the data
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Process button
            if st.button("Process Messages"):
                if not df.empty:
                    with st.spinner(f"Analyzing messages using {selected_model} model... This may take a minute."):
                        try:
                            # Prepare file for upload
                            csv_bytes = uploaded_file.getvalue()
                            files = {"file": ("input.csv", csv_bytes, "text/csv")}
                            
                            # Call bulk API with model type parameter
                            response = requests.post(API_URL, files=files, data={"model_type": model_type})
                            
                            if response.status_code == 200:
                                result_data = response.json()
                                results = result_data.get("results", [])
                                
                                # Convert to DataFrame
                                result_df = pd.DataFrame(results)
                                
                                # Store results in session state
                                st.session_state.bulk_results = result_df
                                st.session_state.current_page = 1
                                
                                st.success(f"Processed {len(result_df)} messages successfully using {selected_model} model!")
                            else:
                                st.error(f"Error: {response.status_code} - {response.text}")
                                
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
            
            # Display results if available
            if st.session_state.bulk_results is not None:
                display_results(st.session_state.bulk_results)
                
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# Add requirements information at the bottom
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer'>To run this app, you need to have the FastAPI backend running at http://127.0.0.1:3000</p>", unsafe_allow_html=True) 