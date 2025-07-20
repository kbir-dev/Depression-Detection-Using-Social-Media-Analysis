import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
import numpy as np
import re
import string
import logging
import os
import io
import pickle
from typing import List, Dict
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML libraries with error handling
import warnings
import os
import re

# Check if running on Streamlit Cloud
is_streamlit_cloud = os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit_sharing'

# Force fallback mode on Streamlit Cloud
if is_streamlit_cloud:
    ML_LIBRARIES_AVAILABLE = False
    warnings.warn("Running on Streamlit Cloud - using fallback mode without TensorFlow")
else:
    # Try to import ML libraries only if not on Streamlit Cloud
    try:
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import load_model
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        import spacy
        from gensim.models import Word2Vec
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Set TensorFlow log level to suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        ML_LIBRARIES_AVAILABLE = True
    except ImportError as e:
        warnings.warn(f"ML libraries not available, running in fallback mode: {e}")
        ML_LIBRARIES_AVAILABLE = False

# Flag to track if we're in fallback mode
FALLBACK_MODE = not ML_LIBRARIES_AVAILABLE

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    if FALLBACK_MODE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        logger.error(f"Error loading spaCy model: {str(e)}")
        st.warning("Could not load spaCy model. Some features will be limited.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading spaCy model: {str(e)}")
        return None

# Initialize lemmatizer
@st.cache_resource
def get_lemmatizer():
    if FALLBACK_MODE:
        return None
        
    try:
        return WordNetLemmatizer()
    except Exception as e:
        logger.error(f"Error initializing lemmatizer: {str(e)}")
        return None

# Define a constant
DEPRESSION_THRESHOLD = 0.58

# Load models
@st.cache_resource
def load_models():
    if FALLBACK_MODE:
        return {
            "fallback": True,
            "active_model_name": "fallback",
            "model_version": "Fallback mode - models not loaded"
        }
        
    try:
        # Define model paths
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "models")
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            st.warning(f"Models directory not found at {models_dir}")
            return {"fallback": True, "active_model_name": "fallback", "model_version": "Models directory not found"}
        
        lstm_model_path = os.path.join(models_dir, "final_lstm_model.keras")
        gru_model_path = os.path.join(models_dir, "final_gru_model.keras")
        word2vec_path = os.path.join(models_dir, "word2vec_model.bin") 
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        
        # Check if all model files exist
        missing_files = []
        for path in [lstm_model_path, gru_model_path, word2vec_path, vectorizer_path]:
            if not os.path.exists(path):
                missing_files.append(os.path.basename(path))
        
        if missing_files:
            st.warning(f"Missing model files: {', '.join(missing_files)}")
            return {"fallback": True, "active_model_name": "fallback", "model_version": "Missing model files"}

        # Load vectorizer first (smallest file)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load word2vec model
        word2vec_model = Word2Vec.load(word2vec_path)
        
        # Load TensorFlow models
        try:
            lstm_model = tf.keras.models.load_model(lstm_model_path)
            gru_model = tf.keras.models.load_model(gru_model_path)
            
            # Default model to use
            active_model = lstm_model
            active_model_name = "lstm"
            
            # Get model version or creation date
            model_stats = os.stat(lstm_model_path)
            model_date = model_stats.st_mtime
            MODEL_VERSION = f"Model date: {pd.to_datetime(model_date, unit='s')}"
            
            return {
                "lstm_model": lstm_model,
                "gru_model": gru_model,
                "word2vec_model": word2vec_model,
                "vectorizer": vectorizer,
                "active_model": active_model,
                "active_model_name": active_model_name,
                "model_version": MODEL_VERSION,
                "fallback": False
            }
        except Exception as e:
            logger.error(f"Error loading TensorFlow models: {str(e)}")
            st.warning(f"Could not load TensorFlow models: {str(e)}")
            return {
                "word2vec_model": word2vec_model,
                "vectorizer": vectorizer,
                "active_model_name": "fallback",
                "model_version": f"Error loading TensorFlow models: {str(e)}",
                "fallback": True
            }
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.warning(f"Error loading models: {str(e)}")
        return {"fallback": True, "active_model_name": "fallback", "model_version": f"Error: {str(e)}"}


def cleanText(text):
    """
    Enhanced text cleaning function that handles:
    - Multiple lines
    - Extra spaces
    - Quotation marks
    - Special characters
    - URLs
    - Emojis
    - Numbers
    - HTML tags
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t\r]+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    
    # Remove numbers and number-word combinations
    text = re.sub(r'\b\d+\w*\b|\b\w*\d+\b', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'<Emoji:.*?>', '', text)  # Remove emoji tags
    
    # Remove punctuation (keeping apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Handle contractions properly (e.g., don't -> dont)
    text = re.sub(r'\'', '', text)
    
    # Remove extra quotes
    text = text.replace('"', '').replace('"', '').replace('"', '')
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Final cleanup with BeautifulSoup (handles any remaining HTML entities)
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Final whitespace cleanup
    text = text.strip()
    
    return text

def get_word_embeddings(words, word2vec_model):
    """
    Get word embeddings using the word2vec model
    """
    embeddings = []
    for word in words:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
        else:
            # Add zero vector for unknown words
            embeddings.append(np.zeros(100))  # Assuming 100-dimensional vectors
    return embeddings

def extract_morph_features(pos_tags):
    """
    Extract morphological features
    """
    categories = {"NOUN": 0, "PRON": 0, "ADV": 0, "ADJ": 0, "VERB": 0, "CONJ": 0, "DET": 0}
    total_words = len(pos_tags)

    if total_words > 0:
        for _, tag in pos_tags:
            if tag in categories:
                categories[tag] += 1

        for key in categories:
            categories[key] /= total_words

    return categories

def compute_stylometric_features(texts, nlp):
    """
    Compute stylometric features
    """
    word_counts = []
    sentence_counts = []
    words_per_sentence = []

    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Calculate words per sentence
        if sentence_count > 0:
            wps = word_count / sentence_count
        else:
            wps = 0
            
        word_counts.append(word_count)
        sentence_counts.append(sentence_count)
        words_per_sentence.append(wps)
        
    return word_counts, sentence_counts, words_per_sentence

def predict_depression(message, model_type="lstm", models=None, nlp=None, lemmatizer=None):
    """
    Predict depression from a message using the selected model.
    
    Args:
        message (str): The input message to analyze
        model_type (str): The model type to use ("lstm" or "gru")
        
    Returns:
        tuple: Prediction results including probability and features
    """
    # Select the model based on model_type
    if model_type.lower() == "gru":
        active_model = models["gru_model"]
        active_model_name = "gru"
    else:
        active_model = models["lstm_model"]
        active_model_name = "lstm"
    
    # 1. Text Cleaning and Preprocessing
    cleaned_message = cleanText(message)
    tokens = word_tokenize(cleaned_message)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_message = " ".join(tokens)
    
    # 2. Feature Extraction
    # a. Basic Features (shape: (1, 8))
    doc = nlp(processed_message)
    pos_tags = [(token.text, token.pos_) for token in doc]
    morph_features = extract_morph_features(pos_tags)
    word_counts, sentence_counts, words_per_sentence = compute_stylometric_features([processed_message], nlp)
    basic_features = np.array([[
        word_counts[0], 
        sentence_counts[0], 
        words_per_sentence[0],
        morph_features['NOUN'], 
        morph_features['PRON'], 
        morph_features['ADV'],
        morph_features['ADJ'], 
        morph_features['VERB']
    ]], dtype=np.float32)

    # b. Word Embeddings (shape: (1, 300, 100))
    word_embeddings = get_word_embeddings(tokens, models["word2vec_model"])
    word_embeddings = np.array(word_embeddings)
    
    # Pad sequences to shape (1, 300, 100)
    if len(word_embeddings) > 300:
        word_embeddings = word_embeddings[:300]
    elif len(word_embeddings) < 300:
        padding = np.zeros((300 - len(word_embeddings), 100))
        word_embeddings = np.vstack((word_embeddings, padding))
    
    # Add batch dimension if needed
    padded_embeddings = np.expand_dims(word_embeddings, axis=0)

    # c. TF-IDF (shape: (1, 8805))
    tfidf_features = models["vectorizer"].transform([processed_message])
    tfidf_features = tfidf_features.toarray()
    
    # 3. Prediction using selected model
    prediction = active_model.predict([padded_embeddings, basic_features, tfidf_features])
    
    return prediction[0][0], padded_embeddings, basic_features, tfidf_features, active_model_name

def predict_depression_paragraph(paragraph, model_type="lstm", models=None, nlp=None, lemmatizer=None):
    """
    Predict depression for a paragraph by analyzing individual sentences and aggregating results.
    Returns both sentence-level predictions and an overall paragraph prediction.
    """
    # Split paragraph into sentences using spaCy
    doc = nlp(paragraph)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # If no valid sentences found, process the entire paragraph as one
    if not sentences:
        sentences = [paragraph]
    
    # Store predictions for each sentence
    sentence_predictions = []
    sentence_features = []
    
    # Process each sentence
    for sentence in sentences:
        if len(sentence.split()) < 3:  # Skip very short sentences
            continue
            
        try:
            probability, embeddings, basic_feats, tfidf_feats, model_name = predict_depression(
                sentence, model_type, models, nlp, lemmatizer
            )
            sentence_predictions.append({
                'sentence': sentence,
                'probability': float(probability),
                'features': {
                    'embeddings': embeddings,
                    'basic': basic_feats,
                    'tfidf': tfidf_feats
                },
                'model': model_name
            })
        except Exception as e:
            st.error(f"Error processing sentence: {sentence[:50]}... Error: {str(e)}")
            continue
    
    if not sentence_predictions:
        return {
            'paragraph_prediction': 0.0,
            'average_prediction': 0.0,
            'sentence_predictions': [],
            'confidence': 'None',
            'model_used': model_type
        }
    
    # Calculate weighted average based on sentence length
    total_weight = 0
    weighted_sum = 0
    
    for pred in sentence_predictions:
        # Weight by sentence length (longer sentences have more impact)
        weight = len(pred['sentence'].split())
        weighted_sum += pred['probability'] * weight
        total_weight += weight
    
    paragraph_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Calculate simple average (in addition to weighted average)
    simple_average = sum(pred['probability'] for pred in sentence_predictions) / len(sentence_predictions)
    
    # Calculate confidence based on:
    # 1. Consistency of predictions
    # 2. Number of sentences analyzed
    probabilities = [pred['probability'] for pred in sentence_predictions]
    prediction_std = np.std(probabilities)
    
    if len(sentence_predictions) >= 3 and prediction_std < 0.15:
        confidence = "High"
    elif len(sentence_predictions) >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return {
        'paragraph_prediction': float(paragraph_prediction),
        'average_prediction': float(simple_average),
        'sentence_predictions': [{
            'sentence': pred['sentence'],
            'probability': pred['probability'],
            'model': pred['model']
        } for pred in sentence_predictions],
        'confidence': confidence,
        'model_used': sentence_predictions[0]['model'] if sentence_predictions else model_type
    }

def simulate_analysis(message, model_type):
    """
    Simulate text analysis when running in fallback mode
    """
    # Check if message is empty or too short
    if not message or len(message.strip()) < 10:
        return {
            "message": message,
            "prediction": "Insufficient data for analysis",
            "probability": 0.0,
            "average_score": 0.0,
            "confidence": "None",
            "sentence_analysis": [],
            "model_used": "fallback"
        }
    
    # Use simple heuristics to simulate depression detection
    # This is just a placeholder and not a real depression detection algorithm
    depression_keywords = [
        "sad", "depressed", "unhappy", "miserable", "hopeless", "worthless",
        "tired", "exhausted", "lonely", "alone", "suicide", "die", "death",
        "crying", "tears", "pain", "hurt", "suffering", "anxiety", "worried"
    ]
    
    # Count depression keywords
    message_lower = message.lower()
    keyword_count = sum(1 for keyword in depression_keywords if keyword in message_lower)
    
    # Simple sentence splitting
    sentences = [s.strip() for s in re.split(r'[.!?]+', message) if s.strip()]
    
    # Calculate a simulated probability based on keyword density
    total_words = len(message_lower.split())
    if total_words > 0:
        base_probability = min(0.9, keyword_count / (total_words * 0.3))
    else:
        base_probability = 0.0
    
    # Add some randomness to make it look more realistic
    import random
    probability = min(0.95, max(0.05, base_probability + random.uniform(-0.1, 0.1)))
    
    # Determine overall result based on the threshold
    is_depressed = probability > DEPRESSION_THRESHOLD
    prediction_text = "Potentially Depressed" if is_depressed else "Not Depressed"
    emoji = "😞" if is_depressed else "🙂"
    
    # Generate simulated sentence analysis
    sentence_analysis = []
    for sentence in sentences:
        if len(sentence.split()) < 3:  # Skip very short sentences
            continue
            
        # Calculate sentence-level probability
        sentence_lower = sentence.lower()
        sentence_keyword_count = sum(1 for keyword in depression_keywords if keyword in sentence_lower)
        sentence_words = len(sentence_lower.split())
        if sentence_words > 0:
            sentence_probability = min(0.95, max(0.05, sentence_keyword_count / (sentence_words * 0.3) + random.uniform(-0.15, 0.15)))
        else:
            sentence_probability = 0.0
            
        is_sent_depressed = sentence_probability > DEPRESSION_THRESHOLD
        sent_emoji = "😞" if is_sent_depressed else "🙂"
        
        sentence_analysis.append({
            "sentence": sentence,
            "prediction": f"{'Potentially Depressed' if is_sent_depressed else 'Not Depressed'} {sent_emoji}",
            "probability": sentence_probability,
            "model": f"fallback-{model_type}"
        })
    
    # Calculate average score as slightly different from probability
    average_score = min(0.95, max(0.05, probability + random.uniform(-0.05, 0.05)))
    
    # Determine confidence based on message length and sentence count
    if len(sentences) >= 3 and len(message) > 100:
        confidence = "Medium"
    elif len(sentences) >= 5 and len(message) > 200:
        confidence = "High"
    else:
        confidence = "Low"
    
    return {
        "message": message,
        "prediction": f"{prediction_text} {emoji}",
        "probability": probability,
        "average_score": average_score,
        "confidence": confidence,
        "sentence_analysis": sentence_analysis,
        "model_used": f"fallback-{model_type}"
    }

def analyze_text(message, model_type, models, nlp, lemmatizer):
    """
    Analyze text for depression indicators
    """
    # Check if message is empty or too short
    if not message or len(message.strip()) < 10:
        return {
            "message": message,
            "prediction": "Insufficient data for analysis",
            "probability": 0.0,
            "average_score": 0.0,
            "confidence": "None",
            "sentence_analysis": []
        }
    
    # Call prediction function with model type
    result = predict_depression_paragraph(message, model_type, models, nlp, lemmatizer)
    
    # Determine overall result based on the threshold
    is_depressed = result['paragraph_prediction'] > DEPRESSION_THRESHOLD
    prediction_text = "Potentially Depressed" if is_depressed else "Not Depressed"
    emoji = "😞" if is_depressed else "🙂"
    
    # Format sentence-level results
    sentence_analysis = []
    for pred in result['sentence_predictions']:
        is_sent_depressed = pred['probability'] > DEPRESSION_THRESHOLD
        sent_emoji = "😞" if is_sent_depressed else "🙂"
        sentence_analysis.append({
            "sentence": pred['sentence'],
            "prediction": f"{'Potentially Depressed' if is_sent_depressed else 'Not Depressed'} {sent_emoji}",
            "probability": pred['probability'],
            "model": pred['model']
        })
    
    return {
        "message": message,
        "prediction": f"{prediction_text} {emoji}",
        "probability": result['paragraph_prediction'],
        "average_score": result['average_prediction'],
        "confidence": result['confidence'],
        "sentence_analysis": sentence_analysis,
        "model_used": result['model_used']
    }

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

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1 class='main-header'>🧠 DepDetect AI - Depression Detection Tool</h1>", unsafe_allow_html=True)

# Brief description
st.markdown("<p class='info-text'>This tool uses AI to analyze text for potential signs of depression. Enter your text below and click 'Analyze'.</p>", unsafe_allow_html=True)

# Disclaimer
st.markdown("<p class='disclaimer'>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional mental health diagnosis or treatment.</p>", unsafe_allow_html=True)

# Load resources and models
with st.spinner("Loading resources..."):
    nltk_status = download_nltk_resources()
    nlp = load_spacy_model()
    lemmatizer = get_lemmatizer()
    models = load_models()
    
# Display fallback mode notice if needed
if FALLBACK_MODE or models.get("fallback", False):
    st.warning(
        "⚠️ **Running in fallback mode**\n\n"
        "The app is running with limited functionality because some required ML libraries "
        "or models couldn't be loaded. You can still use the interface, but prediction "
        "results will be simulated."
    )

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
                
                # Process the text directly without API call
                if FALLBACK_MODE or models.get("fallback", False):
                    # In fallback mode, provide a simulated response
                    result = simulate_analysis(message, model_type.lower())
                else:
                    # Normal processing with ML models
                    result = analyze_text(message, model_type.lower(), models, nlp, lemmatizer)
                
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
                    title={'text': "Depression Probability", 'font': {'size': 16, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                        'bar': {'color': "#4F46E5"},
                        'bgcolor': "rgba(50, 50, 50, 0.8)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 58], 'color': '#10B981'},  # Green for non-depressed
                            {'range': [58, 100], 'color': '#EF4444'}  # Red for depressed
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sentence-level analysis
                if sentence_analysis:
                    st.markdown("<h3 class='subheader'>Sentence-Level Analysis</h3>", unsafe_allow_html=True)
                    st.markdown("<p class='info-text'>Each sentence is analyzed individually to identify potential signs of depression.</p>", unsafe_allow_html=True)
                    
                    for sentence in sentence_analysis:
                        is_depressed = sentence["probability"] > 0.58
                        card_class = "sentence-card sentence-depressed" if is_depressed else "sentence-card sentence-normal"
                        
                        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
                        st.markdown(f"<p class='sentence-text'>\"{sentence['sentence']}\"</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='sentence-analysis'>{sentence['prediction']} (Score: {sentence['probability']:.1%})</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
    else:
        st.warning("Please enter a message to analyze.")

# Show history
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Previous Analyses")
    
    for i, result in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 analyses
        with st.expander(f"Analysis {len(st.session_state.history) - i}: {result['prediction']}"):
            st.markdown(f"**Message:** {result['message'][:100]}..." if len(result['message']) > 100 else f"**Message:** {result['message']}")
            st.markdown(f"**Prediction:** {result['prediction']}")
            st.markdown(f"**Score:** {result['probability']:.1%}")
            st.markdown(f"**Model Used:** {result['model_used'].upper()}")
