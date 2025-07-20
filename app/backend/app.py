import uvicorn
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import pickle
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import logging
import pandas as pd
import io
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("All NLTK data downloaded successfully!")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model successfully")
except OSError:
    logger.error("Error loading spaCy model. Make sure to run: python -m spacy download en_core_web_sm")
    raise

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a constant at the top of your file
DEPRESSION_THRESHOLD = 0.58

class MessageInput(BaseModel):
    message: str
    model_type: str = "lstm"  # Default to LSTM model

class Message(BaseModel):
    message: str

class BulkResponse(BaseModel):
    results: List[Dict]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models
try:
    # Define model paths
    lstm_model_path = os.path.join("models", "final_lstm_model.keras")
    gru_model_path = os.path.join("models", "final_gru_model.keras")
    word2vec_path = os.path.join("models", "word2vec_model.bin") 
    vectorizer_path = os.path.join("models", "vectorizer.pkl")

    # Load both models
    logger.info(f"Loading LSTM model from {lstm_model_path}")
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    
    logger.info(f"Loading GRU model from {gru_model_path}")
    gru_model = tf.keras.models.load_model(gru_model_path)
    
    # Default model to use (can be changed via API)
    active_model = lstm_model
    active_model_name = "lstm"
    
    # Get model version or creation date
    model_stats = os.stat(lstm_model_path)
    model_date = model_stats.st_mtime
    MODEL_VERSION = f"Model date: {pd.to_datetime(model_date, unit='s')}"
    logger.info(f"Model version: {MODEL_VERSION}")

    logger.info(f"Loading word2vec from {word2vec_path}")
    word2vec_model = Word2Vec.load(word2vec_path)  # Use correct loading method from notebook

    logger.info(f"Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

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

def get_word_embeddings(words):
    """
    Get word embeddings using the same approach as in the notebook
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
    Extract morphological features using the same approach as in the notebook
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

def compute_stylometric_features(texts):
    """
    Compute stylometric features using the same approach as in the notebook
    """
    word_counts = []
    sentence_counts = []
    words_per_sentence = []

    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)

        word_counts.append(len(words))  # Total words in message
        sentence_counts.append(len(sentences))  # Total sentences in message
        words_per_sentence.append(
            np.mean([len(sent.text.split()) for sent in sentences]) if sentences else 0
        )

    return word_counts, sentence_counts, words_per_sentence

def predict_depression(message, model_type="lstm"):
    """
    Predict depression from a message using the selected model.
    
    Args:
        message (str): The input message to analyze
        model_type (str): The model type to use ("lstm" or "gru")
        
    Returns:
        dict: Prediction results including probability and classification
    """
    global active_model, active_model_name
    
    # Select the model based on model_type
    if model_type.lower() == "gru" and active_model_name != "gru":
        active_model = gru_model
        active_model_name = "gru"
        logger.info("Switched to GRU model")
    elif model_type.lower() == "lstm" and active_model_name != "lstm":
        active_model = lstm_model
        active_model_name = "lstm"
        logger.info("Switched to LSTM model")
    
    # Log the input message
    logger.info(f"Predicting depression for message: {message[:50]}...")
    
    # 1. Text Cleaning and Preprocessing
    cleaned_message = cleanText(message)
    tokens = word_tokenize(cleaned_message)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_message = " ".join(tokens)
    
    logger.info(f"Processed message: {processed_message[:50]}...")
    
    # 2. Feature Extraction
    # a. Basic Features (shape: (1, 8))
    doc = nlp(processed_message)
    pos_tags = [(token.text, token.pos_) for token in doc]
    morph_features = extract_morph_features(pos_tags)
    word_counts, sentence_counts, words_per_sentence = compute_stylometric_features([processed_message])
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
    word_embeddings = get_word_embeddings(tokens)
    word_embeddings = np.array(word_embeddings)
    logger.info(f"Original word embeddings shape: {word_embeddings.shape}")
    
    # Pad sequences to shape (1, 300, 100)
    if len(word_embeddings) > 300:
        word_embeddings = word_embeddings[:300]
    elif len(word_embeddings) < 300:
        padding = np.zeros((300 - len(word_embeddings), 100))
        word_embeddings = np.vstack((word_embeddings, padding))
    
    # Add batch dimension if needed
    padded_embeddings = np.expand_dims(word_embeddings, axis=0)
    logger.info(f"Final embeddings shape: {padded_embeddings.shape}")

    # c. TF-IDF (shape: (1, 8805))
    tfidf_features = vectorizer.transform([processed_message])
    tfidf_features = tfidf_features.toarray()
    
    # Verify shapes match model requirements
    logger.info(f"""Feature shapes:
    - Word Embeddings: {padded_embeddings.shape} (expected: (1, 300, 100))
    - Basic Features: {basic_features.shape} (expected: (1, 8))
    - TF-IDF Features: {tfidf_features.shape} (expected: (1, 8805))
    """)

    # Verify shapes match model requirements
    assert padded_embeddings.shape == (1, 300, 100), "Wrong shape for embeddings"
    assert basic_features.shape == (1, 8), "Wrong shape for basic features"
    assert tfidf_features.shape == (1, 8805), "Wrong shape for TF-IDF features"

    # 3. Prediction using LSTM model
    prediction = active_model.predict([padded_embeddings, basic_features, tfidf_features])
    
    logger.info(f"Raw prediction value: {prediction[0][0]}")
    
    return prediction[0][0], padded_embeddings, basic_features, tfidf_features, active_model_name

def predict_depression_paragraph(paragraph, model_type="lstm"):
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
    
    logger.info(f"Processing paragraph with {len(sentences)} sentences")
    
    # Store predictions for each sentence
    sentence_predictions = []
    sentence_features = []
    
    # Process each sentence
    for sentence in sentences:
        if len(sentence.split()) < 3:  # Skip very short sentences
            continue
            
        try:
            probability, embeddings, basic_feats, tfidf_feats, model_name = predict_depression(sentence, model_type)
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
            logger.error(f"Error processing sentence: {sentence[:50]}... Error: {str(e)}")
            continue
    
    if not sentence_predictions:
        logger.warning("No valid sentences could be processed")
        return {
            'paragraph_prediction': 0.0,
            'average_prediction': 0.0,
            'sentence_predictions': [],
            'confidence': 'None',
            'model_used': active_model_name
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
        'model_used': sentence_predictions[0]['model'] if sentence_predictions else active_model_name
    }

@app.get("/")
async def root():
    return {"message": "Depression Detection API is running. Use /predict endpoint to analyze text."}

@app.post("/predict")
async def predict_endpoint(data: MessageInput):
    """
    Endpoint to predict depression from a single message.
    """
    try:
        # Extract message and model type from request
        message = data.message
        model_type = data.model_type
        
        logger.info(f"Received prediction request with model type: {model_type}")
        
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
        result = predict_depression_paragraph(message, model_type)
        
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
            "message": data.message,
            "prediction": f"{prediction_text} {emoji}",
            "probability": result['paragraph_prediction'],
            "average_score": result['average_prediction'],
            "confidence": result['confidence'],
            "sentence_analysis": sentence_analysis,
            "model_used": result['model_used']
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/bulk-predict")
async def bulk_predict(file: UploadFile = File(...), model_type: str = Form(...)):
    """
    Endpoint to predict depression from a CSV file containing messages.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        logger.info(f"Received bulk prediction request with model type: {model_type}")
        
        # Read and process CSV file
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        results = []
        
        # Validate required columns
        required_columns = ['username', 'message']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"CSV is missing required column: {col}")
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Get username and message
                username = str(row['username'])
                message = str(row['message'])
                
                # Skip empty messages
                if not message or len(message.strip()) < 10:
                    results.append({
                        "username": username,
                        "message": message,
                        "prediction": "Insufficient data for analysis",
                        "depression_probability": 0.0,
                        "is_depressed": False,
                        "model_used": model_type
                    })
                    continue
                
                # Use paragraph-based prediction
                result = predict_depression_paragraph(message, model_type)
                
                # Determine overall result based on the threshold
                is_depressed = result['paragraph_prediction'] > DEPRESSION_THRESHOLD
                
                # Add to results
                results.append({
                    "username": username,
                    "message": message[:100] + "..." if len(message) > 100 else message,
                    "prediction": "Depressed" if is_depressed else "Not Depressed",
                    "depression_probability": float(result['paragraph_prediction']),
                    "average_score": float(result['average_prediction']),
                    "is_depressed": is_depressed,
                    "confidence": result['confidence'],
                    "model_used": result['model_used'],
                    "sentence_analysis": result.get('sentence_predictions', [])
                })
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append({
                    "username": row.get('username', 'Unknown'),
                    "message": message if 'message' in locals() else "Unknown",
                    "prediction": "Error",
                    "depression_probability": 0.0,
                    "is_depressed": False,
                    "error": str(e),
                    "model_used": model_type
                })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error processing bulk request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing bulk request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)