# START OF FULL CODE: hello.py (Corrected Feature Lists and Prediction Logic)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from dotenv import load_dotenv
import os
import traceback
import re
import bcrypt
from pymongo import MongoClient, DESCENDING
from datetime import datetime, timedelta
import pandas as pd
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# --- ML/DL Imports ---
import tensorflow as tf # Still needed for emotion detection model
import joblib
# Make sure sklearn is installed: pip install scikit-learn joblib tensorflow pandas numpy opencv-python streamlit python-dotenv pymongo bcrypt vaderSentiment Pillow
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Keep for reference
from sklearn.compose import ColumnTransformer # Keep for reference
from sklearn.pipeline import Pipeline # Keep for reference
from sklearn.impute import SimpleImputer # Keep for reference
from sklearn.neighbors import KNeighborsClassifier # Keep for reference
# --- End ML/DL Imports ---

# Optional: Import google genai if using
try:
    import google.generativeai as genai
except ImportError:
    genai = None

st.set_page_config(page_title="Serene - AI Mental Health Assistant", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# --- Application Config ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CURRENT_LOCATION_CONTEXT = "Anekal, Karnataka, India"

# --- Custom Emotion Model Config ---
CUSTOM_MODEL_PATH = 'emotion_model.h5'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_HEIGHT = 48
IMG_WIDTH = 48
COLOR_MODE = 'grayscale'

# --- Feature Constants ---
MAX_HISTORY_LEN = 10

# --- ML Model Integration Config (MODIFIED FOR SKLEARN PIPELINE) ---
# Path to the saved Scikit-learn pipeline from **CORRECTED** train.py
SKLEARN_PIPELINE_PATH = 'sklearn_knn_mental_health_pipeline.joblib'

# Define expected feature columns - **MUST MATCH intended_feature_columns in CORRECTED train.py**
# **>>> ADJUST THESE LISTS TO MATCH YOUR FINAL intended_feature_columns IN train.py <<<**
EXPECTED_NUMERICAL_COLS = [
    'final_phq9_score', 'final_gad7_score', 'stress_level',
    'social_interactions', 'emotion_volatility_estimate',
    'avg_sentiment_last_5'
    # Add/remove numerical columns based on your intended_feature_columns in train.py
]
EXPECTED_CATEGORICAL_COLS = [
    'sleep_quality', 'activity_level', 'final_emotion', 'final_text_sentiment'
    # Add/remove categorical columns based on your intended_feature_columns in train.py
]
# This list MUST contain exactly the columns used for training in the corrected train.py
ALL_EXPECTED_FEATURES = EXPECTED_NUMERICAL_COLS + EXPECTED_CATEGORICAL_COLS
# --- End ML Model Integration Config ---


# --- Other Constants ---
INDIA_MENTAL_HEALTH_HELPLINES = {
    "KIRAN (National Helpline)": "1800-599-0019", "Vandrevala Foundation": "9999666555 (24x7)",
    "Fortis Stress Helpline": "+91-8376804102", "Nimhans Helpline (Bangalore)": "080-46110007",
    "AASRA": "9820466726 (24x7)"
}
GENERAL_SEARCH_ADVICE_INDIA = f"""
You can also:
* Contact your General Practitioner (GP) or visit a local primary health centre/government hospital in {CURRENT_LOCATION_CONTEXT.split(',')[0]} or nearby for guidance.
* Search online for mental health professionals (Psychiatrists, Psychologists, Counselors) using terms like "psychologist {CURRENT_LOCATION_CONTEXT.split(',')[0]}", "psychiatrist Bangalore", "counselor near {CURRENT_LOCATION_CONTEXT.split(',')[0]}". Look for directories on platforms like Practo, Lybrate.
* Check the Karnataka Department of Health and Family Welfare website for District Mental Health Programme (DMHP) contacts.
* Explore reputable online therapy platforms available in India (e.g., YourDOST, BetterLYF).
"""

# --- PHQ-9 Constants ---
PHQ9_QUESTIONS = [
    {"id": "q1", "text": "Little interest or pleasure in doing things"}, {"id": "q2", "text": "Feeling down, depressed, or hopeless"},
    {"id": "q3", "text": "Trouble falling or staying asleep, or sleeping too much"}, {"id": "q4", "text": "Feeling tired or having little energy"},
    {"id": "q5", "text": "Poor appetite or overeating"}, {"id": "q6", "text": "Feeling bad about yourself - or that you are a failure or have let yourself or your family down"},
    {"id": "q7", "text": "Trouble concentrating on things, such as reading the newspaper or watching television"},
    {"id": "q8", "text": "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual"},
    {"id": "q9", "text": "Thoughts that you would be better off dead, or of hurting yourself in some way"}
]
PHQ9_OPTIONS = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
PHQ9_OPTION_LIST = list(PHQ9_OPTIONS.keys())

# --- GAD-7 Constants ---
GAD7_QUESTIONS = [
    {"id": "g1", "text": "Feeling nervous, anxious, or on edge"},
    {"id": "g2", "text": "Not being able to stop or control worrying"},
    {"id": "g3", "text": "Worrying too much about different things"},
    {"id": "g4", "text": "Trouble relaxing"},
    {"id": "g5", "text": "Being so restless that it is hard to sit still"},
    {"id": "g6", "text": "Becoming easily annoyed or irritable"},
    {"id": "g7", "text": "Feeling afraid as if something awful might happen"}
]
GAD7_OPTIONS = PHQ9_OPTIONS
GAD7_OPTION_LIST = PHQ9_OPTION_LIST


# ==============================================================================
# Setup Logging, DB, AI Models
# ==============================================================================
# Define logger format - Use a consistent format
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format) # Changed format
analyzer = SentimentIntensityAnalyzer()

# --- MongoDB Connection ---
try:
    client = MongoClient(MONGO_URI)
    db = client["mental_health_bot"]
    users_collection = db["users"]
    chats_collection = db["chats"]
    phq_collection = db["phq_scores"]
    session_summary_collection = db["session_summaries"]
    client.admin.command('ping')
    logging.info("MongoDB connection successful.")
except Exception as e:
    logging.error(f"CRITICAL: Failed to connect to MongoDB at {MONGO_URI}: {e}", exc_info=True) # Added exc_info
    st.error(f"Database connection failed. Please check MongoDB connection. Error: {e}")
    st.stop()

# --- Gemini Model Setup (Optional) ---
gemini_model = None
if genai and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        logging.info("Gemini configured successfully for responses.")
    except Exception as e:
        st.error(f"Failed to configure Gemini for responses: {e}")
        logging.error(f"Gemini configuration failed for responses: {e}", exc_info=True) # Added exc_info
elif genai is None:
     st.warning("Google Generative AI library not installed. AI responses will be basic.")
     logging.warning("Google Generative AI library not installed.")
else:
    st.warning("GOOGLE_API_KEY not set in .env file. AI responses will be basic.")
    logging.warning("GOOGLE_API_KEY not set. AI responses will be basic.")


# --- Load Custom Emotion Model and Face Detector (Cached) ---
@st.cache_resource
def load_emotion_model_and_detector():
    loaded_model = None
    face_detector = None
    haar_cascade_full_path = HAAR_CASCADE_PATH

    if not os.path.exists(haar_cascade_full_path):
        logging.warning(f"Haar Cascade not found at specified path: {HAAR_CASCADE_PATH}. Trying default OpenCV path...")
        alt_haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(alt_haar_path):
            haar_cascade_full_path = alt_haar_path
            logging.info(f"Found Haar Cascade at: {haar_cascade_full_path}")
        else:
            logging.error("Haar Cascade file not found at specified or default paths.")
            st.error("Critical Error: Haar Cascade file for face detection not found. Emotion detection disabled.")
            return None, None

    try:
        face_detector = cv2.CascadeClassifier(haar_cascade_full_path)
        if face_detector.empty(): raise IOError(f"Failed to load Haar Cascade from {haar_cascade_full_path}")
        logging.info(f"Face detector loaded successfully from {haar_cascade_full_path}")
    except Exception as e:
        logging.error(f"Error loading face detector: {e}", exc_info=True) # Added exc_info
        st.error(f"Error loading face detector: {e}. Emotion detection may fail.")
        return None, None

    try:
        if os.path.exists(CUSTOM_MODEL_PATH):
            loaded_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH, compile=False)
            logging.info(f"Custom emotion model loaded successfully from {CUSTOM_MODEL_PATH}")
        else:
            logging.error(f"Custom emotion model file not found at {CUSTOM_MODEL_PATH}")
            st.error(f"Emotion model file '{CUSTOM_MODEL_PATH}' not found. Emotion detection disabled.")
            loaded_model = None
    except Exception as e:
        logging.error(f"Error loading custom emotion model: {e}", exc_info=True)
        st.error(f"Error loading emotion model: {e}. Emotion detection disabled.")
        loaded_model = None

    return loaded_model, face_detector

# Load models when the script runs
loaded_emotion_model, loaded_face_detector = load_emotion_model_and_detector()


# --- Load Scikit-learn Prediction Pipeline ---
@st.cache_resource
def load_sklearn_pipeline(pipeline_path):
    """Loads the saved Scikit-learn pipeline using joblib."""
    if not os.path.exists(pipeline_path):
        st.error(f"Prediction pipeline file not found at {pipeline_path}. Predictions disabled.")
        logging.error(f"Prediction pipeline file not found at {pipeline_path}.")
        return None
    try:
        pipeline = joblib.load(pipeline_path)
        logging.info(f"Scikit-learn prediction pipeline loaded successfully from {pipeline_path}")
        if not hasattr(pipeline, 'predict'):
            st.error(f"Loaded object from {pipeline_path} does not have a 'predict' method. Is it a valid Scikit-learn pipeline?")
            logging.error(f"Loaded object from {pipeline_path} is not a valid pipeline.")
            return None
        return pipeline
    except Exception as e:
        st.error(f"Error loading Scikit-learn pipeline: {e}")
        logging.error(f"Error loading Scikit-learn pipeline from {pipeline_path}: {e}", exc_info=True)
        return None

# Load the prediction pipeline when the script runs
loaded_prediction_pipeline = load_sklearn_pipeline(SKLEARN_PIPELINE_PATH)
# --- END Loading Prediction Resources ---


# ==============================================================================
# Helper Functions
# ==============================================================================

# --- Authentication Functions ---
def signup(username, password, email):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return False, "Invalid email format"
    if len(password) < 6: return False, "Password must be at least 6 characters"
    if users_collection.count_documents({"username": username}) > 0: return False, "Username already exists"
    if users_collection.count_documents({"email": email}) > 0: return False, "Email already registered"
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        users_collection.insert_one({
            "username": username, "password_hash": hashed_pw, "email": email, "created_at": datetime.utcnow()
        })
        logging.info(f"User '{username}' signed up successfully.")
        return True, "Signup successful! Please login."
    except Exception as e:
        logging.error(f"Error during signup for {username}: {e}", exc_info=True)
        return False, f"Signup failed due to a server error: {e}"

def login(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        logging.info(f"User '{username}' logged in successfully.")
        return True, "Login successful"
    logging.warning(f"Failed login attempt for username: {username}")
    return False, "Invalid username or password"

# --- Database & Assessment Functions ---
def save_message(username, role, message):
    try:
        chats_collection.insert_one({
            "username": username, "role": role, "message": message, "timestamp": datetime.utcnow()
        })
        logging.debug(f"Saved message for {username}: Role={role}")
    except Exception as e:
        logging.error(f"Failed to save message for {username}: {e}", exc_info=True)
        st.warning("Could not save message history.")

def get_phq9_interpretation(score):
    if score is None: return "Score not calculated yet.", "unknown"
    if not isinstance(score, (int, float)): return "Invalid score.", "unknown"
    try: # Add try-except for robust conversion
        score = int(score)
    except (ValueError, TypeError):
        logging.error(f"Could not convert PHQ-9 score '{score}' to int.")
        return "Invalid score format.", "unknown"
    severity = "unknown"; text = f"PHQ-9 Score: {score}. "; action = ""
    if 0 <= score <= 4: severity = "minimal"; text += "Depression Severity: Minimal or None."; action = "Continue monitoring feelings."
    elif 5 <= score <= 9: severity = "mild"; text += "Depression Severity: Mild."; action = "Suggested Action: Watchful waiting; monitor symptoms."
    elif 10 <= score <= 14: severity = "moderate"; text += "Depression Severity: Moderate."; action = "Suggested Action: Consider seeking support (counseling/doctor). Follow-up recommended."
    elif 15 <= score <= 19: severity = "moderately_severe"; text += "Depression Severity: Moderately Severe."; action = "Suggested Action: Strongly recommended to seek professional help soon (therapy/medication)."
    elif 20 <= score <= 27: severity = "severe"; text += "Depression Severity: Severe."; action = "Suggested Action: Very important to seek professional help immediately (therapy/medication)."
    else: return f"Score {score} out of range [0-27].", "unknown"
    return f"{text} {action}", severity

def save_phq9_score(username, score):
    interpretation, severity = get_phq9_interpretation(score)
    try:
        phq_collection.insert_one({
            "username": username, "score": score, "severity": severity,
            "interpretation_text": interpretation, "timestamp": datetime.utcnow()
        })
        logging.info(f"Saved PHQ-9 score for {username}: Score={score}, Severity={severity}")
    except Exception as e:
        logging.error(f"Failed to save PHQ-9 score for {username}: {e}", exc_info=True)
        st.warning("Could not save PHQ-9 score.")

def load_phq9_history(username):
    try:
        records = list(phq_collection.find({"username": username}).sort("timestamp"))
        if not records: return pd.DataFrame(columns=["timestamp", "score"])
        df = pd.DataFrame([{"timestamp": pd.to_datetime(r["timestamp"]).tz_localize(None), "score": r["score"]} for r in records])
        return df
    except Exception as e:
        logging.error(f"Failed to load PHQ-9 history for {username}: {e}", exc_info=True)
        st.warning("Could not load PHQ-9 history.")
        return pd.DataFrame(columns=["timestamp", "score"])

def get_gad7_interpretation(score):
    if score is None: return "Score not calculated yet.", "unknown"
    if not isinstance(score, (int, float)): return "Invalid score.", "unknown"
    try: # Add try-except for robust conversion
        score = int(score)
    except (ValueError, TypeError):
         logging.error(f"Could not convert GAD-7 score '{score}' to int.")
         return "Invalid score format.", "unknown"
    severity = "unknown"; text = f"GAD-7 Score: {score}. "; action = ""
    if 0 <= score <= 4: severity = "minimal"; text += "Anxiety Severity: Minimal."; action = "Symptoms are minimal, monitoring is sufficient."
    elif 5 <= score <= 9: severity = "mild"; text += "Anxiety Severity: Mild."; action = "Consider discussing symptoms if they persist or worsen."
    elif 10 <= score <= 14: severity = "moderate"; text += "Anxiety Severity: Moderate."; action = "Further evaluation recommended; consider seeking support."
    elif 15 <= score <= 21: severity = "severe"; text += "Anxiety Severity: Severe."; action = "Important to seek professional evaluation and treatment."
    else: return f"Score {score} out of range [0-21].", "unknown"
    return f"{text} {action}", severity

# --- Sentiment Analysis (VADER) ---
def analyze_sentiment(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return "Neutral", 0.0
    try:
        vs = analyzer.polarity_scores(text)
        compound_score = vs['compound']
        if compound_score >= 0.05: sentiment = "Positive"
        elif compound_score <= -0.05: sentiment = "Negative"
        else: sentiment = "Neutral"
        logging.info(f"Sentiment analysis (VADER): Compound={compound_score:.3f} -> {sentiment}")
        if 'sentiment_scores_history' in st.session_state:
             # Ensure it's a list before appending
             if isinstance(st.session_state.sentiment_scores_history, list):
                 st.session_state.sentiment_scores_history.append(compound_score)
                 st.session_state.sentiment_scores_history = st.session_state.sentiment_scores_history[-MAX_HISTORY_LEN:]
             else: # Initialize if it's not a list
                 st.session_state.sentiment_scores_history = [compound_score]
        else:
            st.session_state.sentiment_scores_history = [compound_score]
        return sentiment, compound_score
    except Exception as e:
        logging.error(f"VADER sentiment analysis error: {e}", exc_info=True)
        return "Neutral", 0.0

# --- Feature Calculation Functions ---
def calculate_average_phq9_change_rate(username):
    try:
        history = list(phq_collection.find({"username": username}).sort("timestamp", DESCENDING).limit(5))
        if len(history) < 2: return None
        history.reverse() # Oldest first for calculation
        total_change = 0; total_days = 0
        for i in range(len(history) - 1):
            try:
                score_diff = float(history[i+1]['score']) - float(history[i]['score'])
                time_diff = history[i+1]['timestamp'] - history[i]['timestamp']
                days_diff = time_diff.total_seconds() / (60*60*24)
                if days_diff >= 1: # Only consider changes over at least 1 day
                    total_change += score_diff
                    total_days += days_diff
            except (ValueError, TypeError, KeyError) as item_err:
                logging.warning(f"Skipping item in PHQ change rate calc due to error: {item_err}")
                continue # Skip corrupted entries
        if total_days == 0: return None
        average_rate = total_change / total_days
        logging.info(f"Calculated avg PHQ-9 change rate for {username}: {average_rate:.2f}/day over {total_days:.1f} days")
        return average_rate
    except Exception as e:
        logging.error(f"Error calculating PHQ-9 change rate for {username}: {e}", exc_info=True)
        return None

def calculate_recent_mood_trend(phq_history_df, sentiment_history, emotion_history):
    # Simplified: trend primarily based on recent sentiment
    sentiment_trend = "Stable"
    if isinstance(sentiment_history, list) and len(sentiment_history) >= 3:
        try:
            # Ensure values are numeric
            numeric_sentiments = [s for s in sentiment_history if isinstance(s, (int, float))]
            if len(numeric_sentiments) >= 3:
                mid_point = len(numeric_sentiments) // 2
                avg_first = sum(numeric_sentiments[:mid_point]) / mid_point if mid_point > 0 else 0
                avg_second = sum(numeric_sentiments[mid_point:]) / (len(numeric_sentiments) - mid_point) if (len(numeric_sentiments) - mid_point) > 0 else 0
                if avg_second > avg_first + 0.1: sentiment_trend = "Improving"
                elif avg_second < avg_first - 0.1: sentiment_trend = "Declining"
            else:
                sentiment_trend = "Insufficient Data"
        except Exception as e:
             logging.error(f"Error calculating sentiment trend: {e}", exc_info=True)
             sentiment_trend = "Calculation Error"
    elif not isinstance(sentiment_history, list):
         logging.warning("sentiment_history is not a list, cannot calculate trend.")
         sentiment_trend = "Data Error"

    final_trend = sentiment_trend
    logging.debug(f"Estimated mood trend based on sentiment: {final_trend}")
    # Returning a simpler state for clarity
    if final_trend in ["Improving", "Declining"]:
        return final_trend
    else:
        return "Stable/Not Clear"


def calculate_emotion_volatility(emotion_history):
    if not isinstance(emotion_history, list) or len(emotion_history) < 2: return 0.0
    changes = 0; valid_emotion_count = 0; last_valid_emotion = None
    for entry in emotion_history:
        # Check if entry is a dict and has 'emotion' key
        if isinstance(entry, dict):
            emotion = entry.get('emotion')
            # Check if emotion is one of the known labels
            if emotion in EMOTION_LABELS:
                valid_emotion_count += 1
                if last_valid_emotion is not None and emotion != last_valid_emotion:
                    changes += 1
                last_valid_emotion = emotion
        else:
             logging.warning(f"Skipping invalid entry in emotion_history: {entry}")

    if valid_emotion_count < 2: return 0.0
    # Avoid division by zero if only one valid emotion found after filtering
    denominator = valid_emotion_count - 1
    volatility_score = changes / denominator if denominator > 0 else 0.0

    logging.debug(f"Calculated emotion volatility: {volatility_score:.2f} ({changes} changes / {valid_emotion_count} valid points)")
    return volatility_score

def calculate_sentiment_avg_last_5(sentiment_history):
    if not isinstance(sentiment_history, list) or not sentiment_history: return 0.0
    # Filter only numeric values and take last 5
    numeric_sentiments = [s for s in sentiment_history if isinstance(s, (int, float))]
    last_5 = numeric_sentiments[-5:]
    return sum(last_5) / len(last_5) if last_5 else 0.0

def check_sentiment_emotion_mismatch(last_sentiment_text, last_emotion):
    if not last_sentiment_text or not last_emotion or last_emotion not in EMOTION_LABELS: return False
    positive_emotions = ['Happy', 'Surprise'] # Include Surprise as potentially positive context
    negative_emotions = ['Sad', 'Angry', 'Fear', 'Disgust']
    neutral_emotions = ['Neutral']
    is_mismatch = False
    if last_sentiment_text == "Positive" and last_emotion in negative_emotions: is_mismatch = True; logging.debug("Mismatch detected: Positive Text + Negative Emotion")
    elif last_sentiment_text == "Negative" and last_emotion in positive_emotions: is_mismatch = True; logging.debug("Mismatch detected: Negative Text + Positive Emotion")
    # Optional: Consider mismatch if strong sentiment and neutral emotion
    # elif last_sentiment_text != "Neutral" and last_emotion in neutral_emotions: is_mismatch = True; logging.debug("Mismatch detected: Non-Neutral Text + Neutral Emotion")
    return is_mismatch

def calculate_stress_risk_level(session_state):
    risk_score = 0.0; factors_considered = 0.0 # Use floats for division
    try:
        # Stress Level (0-10) -> Contribution 0 to 3
        stress_level = session_state.get("stress_level")
        if stress_level is not None:
            try:
                stress_val = float(stress_level)
                if 0 <= stress_val <= 10:
                    risk_score += (stress_val / 10.0) * 3.0
                    factors_considered += 3.0
                else:
                    logging.warning(f"Invalid stress level value: {stress_val}. Ignored.")
            except (ValueError, TypeError):
                logging.warning(f"Could not convert stress level '{stress_level}' to float. Ignored.")

        # Avg Sentiment (-1 to 1) -> Contribution ~0 to 1 (higher risk for negative)
        avg_sentiment = calculate_sentiment_avg_last_5(session_state.get('sentiment_scores_history', []))
        # Check if history exists to add factor weight
        if isinstance(session_state.get('sentiment_scores_history'), list) and session_state.get('sentiment_scores_history'):
            risk_score += (0.5 - avg_sentiment / 2.0) # Maps -1 to 1, 0 to 0.5, 1 to 0
            factors_considered += 1.0

        # Volatility (0 to 1) -> Contribution 0 to 1.5
        volatility = calculate_emotion_volatility(session_state.get('emotion_history', []))
        # Check if history exists to add factor weight
        if isinstance(session_state.get('emotion_history'), list) and session_state.get('emotion_history'):
             risk_score += volatility * 1.5
             factors_considered += 1.5

        # Sleep Quality (Categorical) -> Contribution 0, 0.5, 1
        sleep = session_state.get("sleep_quality")
        if sleep == 'Poor': risk_score += 1.0; factors_considered += 1.0
        elif sleep == 'Fair': risk_score += 0.5; factors_considered += 1.0
        elif sleep == 'Good': factors_considered += 1.0 # Considered but no risk points

        # Activity Level (Categorical) -> Contribution -0.3, 0, 0.3
        activity = session_state.get("activity_level")
        if activity == 'High': risk_score -= 0.3; factors_considered += 1.0
        elif activity == 'Medium': factors_considered += 1.0 # Considered
        elif activity == 'Low': risk_score += 0.3; factors_considered += 1.0

        if factors_considered == 0: return "Unknown"

        # Normalize risk score (0 to ~1 range)
        normalized_risk = max(0, risk_score) / factors_considered # Ensure non-negative before dividing
        logging.debug(f"Calculated Stress Risk: Raw={risk_score:.2f}, Factors={factors_considered}, Normalized={normalized_risk:.3f}")

        # Define thresholds based on normalized risk
        if normalized_risk >= 0.6: return "High"
        elif normalized_risk >= 0.35: return "Medium"
        else: return "Low"

    except Exception as e:
        logging.error(f"Error calculating stress risk level: {e}", exc_info=True)
        return "Calculation Error"


# --- History Fetching Function ---
def get_historical_summary(username, limit=5):
    summary = { # Initialize with defaults
        "phq9_trend": "No History", "phq9_avg": None, "gad7_trend": "No History", "gad7_avg": None,
        "stress_trend": "No History", "stress_avg": None, "sleep_trend": "No History",
        "activity_trend": "No History", "sentiment_trend": "No History", "sentiment_avg": None,
        "volatility_trend": "No History", "volatility_avg": None,
    }
    try:
        # PHQ-9 History from dedicated collection
        phq_history = list(phq_collection.find({"username": username}).sort("timestamp", DESCENDING).limit(limit + 1)) # Fetch limit+1 for trend
        if len(phq_history) >= 1:
            phq_scores_numeric = []
            for r in reversed(phq_history): # Oldest first for trend
                try: phq_scores_numeric.append(float(r['score']))
                except (ValueError, TypeError, KeyError): continue # Skip invalid entries
            if len(phq_scores_numeric) >= 1:
                summary["phq9_avg"] = round(np.mean(phq_scores_numeric), 1)
                if len(phq_scores_numeric) >= 2:
                    avg_previous = np.mean(phq_scores_numeric[:-1])
                    if phq_scores_numeric[-1] > avg_previous + 1: summary["phq9_trend"] = "Increasing"
                    elif phq_scores_numeric[-1] < avg_previous - 1: summary["phq9_trend"] = "Decreasing"
                    else: summary["phq9_trend"] = "Stable"
                else: # Only one entry
                     summary["phq9_trend"] = "First Entry"
            else: # All entries were invalid
                 summary["phq9_trend"] = "Invalid Data"; summary["phq9_avg"] = None


        # History from Session Summaries
        recent_sessions = list(session_summary_collection.find(
            {"username": username},
            {"final_gad7_score": 1, "stress_level": 1, "avg_sentiment_last_5": 1, "emotion_volatility_estimate": 1, "sleep_quality": 1, "activity_level": 1, "timestamp_saved": 1}
        ).sort("timestamp_saved", DESCENDING).limit(limit + 1)) # Fetch limit+1 for trend calc

        if not recent_sessions:
            logging.info(f"No session summaries found for user {username} to generate history.")
            return summary # Return summary with PHQ data if available

        recent_sessions.reverse() # Oldest first for trend calculation

        # Helper function to extract numeric data safely
        def extract_numeric(key):
            values = []
            for s in recent_sessions:
                val = s.get(key)
                if val is not None:
                    try: values.append(float(val))
                    except (ValueError, TypeError): pass # Ignore non-numeric
            return values

        # Helper function to calculate trend
        def calculate_trend(values, threshold=0.1):
            if len(values) < 2: return "First Entry" if len(values) == 1 else "No History"
            avg_previous = np.mean(values[:-1])
            if values[-1] > avg_previous + threshold: return "Increasing" if key != 'avg_sentiment_last_5' else "Improving"
            elif values[-1] < avg_previous - threshold: return "Decreasing" if key != 'avg_sentiment_last_5' else "Declining"
            else: return "Stable"

        # Extract and process data for each metric
        for key, avg_key, trend_key, threshold in [
            ('final_gad7_score', 'gad7_avg', 'gad7_trend', 1.0),
            ('stress_level', 'stress_avg', 'stress_trend', 0.5),
            ('avg_sentiment_last_5', 'sentiment_avg', 'sentiment_trend', 0.1),
            ('emotion_volatility_estimate', 'volatility_avg', 'volatility_trend', 0.1)
        ]:
            numeric_values = extract_numeric(key)
            if numeric_values:
                summary[avg_key] = round(np.mean(numeric_values), 2)
                summary[trend_key] = calculate_trend(numeric_values, threshold)
            else:
                summary[trend_key] = "No Valid Data" if recent_sessions else "No History"


        # Last reported Sleep/Activity (handle non-dict entries)
        sleep_qualities = [s.get('sleep_quality') for s in recent_sessions if isinstance(s, dict) and s.get('sleep_quality') is not None]
        activity_levels = [s.get('activity_level') for s in recent_sessions if isinstance(s, dict) and s.get('activity_level') is not None]
        if sleep_qualities: summary["sleep_trend"] = f"Last: {sleep_qualities[-1]}"
        if activity_levels: summary["activity_trend"] = f"Last: {activity_levels[-1]}"

    except Exception as e:
        logging.error(f"Error fetching historical summary for {username}: {e}", exc_info=True)
        # Return only trend info in case of error, avoid None values for averages
        summary_error = {k: v for k, v in summary.items() if k.endswith('_trend')}
        for k in summary.keys(): # Ensure all keys exist
             if k not in summary_error: summary_error[k] = "Error" if k.endswith("_trend") else None
        return summary_error


    logging.info(f"Generated historical summary for {username}: {json.dumps(summary, default=str)}")
    return summary


# --- Function to Save Session Summary ---
def save_session_summary(username, session_state):
    """Saves a summary of the session data to MongoDB."""
    try:
        duration_seconds = None
        start_time = session_state.get("session_start_time")
        end_time = session_state.get("session_end_time")
        if start_time and end_time: duration_seconds = (end_time - start_time).total_seconds()

        # Get calculated values, recalculate if needed or use defaults
        sentiment_history = session_state.get('sentiment_scores_history', [])
        emotion_history = session_state.get('emotion_history', [])

        # Recalculate metrics based on current session state before saving
        avg_phq_rate = calculate_average_phq9_change_rate(username) # Requires DB query
        mood_trend = calculate_recent_mood_trend(None, sentiment_history, emotion_history) # PHQ history not needed here
        emotion_volatility = calculate_emotion_volatility(emotion_history)
        avg_sentiment_last_5 = calculate_sentiment_avg_last_5(sentiment_history)
        last_valid_sentiment_text = session_state.get("last_sentiment", "Neutral")
        last_emotion = session_state.get('emotion', 'N/A')
        is_mismatch = check_sentiment_emotion_mismatch(last_valid_sentiment_text, last_emotion)
        stress_risk_level = calculate_stress_risk_level(session_state) # Use latest calculation

        phq9_final_score = session_state.get("phq9_score")
        phq9_final_severity = session_state.get("phq9_severity")
        if phq9_final_score is not None and phq9_final_severity is None:
            _, phq9_final_severity = get_phq9_interpretation(phq9_final_score)

        gad7_final_score = session_state.get("gad7_score")
        gad7_final_severity = session_state.get("gad7_severity")
        if gad7_final_score is not None and gad7_final_severity is None:
             _, gad7_final_severity = get_gad7_interpretation(gad7_final_score)

        # Get Sklearn prediction result
        predicted_status_sklearn = session_state.get("predicted_mental_health_status_sklearn")

        summary_doc = {
            "username": username, "session_start_utc": start_time, "session_end_utc": end_time, "session_duration_seconds": duration_seconds,
            "session_local_start_time": session_state.get("session_local_start_time_str", "N/A"),
            "final_phq9_score": phq9_final_score, "phq9_severity": phq9_final_severity,
            "final_gad7_score": gad7_final_score, "gad7_severity": gad7_final_severity,
            "sleep_quality": session_state.get("sleep_quality"), "activity_level": session_state.get("activity_level"),
            "stress_level": session_state.get("stress_level"), "social_interactions": session_state.get("social_interactions"),
            "final_emotion": last_emotion, "final_emotion_confidence": session_state.get("emotion_conf"),
            "final_text_sentiment": last_valid_sentiment_text, "avg_phq9_change_rate": avg_phq_rate,
            "recent_mood_trend_estimate": mood_trend, "emotion_volatility_estimate": emotion_volatility,
            "avg_sentiment_last_5": avg_sentiment_last_5, "sentiment_emotion_mismatch_flag": is_mismatch,
            "calculated_stress_risk_level": stress_risk_level,
            # Store Sklearn prediction
            "sklearn_predicted_status": predicted_status_sklearn,
            "timestamp_saved": datetime.utcnow()
        }

        # Remove keys with None values before inserting
        summary_doc_cleaned = {k: v for k, v in summary_doc.items() if v is not None}
        session_summary_collection.insert_one(summary_doc_cleaned)
        logging.info(f"Saved session summary for {username}. PHQ9 Sev: {phq9_final_severity}, GAD7 Sev: {gad7_final_severity}, Stress Risk: {stress_risk_level}, Sklearn Pred: {predicted_status_sklearn}")
    except Exception as e:
        logging.error(f"Failed to save session summary for {username}: {e}", exc_info=True)


# --- Empathetic Response Generation ---
def generate_empathetic_response(username, triggering_prompt, emotion_status_with_conf, text_sentiment, phq9_score, phq9_severity, gad7_score, gad7_severity, stress_risk_info, session_state_snapshot):
    """Generates empathetic responses based on context. (LLM logic unchanged)"""
    is_post_assessment = phq9_score is not None
    if is_post_assessment:
        logging.info(f"Generating POST-ASSESSMENT suggestions for {username}. PHQ9={phq9_score}({phq9_severity}), GAD7={gad7_score}({gad7_severity})")
        historical_data = get_historical_summary(username) # Fetch latest summary
        history_summary_prompt = f"""
        Historical Context (Recent Trends):
        - PHQ-9 Trend: {historical_data.get('phq9_trend', 'N/A')} (Avg Score: {historical_data.get('phq9_avg', 'N/A')})
        - GAD-7 Trend: {historical_data.get('gad7_trend', 'N/A')} (Avg Score: {historical_data.get('gad7_avg', 'N/A')})
        - Reported Stress Trend: {historical_data.get('stress_trend', 'N/A')} (Avg Level: {historical_data.get('stress_avg', 'N/A')})
        - Calculated Stress Risk: {stress_risk_info if stress_risk_info else 'N/A'}
        - Sleep Quality: {historical_data.get('sleep_trend', 'N/A')}
        - Activity Level: {historical_data.get('activity_trend', 'N/A')}
        - Sentiment Trend: {historical_data.get('sentiment_trend', 'N/A')} (Avg Score: {historical_data.get('sentiment_avg', 'N/A')})
        - Emotion Volatility Trend: {historical_data.get('volatility_trend', 'N/A')} (Avg Score: {historical_data.get('volatility_avg', 'N/A')})
        """
        gad7_result_text = f"GAD-7 score {gad7_score} ({str(gad7_severity).title()})" if gad7_score is not None else "GAD-7 not taken this time"
        phq9_result_text = f"PHQ-9 Score {phq9_score} ({str(phq9_severity).title()})" if phq9_score is not None else "PHQ-9 not taken this time"

        prompt_for_suggestions = f"""
        You are Serene, a compassionate AI assistant for {username}. Goal: empathetic interpretation and supportive suggestions based on recent check-in and history. DO NOT give medical advice/diagnosis. Location: {CURRENT_LOCATION_CONTEXT}.
        User Context: Prompt possibly leading to check-in: '{triggering_prompt[:100]}...'
        {history_summary_prompt}
        Current Check-in: {phq9_result_text}, {gad7_result_text}.
        Task: 1. Acknowledge completion. 2. Synthesize & Reflect empathetically on current results + history + prompt. 3. Provide 2-3 general, supportive, actionable suggestions tailored to severity/trends/context. 4. Tone: Supportive, non-judgmental. 5. Constraint: NO specific helplines/referrals/links. 6. Conclude gently.
        Generate response:
        """
        try:
            if gemini_model:
                response = gemini_model.generate_content(prompt_for_suggestions)
                reply = response.text.strip()
                logging.info(f"Generated LLM post-assessment suggestions for {username}")
                return reply
            else:
                 fallback_reply = f"Thank you for completing the check-in. Your PHQ-9 score is {phq9_score} ({phq9_severity}) and GAD-7 score is {gad7_score} ({gad7_severity}). Considering your history ({historical_data.get('phq9_trend', 'N/A')} mood trend), please continue monitoring how you feel."
                 if phq9_severity in ['moderate', 'moderately_severe', 'severe'] or gad7_severity in ['moderate', 'severe']:
                     fallback_reply += " Based on these results, considering professional support might be helpful."
                 fallback_reply += "\nHow are you feeling after reflecting on these questions?"
                 logging.warning("LLM unavailable for post-assessment suggestions, providing basic fallback.")
                 return fallback_reply
        except Exception as e:
            logging.error(f"LLM suggestion generation failed: {e}", exc_info=True)
            st.warning("AI suggestion generation failed, providing basic reply.")
            return f"Thank you for completing the check-in. Your PHQ-9 score is {phq9_score} and GAD-7 score is {gad7_score}. Please consider discussing these results and your history with a healthcare professional if you have concerns."

    else: # Standard Chat Response Mode
        logging.info(f"Generating standard empathetic response for {username}.")
        if gemini_model is None:
             logging.warning("Gemini model not available. Providing basic response.")
             base_reply = f"Thank you for sharing, {username}. I understand you mentioned: '{triggering_prompt[:50]}...'."
             if text_sentiment == "Negative": base_reply += " It sounds like that might be difficult."
             elif text_sentiment == "Positive": base_reply += " It's good to hear that."
             stress = session_state_snapshot.get("stress_level")
             if stress is not None:
                 try:
                     if float(stress) > 6: base_reply += " It seems like stress levels might be high too."
                 except (ValueError, TypeError): pass # Ignore if stress is not numeric
             base_reply += " How can I help you further?"
             return base_reply

        try:
            # Get latest status strings if available, handle potential None
            phq9_status_string = str(phq9_severity) if phq9_severity else 'PHQ-9 Status: Not Assessed Recently'
            gad7_status_string = str(gad7_severity) if gad7_severity else 'GAD-7 Status: Not Assessed Recently'
            stress_risk_prompt_context = f"Calculated Stress Risk Level: {stress_risk_info}" if stress_risk_info else "Calculated Stress Risk Level: Not Calculated"
            sleep_info = f"User reported sleep quality: {session_state_snapshot.get('sleep_quality', 'N/A')}"
            activity_info = f"User reported activity level: {session_state_snapshot.get('activity_level', 'N/A')}"
            stress_info = f"User reported stress level: {session_state_snapshot.get('stress_level', 'N/A')}/10"
            social_info = f"User reported social interactions (week): {session_state_snapshot.get('social_interactions', 'N/A')}"
            avg_sentiment = calculate_sentiment_avg_last_5(session_state_snapshot.get('sentiment_scores_history', []))
            sentiment_trend_info = f"Recent average text sentiment score: {avg_sentiment:.2f}"
            emotion_volatility = calculate_emotion_volatility(session_state_snapshot.get('emotion_history',[]))
            volatility_info = f"Recent emotion volatility score: {emotion_volatility:.2f}"
            emotion_only = str(emotion_status_with_conf).split(' (')[0] if isinstance(emotion_status_with_conf, str) else 'N/A'
            is_mismatch = check_sentiment_emotion_mismatch(text_sentiment, emotion_only)
            mismatch_info = f"Potential Sentiment/Emotion Mismatch detected: {is_mismatch}"
            prompt_emotion_status = str(emotion_status_with_conf) if isinstance(emotion_status_with_conf, str) else 'N/A'
            if emotion_only in ["No face detected", "Model Error", "Prediction Error", "Camera Error", "Capture Failed", "Detector Error", "Input Shape Error"]: prompt_emotion_status = "Facial expression could not be analyzed reliably."
            elif emotion_only == "Not Analyzed": prompt_emotion_status = "Facial expression not checked this session."
            user_message = triggering_prompt if triggering_prompt else "(User initiated conversation without specific text)"

            full_prompt = f"""
            You are Serene, a compassionate AI assistant for {username}. Goal: empathetic listening, NOT medical advice. Location: {CURRENT_LOCATION_CONTEXT}.
            Context:
            - User message: "{user_message}"
            - Detected Text Sentiment (VADER): {text_sentiment}
            - Facial Emotion Status: {prompt_emotion_status}
            - {phq9_status_string}
            - {gad7_status_string}
            - {stress_risk_prompt_context}
            - Additional Context: Sleep: {sleep_info}, Activity: {activity_info}, Reported Stress: {stress_info}, Social: {social_info}, Sentiment Trend: {sentiment_trend_info}, Emotion Volatility: {volatility_info}, Mismatch Flag: {mismatch_info}.
            Task: 1. Acknowledge user's message sincerely. 2. Respond empathetically (3-5 sentences). Validate feelings, considering context *subtly* if relevant. 3. Focus on the user's current message. 4. End with an open-ended, relevant follow-up question. 5. **Constraint:** Do NOT give medical advice or diagnosis.
            Respond now:
            """
            response = gemini_model.generate_content(full_prompt)
            reply = response.text.strip()
            logging.info(f"Generated standard LLM response for {username}")
            return reply
        except Exception as e:
            logging.error(f"LLM standard response generation failed: {e}", exc_info=True)
            st.warning("AI response generation failed, providing basic reply.")
            # Simpler fallback
            base_reply = f"Thank you for sharing, {username}. I hear you."
            if text_sentiment == "Negative": base_reply += " It sounds like things might be tough right now."
            base_reply += " Tell me more about what's on your mind?"
            return base_reply


# --- Emotion Detection Function ---
def capture_emotion():
    """Captures webcam image and predicts emotion using the loaded Keras model."""
    model, face_detector = loaded_emotion_model, loaded_face_detector
    if model is None or face_detector is None:
        status_msg = "Model Error" if model is None else "Detector Error"
        logging.warning(f"Custom emotion model/detector not loaded. Status: {status_msg}")
        return None, status_msg, "N/A"

    cap = None
    try:
        # Try opening default camera first, then fallback
        cap = cv2.VideoCapture(0)
        cam_index = 0
        if not cap.isOpened():
            logging.warning("Could not open camera at index 0, trying index 1...")
            cap.release()
            cap = cv2.VideoCapture(1)
            cam_index = 1
            if not cap.isOpened():
                logging.error("Cannot open webcam at index 0 or 1.")
                st.toast("Error: Could not open webcam.", icon="📷")
                if cap: cap.release() # Ensure release if second attempt failed but object exists
                return None, "Camera Error", "N/A"
            logging.info("Webcam opened successfully at index 1.")
        else:
             logging.info("Webcam opened successfully at index 0.")

        # Allow camera to stabilize and capture a frame
        time.sleep(1.5) # Increased delay slightly
        ret, frame = cap.read()

        if not ret or frame is None:
            logging.error(f"Failed to capture image frame from camera index {cam_index}.")
            st.toast("Error: Failed to capture image.", icon="📷")
            if cap: cap.release()
            return None, "Capture failed", "N/A"

        # Convert frame for display and processing
        image_pil_display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Always use grayscale for face detection as it's generally more robust
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            (x, y, w, h) = faces[0] # Process only the first detected face

            # Prepare ROI for the *emotion model* based on its expected input (COLOR_MODE)
            if COLOR_MODE == 'grayscale':
                face_roi_model_input = gray_frame[y:y+h, x:x+w]
                expected_channels = 1
            else: # Assuming color model expects RGB
                face_roi_model_input = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                expected_channels = 3

            # Resize and normalize
            resized_face = cv2.resize(face_roi_model_input, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0

            # Prepare input shape for the Keras model
            if COLOR_MODE == 'grayscale':
                input_face = np.expand_dims(np.expand_dims(normalized_face, axis=-1), axis=0) # (1, H, W, 1)
            else:
                input_face = np.expand_dims(normalized_face, axis=0) # (1, H, W, 3)

            # Verify input shape consistency just before prediction
            model_input_shape = tuple(dim for dim in model.input_shape if dim is not None) # e.g., (48, 48, 1)
            input_data_shape = input_face.shape[1:] # e.g., (48, 48, 1)
            if model_input_shape != input_data_shape:
                logging.error(f"Emotion model input shape mismatch! Expected: {model_input_shape}, Got: {input_data_shape}")
                if cap: cap.release()
                return image_pil_display, "Input Shape Error", "N/A"

            # Predict emotion
            try:
                emotion_prediction = model.predict(input_face, verbose=0)
                max_index = np.argmax(emotion_prediction[0])
                dominant_emotion = EMOTION_LABELS[max_index]
                confidence_score = emotion_prediction[0][max_index] * 100
                logging.info(f"Custom Model Emotion Prediction: {dominant_emotion} ({confidence_score:.1f}%)")
                if cap: cap.release()
                return image_pil_display, dominant_emotion, f"{confidence_score:.1f}%"
            except Exception as pred_e:
                logging.error(f"Emotion model prediction failed: {pred_e}", exc_info=True)
                st.toast("Error predicting emotion.", icon="⚠️")
                if cap: cap.release()
                return image_pil_display, "Prediction Error", "N/A"

        else: # No faces detected
            logging.warning("No face detected in the captured frame.")
            if cap: cap.release()
            return image_pil_display, "No face detected", "N/A"

    except Exception as e:
        logging.error(f"Webcam capture or processing error: {e}", exc_info=True)
        st.toast("Webcam/processing error.", icon="📷")
        if cap and cap.isOpened(): cap.release() # Ensure release on general error
        return None, "Capture Error", "N/A"
    finally:
        # Ensure the camera is released if it was opened
        if cap is not None and cap.isOpened():
            cap.release()
            logging.debug("Webcam released.")


# --- Scikit-learn Prediction Function (with detailed logging) ---
def predict_mental_health_with_sklearn(session_data):
    """Predicts mental health status using the loaded Scikit-learn pipeline."""
    global loaded_prediction_pipeline
    if loaded_prediction_pipeline is None:
        logging.error("Prediction cannot proceed: Scikit-learn pipeline not loaded.")
        return "Error: Model Pipeline Not Loaded"

    # Verify ALL_EXPECTED_FEATURES is populated
    if not ALL_EXPECTED_FEATURES:
         logging.error("Prediction cannot proceed: ALL_EXPECTED_FEATURES list is empty.")
         return "Error: Feature List Not Defined"

    try:
        # 1. Prepare Input Data as DataFrame
        input_data = {}
        logging.debug("--- Preparing Data for Sklearn Prediction ---")
        for col in ALL_EXPECTED_FEATURES: # Use the list defined at the top
            value = session_data.get(col, None)
            # Basic type handling/logging
            if isinstance(value, (list, np.ndarray)): # Handle accidental lists
                 input_data[col] = [value[0]] if len(value)>0 else [None]
                 logging.debug(f"Feature '{col}': Handled list/array input, value={input_data[col][0]}")
            else:
                 input_data[col] = [value]
                 logging.debug(f"Feature '{col}': Value = {value} (Type: {type(value)})")

            # Warning for potentially important missing values (pipeline should still impute)
            if value is None and col in ['final_phq9_score', 'final_gad7_score', 'stress_level']:
                 logging.warning(f"Potentially important feature '{col}' has missing value (None). Pipeline imputation will be used.")

        # Create DataFrame with the exact columns the pipeline expects (order matters here if not handled by ColumnTransformer names)
        input_df = pd.DataFrame(input_data, columns=ALL_EXPECTED_FEATURES)

        # --- Detailed Logging ---
        logging.info("Input DataFrame for sklearn prediction (first row):")
        try:
             logging.info(f"\n{input_df.head().to_string()}")
             logging.info("Input DataFrame dtypes:")
             logging.info(f"\n{input_df.dtypes}")
        except Exception as log_e:
             logging.error(f"Error logging input_df details: {log_e}")
        # --- END Detailed Logging ---

        # 2. Predict using the loaded pipeline
        logging.info("Calling loaded_prediction_pipeline.predict()...")
        predictions = loaded_prediction_pipeline.predict(input_df)
        logging.info(f"Prediction successful. Raw prediction output: {predictions}")
        predicted_label = predictions[0]

        logging.info(f"Predicted Mental Health Status (Scikit-learn): {predicted_label}")
        return predicted_label

    except Exception as e:
        logging.error(f"!!! Error during Scikit-learn prediction: {e}", exc_info=True)
        error_str = str(e).lower()
        # Check for common errors based on traceback/message
        if "columns are missing" in error_str or "feature names mismatch" in error_str:
             logging.error("Prediction failed due to feature mismatch (names/count) between input and trained pipeline.")
             # Extract missing/unexpected columns if possible from error message
             try:
                 details = str(e).split(":")[-1].strip()
                 return f"Error: Feature Mismatch ({details})"
             except:
                 return "Error: Feature Mismatch"
        elif "found unknown categories" in error_str:
             logging.error("Prediction failed due to unknown categorical values not seen during training.")
             return "Error: Unknown Category Encountered"
        elif "could not convert string to float" in error_str or "invalid literal for int" in error_str:
            logging.error("Prediction failed due to data type mismatch (e.g., string in numerical column).")
            return "Error: Data Type Mismatch"
        # Generic fallback
        return "Error: Prediction Failed"
# --- End Scikit-learn Prediction Function ---


# ==============================================================================
# Streamlit Application UI and Logic
# ==============================================================================

# --- Session State Defaults ---
default_session_state = {
    "authenticated": False, "username": None, "conversation": [],
    "emotion": "Not Analyzed", "emotion_conf": "N/A", "image": None,
    "last_sentiment": "Neutral",
    "session_start_time": None, "session_end_time": None, "session_local_start_time_str": None,
    "sleep_quality": None, "activity_level": None, "social_interactions": None, "stress_level": None,
    "sentiment_scores_history": [], "emotion_history": [], # Ensure these are lists
    "phq9_consent_pending": False, "phq9_active": False, "phq9_current_q": 0,
    "phq9_answers": {}, "phq9_score": None, "phq9_severity": None, "phq9_high_alert_q9": False,
    "gad7_active": False, "gad7_current_q": 0, "gad7_answers": {}, "gad7_score": None, "gad7_severity": None,
    "triggering_prompt": None, "resource_message_shown": False,
    "predicted_mental_health_status_sklearn": None, # For storing Sklearn prediction result
    "stress_risk_level": None # Keep existing calculated risk
}
# Initialize state robustly
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        # Create copies for mutable defaults
        if isinstance(default_value, list): st.session_state[key] = list(default_value)
        elif isinstance(default_value, dict): st.session_state[key] = dict(default_value)
        else: st.session_state[key] = default_value

# --- Sidebar ---
st.sidebar.title("👤 User")
if st.session_state.authenticated:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    st.sidebar.markdown("---"); st.sidebar.subheader("📝 Quick Check-in")
    # Use unique keys for widgets inside the form to avoid conflicts
    with st.sidebar.form(key="check_in_form"):
        sleep_options_base = ["Not Logged", "Poor", "Fair", "Good"]
        def format_sleep_option(option):
            return {"Poor": "Poor (< 6 hrs)", "Fair": "Fair (6-7 hrs)", "Good": "Good (7+ hrs)"}.get(option, option)

        # Determine current index safely
        current_sleep_index = 0
        if st.session_state.sleep_quality in sleep_options_base:
            current_sleep_index = sleep_options_base.index(st.session_state.sleep_quality)
        st.radio("Sleep last night?", options=sleep_options_base, index=current_sleep_index, format_func=format_sleep_option, key="sidebar_sleep_input", horizontal=True)

        activity_options = ["Not Logged", "Low", "Medium", "High"]
        current_activity_value = st.session_state.activity_level if st.session_state.activity_level in activity_options else "Not Logged"
        st.select_slider("Activity level recently?", options=activity_options, value=current_activity_value, key="sidebar_activity_input")

        # Handle potential non-numeric stress level safely
        current_stress_level = 0
        try:
            stress_val = st.session_state.stress_level
            if stress_val is not None:
                 current_stress_level = int(float(stress_val))
                 if not (0 <= current_stress_level <= 10): current_stress_level = 0 # Reset if out of range
        except (ValueError, TypeError):
             current_stress_level = 0 # Default to 0 if conversion fails
        st.slider("Current stress (1-10)?", min_value=0, max_value=10, value=current_stress_level, key="sidebar_stress_input", help="0 = skip")

        # Handle potential non-numeric social interactions safely
        current_social_interactions = -1
        try:
            social_val = st.session_state.social_interactions
            if social_val is not None:
                 current_social_interactions = int(float(social_val))
                 if current_social_interactions < -1 : current_social_interactions = -1 # Reset if invalid
        except (ValueError, TypeError):
             current_social_interactions = -1 # Default to -1 if conversion fails
        st.number_input("Meaningful interactions (week)?", min_value=-1, step=1, value=current_social_interactions, key="sidebar_social_input", help="-1 = skip")

        submitted = st.form_submit_button("📊 Update Check-in")
        if submitted:
            logging.info(f"Check-in form submitted by {st.session_state.username}")
            # Update session state based on form inputs
            st.session_state.sleep_quality = None if st.session_state.sidebar_sleep_input == "Not Logged" else st.session_state.sidebar_sleep_input
            st.session_state.activity_level = None if st.session_state.sidebar_activity_input == "Not Logged" else st.session_state.sidebar_activity_input
            st.session_state.stress_level = None if st.session_state.sidebar_stress_input == 0 else st.session_state.sidebar_stress_input
            st.session_state.social_interactions = None if st.session_state.sidebar_social_input == -1 else st.session_state.sidebar_social_input
            st.session_state.stress_risk_level = calculate_stress_risk_level(st.session_state) # Recalculate risk
            st.toast("Check-in details updated!", icon="✅")
            st.rerun() # Rerun to reflect changes immediately

    # Show Session Data Button
    st.sidebar.markdown("---")
    if st.sidebar.button("📄 Show Current Session Data"):
        with st.sidebar.expander("Current Session Data Snapshot", expanded=True):
            st.markdown("**User Inputs:**")
            st.write(f"- Sleep Quality: `{st.session_state.get('sleep_quality', 'N/A')}`")
            st.write(f"- Activity Level: `{st.session_state.get('activity_level', 'N/A')}`")
            st.write(f"- Stress Level (Self-Reported): `{st.session_state.get('stress_level', 'N/A')}` / 10")
            st.write(f"- Social Interactions (week): `{st.session_state.get('social_interactions', 'N/A')}`")
            st.markdown("**Analysis Results:**")
            st.write(f"- Last Emotion: `{st.session_state.get('emotion', 'N/A')}` ({st.session_state.get('emotion_conf', 'N/A')})")
            st.write(f"- Last Text Sentiment: `{st.session_state.get('last_sentiment', 'N/A')}`")
            st.markdown("**Check-in Status:**")
            latest_phq_score = st.session_state.get("phq9_score")
            latest_phq_severity = st.session_state.get("phq9_severity", "N/A")
            if latest_phq_score is not None and latest_phq_severity == "N/A": _, latest_phq_severity = get_phq9_interpretation(latest_phq_score)
            st.write(f"- Depression (PHQ-9): Score `{latest_phq_score if latest_phq_score is not None else 'N/A'}`, Severity `{latest_phq_severity}`")
            latest_gad7_score = st.session_state.get("gad7_score")
            latest_gad7_severity = st.session_state.get("gad7_severity", "N/A")
            if latest_gad7_score is not None and latest_gad7_severity == "N/A": _, latest_gad7_severity = get_gad7_interpretation(latest_gad7_score)
            st.write(f"- Anxiety (GAD-7): Score `{latest_gad7_score if latest_gad7_score is not None else 'N/A'}`, Severity `{latest_gad7_severity}`")
            st.markdown("**Calculated Metrics:**")
            # Recalculate metrics for display
            current_volatility = calculate_emotion_volatility(st.session_state.get('emotion_history', []))
            current_avg_sentiment = calculate_sentiment_avg_last_5(st.session_state.get('sentiment_scores_history', []))
            current_mismatch = check_sentiment_emotion_mismatch(st.session_state.get('last_sentiment', 'N/A'), st.session_state.get('emotion', 'N/A'))
            current_stress_risk = calculate_stress_risk_level(st.session_state)
            st.write(f"- Emotion Volatility (Session): `{current_volatility:.2f}`")
            st.write(f"- Avg Sentiment Score (Recent): `{current_avg_sentiment:.2f}`")
            st.write(f"- Sentiment/Emotion Mismatch Flag: `{current_mismatch}`")
            st.write(f"- Rule-Based Stress Risk Level: `{current_stress_risk}`")
            st.markdown("**AI Prediction:**")
            st.write(f"- Scikit-learn Predicted Status: `{st.session_state.get('predicted_mental_health_status_sklearn', 'N/A')}`")
            st.markdown("**Session Info:**")
            start_time = st.session_state.get("session_start_time")
            if start_time:
                 now_utc = datetime.utcnow()
                 if isinstance(start_time, datetime):
                      duration = now_utc - start_time
                      duration_str = str(duration).split('.')[0]
                      st.write(f"- Started (UTC): `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`")
                      st.write(f"- Duration: `{duration_str}`")
                 else:
                      st.write(f"- Started: `{start_time}` (Invalid DateTime)")
                      st.write("- Duration: `Cannot calculate`")
            else: st.write("- Session Start Time: `N/A`")


    # Emotion Check Button
    st.sidebar.markdown("---"); st.sidebar.subheader("😊 Emotion Check")
    is_assessment_active = (st.session_state.get("phq9_consent_pending", False) or
                           st.session_state.get("phq9_active", False) or
                           st.session_state.get("gad7_active", False))
    emotion_model_ready = (loaded_emotion_model is not None and loaded_face_detector is not None)
    emotion_button_disabled = (is_assessment_active or not emotion_model_ready)
    reason = ""
    if not emotion_model_ready: reason = " (Model unavailable)"
    elif is_assessment_active: reason = " (Disabled during check-in)"

    if st.sidebar.button("📸 Analyze My Expression", disabled=emotion_button_disabled, help=f"Uses webcam.{reason}"):
         with st.spinner("Analyzing..."):
            img, emo, conf = capture_emotion()
            st.session_state.image = img
            st.session_state.emotion = emo # Store latest emotion status/error
            st.session_state.emotion_conf = conf # Store confidence or N/A

            # Only add valid emotions to history
            if emo in EMOTION_LABELS:
                # Ensure history is a list
                if not isinstance(st.session_state.emotion_history, list):
                    st.session_state.emotion_history = []
                st.session_state.emotion_history.append({"emotion": emo, "confidence": conf, "timestamp": datetime.utcnow()})
                st.session_state.emotion_history = st.session_state.emotion_history[-MAX_HISTORY_LEN:]
                st.session_state.stress_risk_level = calculate_stress_risk_level(st.session_state) # Recalculate risk on new emotion data

            st.rerun() # Rerun to update sidebar display immediately

    # Display last captured image and emotion status
    last_img = st.session_state.get("image")
    last_emo = st.session_state.get("emotion", "Not Analyzed")
    last_conf = st.session_state.get("emotion_conf", "N/A")
    if last_img:
        caption_text = f"Last: {last_emo}" + (f" ({last_conf})" if last_conf != "N/A" else "")
        st.sidebar.image(last_img, width=150, caption=caption_text)
        if last_emo not in ["Not Analyzed", "Camera Error", "Capture failed", "Model Error", "Detector Error", "Prediction Error", "Input Shape Error", "No face detected", "Capture Error"]:
             st.sidebar.metric("Last Emotion", f"{last_emo}", f"{last_conf}")
        else:
             st.sidebar.metric("Last Emotion", f"{last_emo}", "") # No delta for errors/failures
    else:
        st.sidebar.info("Click button to analyze expression." + reason)

    # PHQ-9 History Chart
    st.sidebar.markdown("---"); st.sidebar.subheader("📉 PHQ-9 History")
    df_hist = load_phq9_history(st.session_state.username)
    if not df_hist.empty:
        try:
             df_chart = df_hist.set_index("timestamp")
             st.sidebar.line_chart(df_chart["score"])
             latest_score = int(df_hist.iloc[-1]["score"])
             _, latest_severity = get_phq9_interpretation(latest_score)
             st.sidebar.metric("Latest PHQ-9 Score", f"{latest_score}", f"{latest_severity.replace('_',' ').title()}")
        except Exception as chart_err:
             st.sidebar.warning(f"Could not display PHQ-9 chart: {chart_err}")
             logging.error(f"Error displaying PHQ-9 chart: {chart_err}", exc_info=True)
    else: st.sidebar.info("No PHQ-9 history yet.")

    # Logout Button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔓 Logout"):
        logging.info(f"User '{st.session_state.username}' logging out.")
        st.session_state.session_end_time = datetime.utcnow()
        # Save summary before clearing state
        save_session_summary(st.session_state.username, st.session_state)
        username_logged_out = st.session_state.username
        # Clear session state robustly
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            try: del st.session_state[key]
            except (AttributeError, KeyError): st.session_state[key] = None # Fallback
        # Restore essential keys for login page
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.conversation = [] # Ensure conversation is reset

        st.success(f"Successfully logged out {username_logged_out}."); time.sleep(1.5); st.rerun()


# --- Login / Signup Screen ---
else: # Not authenticated
    st.title("Welcome to Serene - Your AI Companion")
    st.caption(f"AI for mental well-being support. Context: {CURRENT_LOCATION_CONTEXT}. Please login or sign up.")
    login_tab, signup_tab = st.tabs(["🔑 Login", "📝 Sign Up"])
    with login_tab:
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_username_input")
            login_pass = st.text_input("Password", type="password", key="login_password_input")
            login_button = st.form_submit_button("Login")
            if login_button:
                if not login_user or not login_pass: st.warning("Enter username and password.")
                else:
                    success, msg = login(login_user, login_pass)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = login_user
                        # Initialize session state completely on login
                        for key, default_value in default_session_state.items():
                            if key not in ["authenticated", "username"]: # Keep logged in user
                                if isinstance(default_value, list): st.session_state[key] = list(default_value)
                                elif isinstance(default_value, dict): st.session_state[key] = dict(default_value)
                                else: st.session_state[key] = default_value
                        # Set specific initial states
                        st.session_state.conversation = [{"role": "assistant", "content": f"Hello {login_user}! How are you feeling today?"}]
                        st.session_state.session_start_time = datetime.utcnow()
                        try: st.session_state.session_local_start_time_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                        except Exception as tz_err:
                             logging.warning(f"Could not get local timezone: {tz_err}")
                             st.session_state.session_local_start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (Timezone Unknown)"
                        st.session_state.resource_message_shown = False # Reset this flag
                        st.session_state.predicted_mental_health_status_sklearn = None # Clear prediction

                        st.success(msg); time.sleep(1); st.rerun()
                    else: st.error(msg)
    with signup_tab:
        with st.form("signup_form"):
            signup_user = st.text_input("New Username", key="signup_username_input")
            signup_email = st.text_input("Email", key="signup_email_input")
            signup_pass = st.text_input("Password (min 6 chars)", type="password", key="signup_password_input")
            signup_confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm_password_input")
            signup_button = st.form_submit_button("Sign Up")
            if signup_button:
                if not all([signup_user, signup_email, signup_pass, signup_confirm_pass]): st.warning("Fill all fields.")
                elif signup_pass != signup_confirm_pass: st.error("Passwords don't match.")
                else:
                    success, msg = signup(signup_user, signup_pass, signup_email)
                    if success: st.success(msg)
                    else: st.error(msg)


# --- Main Interface (Authenticated User) ---
if st.session_state.authenticated:

    st.title("🤖 Serene: AI Mental Health Assistant")
    st.caption(f"Logged in as {st.session_state.username}. Context: {CURRENT_LOCATION_CONTEXT}. Remember, I'm not a substitute for professional help.")

    # --- Display Chat History ---
    chat_container = st.container()
    with chat_container:
        if "conversation" in st.session_state and isinstance(st.session_state.conversation, list):
             for msg in st.session_state.conversation:
                 if isinstance(msg, dict) and "role" in msg and "content" in msg:
                     with st.chat_message(msg["role"]):
                         # Render markdown if content seems like it, otherwise plain text
                         content = msg["content"]
                         if isinstance(content, str) and ('*' in content or '#' in content or '`' in content or '\n' in content):
                              st.markdown(content, unsafe_allow_html=False)
                         else:
                              st.text(str(content)) # Convert non-strings
                 else:
                      logging.warning(f"Skipping invalid message format in conversation: {msg}")
        else:
             st.session_state.conversation = [] # Initialize if missing or wrong type


    # --- Combined Consent Section Logic ---
    consent_container = st.container()
    with consent_container:
        # Use .get() for safer access to session state keys
        if st.session_state.get("phq9_consent_pending", False):
            st.markdown("---")
            st.info("I sense you might be going through a difficult time. To understand better, would you be open to answering some brief check-in questions about mood and anxiety (PHQ-9 & GAD-7)? It's completely optional and based on how you've felt over the **last 2 weeks**.")
            col1, col2, col3 = st.columns([1.5, 1.5, 4]) # Adjust column widths if needed
            with col1:
                if st.button("✅ Yes, start Check-in", key="checkin_consent_yes"):
                    logging.info(f"User {st.session_state.username} consented to combined PHQ-9/GAD-7 check-in.")
                    # Reset relevant states before starting
                    st.session_state.phq9_consent_pending = False; st.session_state.phq9_active = True; st.session_state.gad7_active = False
                    st.session_state.phq9_current_q = 0; st.session_state.phq9_answers = {}; st.session_state.phq9_score = None; st.session_state.phq9_severity = None; st.session_state.phq9_high_alert_q9 = False
                    st.session_state.gad7_current_q = 0; st.session_state.gad7_answers = {}; st.session_state.gad7_score = None; st.session_state.gad7_severity = None
                    st.session_state.predicted_mental_health_status_sklearn = None # Reset prediction
                    start_message = "Okay, let's begin the check-in. First, some questions about mood (PHQ-9). Please answer based on the **last 2 weeks**."
                    st.session_state.conversation.append({"role": "assistant", "content": start_message})
                    save_message(st.session_state.username, "assistant", start_message)
                    st.rerun()
            with col2:
                if st.button("❌ No Check-in now", key="checkin_consent_no"):
                    st.session_state.phq9_consent_pending = False; st.session_state.phq9_active = False; st.session_state.gad7_active = False
                    decline_message = "Okay, no problem at all. We don't have to do the check-in now. Let's continue our chat."
                    st.session_state.conversation.append({"role": "assistant", "content": decline_message})
                    save_message(st.session_state.username, "assistant", decline_message)
                    logging.info(f"User {st.session_state.username} declined combined check-in.")
                    original_prompt = st.session_state.get("triggering_prompt") # Use .get()
                    if original_prompt:
                        with st.spinner("Okay, let's get back to what you were saying..."):
                            # Use .get() for safer access
                            emotion_context = f"{st.session_state.get('emotion', 'N/A')} ({st.session_state.get('emotion_conf', 'N/A')})"
                            phq9_context_latest = "Check-in (PHQ-9) declined"
                            gad7_context_latest = "Check-in (GAD-7) declined"
                            stress_risk = calculate_stress_risk_level(st.session_state)
                            stress_risk_context = f"Stress Risk: {stress_risk}" if stress_risk != "Unknown" else None
                            assistant_reply = generate_empathetic_response(
                                st.session_state.username, original_prompt, emotion_context,
                                st.session_state.get("last_sentiment", "Neutral"),
                                None, phq9_context_latest, None, gad7_context_latest,
                                stress_risk_context, st.session_state
                             )
                            st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
                            save_message(st.session_state.username, "assistant", assistant_reply)
                            st.session_state.triggering_prompt = None # Clear the prompt after use
                    st.rerun()
            st.markdown("---")


    # --- Combined Assessment Questionnaire Section ---
    assessment_question_container = st.container()
    with assessment_question_container:

        # --- PHQ-9 Questionnaire Display ---
        if st.session_state.get("phq9_active", False):
            st.markdown("---")
            st.subheader("Check-in Part 1: Mood (PHQ-9)")
            current_q_index = st.session_state.get("phq9_current_q", 0)
            if current_q_index < len(PHQ9_QUESTIONS):
                question = PHQ9_QUESTIONS[current_q_index]
                q_id = question['id']; q_text = question['text']
                st.markdown(f"**1.{current_q_index + 1}/{len(PHQ9_QUESTIONS)}. Over the last 2 weeks, how often bothered by: {q_text}?**")
                # Ensure unique keys for widgets in loops/dynamic UI
                with st.form(key=f"phq9_form_{q_id}"):
                    answer_text = st.radio("Select:", options=PHQ9_OPTION_LIST, key=f"phq9_radio_{q_id}", horizontal=True, label_visibility="collapsed")
                    submitted = st.form_submit_button("Next Question")
                    if submitted:
                        answer_score = PHQ9_OPTIONS[answer_text]
                        # Ensure answers dict exists
                        if not isinstance(st.session_state.get("phq9_answers"), dict):
                            st.session_state.phq9_answers = {}
                        st.session_state.phq9_answers[q_id] = answer_score
                        logging.debug(f"User {st.session_state.username} PHQ-9 Q{current_q_index+1} ({q_id}) score {answer_score}")
                        if q_id == 'q9' and answer_score > 0: st.session_state.phq9_high_alert_q9 = True
                        st.session_state.phq9_current_q += 1
                        st.rerun()
            else: # PHQ-9 Completion -> Transition to GAD-7
                final_phq9_score = sum(st.session_state.get("phq9_answers", {}).values()) # Use .get() with default
                _, final_phq9_severity = get_phq9_interpretation(final_phq9_score)
                st.session_state.phq9_score = final_phq9_score
                st.session_state.phq9_severity = final_phq9_severity
                save_phq9_score(st.session_state.username, final_phq9_score)
                logging.info(f"PHQ-9 part completed for {st.session_state.username}. Score: {final_phq9_score}, Severity: {final_phq9_severity}.")
                transition_message = "Thanks for answering those. Now, just a few questions about anxiety (GAD-7), also based on the **last 2 weeks**."
                st.session_state.conversation.append({"role": "assistant", "content": transition_message})
                save_message(st.session_state.username, "assistant", transition_message)
                st.session_state.phq9_active = False
                st.session_state.gad7_active = True
                st.session_state.gad7_current_q = 0
                st.session_state.gad7_answers = {} # Reset GAD-7 answers
                time.sleep(0.5); st.rerun()
            st.markdown("---")

        # --- GAD-7 Questionnaire Display ---
        elif st.session_state.get("gad7_active", False):
            st.markdown("---")
            st.subheader("Check-in Part 2: Anxiety (GAD-7)")
            current_q_index = st.session_state.get("gad7_current_q", 0)
            if current_q_index < len(GAD7_QUESTIONS):
                question = GAD7_QUESTIONS[current_q_index]
                q_id = question['id']; q_text = question['text']
                st.markdown(f"**2.{current_q_index + 1}/{len(GAD7_QUESTIONS)}. Over the last 2 weeks, how often bothered by: {q_text}?**")
                with st.form(key=f"gad7_form_{q_id}"):
                    answer_text = st.radio("Select:", options=GAD7_OPTION_LIST, key=f"gad7_radio_{q_id}", horizontal=True, label_visibility="collapsed")
                    submitted = st.form_submit_button("Next Question")
                    if submitted:
                        answer_score = GAD7_OPTIONS[answer_text]
                         # Ensure answers dict exists
                        if not isinstance(st.session_state.get("gad7_answers"), dict):
                            st.session_state.gad7_answers = {}
                        st.session_state.gad7_answers[q_id] = answer_score
                        logging.debug(f"User {st.session_state.username} GAD-7 Q{current_q_index+1} ({q_id}) score {answer_score}")
                        st.session_state.gad7_current_q += 1
                        st.rerun()
            else: # GAD-7 Completion -> Final Processing
                final_gad7_score = sum(st.session_state.get("gad7_answers", {}).values()) # Use .get() with default
                _, final_gad7_severity = get_gad7_interpretation(final_gad7_score)
                st.session_state.gad7_score = final_gad7_score
                st.session_state.gad7_severity = final_gad7_severity
                logging.info(f"GAD-7 part completed for {st.session_state.username}. Score: {final_gad7_score}, Severity: {final_gad7_severity}.")

                # --- Generate History-Informed LLM Suggestions ---
                with st.spinner("Analyzing results and preparing suggestions..."):
                    final_phq9_score = st.session_state.get("phq9_score")
                    final_phq9_severity = st.session_state.get("phq9_severity")
                    triggering_prompt = st.session_state.get("triggering_prompt", "")
                    stress_risk = calculate_stress_risk_level(st.session_state)
                    emotion_context = f"{st.session_state.get('emotion', 'N/A')} ({st.session_state.get('emotion_conf', 'N/A')})"
                    llm_suggestions = generate_empathetic_response(
                        st.session_state.username, triggering_prompt, emotion_context,
                        None, # No current text sentiment at this stage
                        final_phq9_score, final_phq9_severity,
                        final_gad7_score, final_gad7_severity,
                        stress_risk, st.session_state
                    )
                    if llm_suggestions:
                        st.session_state.conversation.append({"role": "assistant", "content": llm_suggestions})
                        save_message(st.session_state.username, "assistant", llm_suggestions)
                    else:
                        fallback_msg = "Thank you for completing the check-in."
                        st.session_state.conversation.append({"role": "assistant", "content": fallback_msg})
                        save_message(st.session_state.username, "assistant", fallback_msg)

                # --- Scikit-learn Model Prediction ---
                with st.spinner("Analyzing trends and predicting status..."):
                     # Prepare data dictionary directly from session state for prediction function
                     # The function itself will select the required features based on ALL_EXPECTED_FEATURES
                     current_features_data = dict(st.session_state) # Pass a copy
                     # Ensure the final scores are included correctly
                     current_features_data['final_phq9_score'] = final_phq9_score
                     current_features_data['final_gad7_score'] = final_gad7_score
                     # Recalculate/update features that might change during check-in
                     current_features_data['emotion_volatility_estimate'] = calculate_emotion_volatility(st.session_state.get('emotion_history', []))
                     current_features_data['avg_sentiment_last_5'] = calculate_sentiment_avg_last_5(st.session_state.get('sentiment_scores_history', []))
                     current_features_data['final_text_sentiment'] = st.session_state.get("last_sentiment", "Neutral")
                     current_features_data['final_emotion'] = st.session_state.get("emotion", "N/A") # Use latest emotion

                     predicted_status_sklearn = predict_mental_health_with_sklearn(current_features_data)

                     # Define status descriptions (MUST MATCH LABELS IN train.py's TARGET_COLUMN)
                     status_descriptions = {
                         "coping": "Generally managing daily stressors, though may experience some mild symptoms.",
                         "struggling": "Likely experiencing significant symptoms affecting daily functioning; professional support could be beneficial.",
                         "thriving": "Indicating overall positive well-being with minimal reported symptoms.",
                         "vulnerable": "Experiencing some symptoms or facing factors that might increase risk; preventative strategies or support could be helpful.",
                         "crisis_risk": "Suggesting severe distress where seeking immediate professional help is strongly recommended."
                         # Add/modify labels based on your train.py output and TARGET_COLUMN unique values
                     }

                     prediction_message = "" # Initialize message
                     if isinstance(predicted_status_sklearn, str) and "Error:" not in predicted_status_sklearn:
                         status_desc = status_descriptions.get(predicted_status_sklearn, "Description not available for this status.")
                         prediction_message = f"""
                         **AI Prediction (KNN Model):** Based on the current session and recent trends, the analysis suggests a status of **'{predicted_status_sklearn}'**.

                         *General meaning: {status_desc}*

                         Remember, this is an AI prediction and not a diagnosis.
                         """
                         st.session_state.predicted_mental_health_status_sklearn = predicted_status_sklearn
                     elif isinstance(predicted_status_sklearn, str) and "Error:" in predicted_status_sklearn:
                          prediction_message = f"**AI Prediction (KNN Model):** Could not generate a prediction. ({predicted_status_sklearn})"
                          st.session_state.predicted_mental_health_status_sklearn = None # Clear prediction on error
                          logging.error(f"Prediction failed with message: {predicted_status_sklearn}")
                     else:
                         prediction_message = "**AI Prediction (KNN Model):** An unexpected result occurred during prediction."
                         logging.warning(f"Unexpected predicted_status_sklearn value or type: {predicted_status_sklearn} ({type(predicted_status_sklearn)})")
                         st.session_state.predicted_mental_health_status_sklearn = None

                     if prediction_message: # Add message only if it was generated
                          st.session_state.conversation.append({"role": "assistant", "content": prediction_message})
                          save_message(st.session_state.username, "assistant", prediction_message)


                # --- Conditional Resource Display Logic ---
                phq9_is_severe = st.session_state.get("phq9_severity") == 'severe'
                gad7_is_severe = final_gad7_severity == 'severe'
                q9_alert = st.session_state.get("phq9_high_alert_q9", False)
                # Make resource message flag user-specific
                resource_message_shown_key = f"resource_message_shown_{st.session_state.username}"

                if (phq9_is_severe or gad7_is_severe or q9_alert) and not st.session_state.get(resource_message_shown_key, False):
                    logging.info(f"Displaying resources for {st.session_state.username} due to severe score or Q9 alert.")
                    alert_reason = "severe symptoms" if (phq9_is_severe or gad7_is_severe) else "your answer to the question about self-harm thoughts"
                    resource_message = f"\n**Given {alert_reason}, reaching out for professional support is highly recommended, especially if these feelings persist.**\n\n"
                    resource_message += "**Immediate Support & Helplines (India):**\n"
                    for name, number in INDIA_MENTAL_HEALTH_HELPLINES.items(): resource_message += f"* **{name}:** {number}\n"
                    resource_message += "\n**Finding Professionals:**\n" + GENERAL_SEARCH_ADVICE_INDIA + "\n"
                    resource_message += "**Please reach out to connect with help. You don't have to go through this alone.**\n"
                    st.session_state.conversation.append({"role": "assistant", "content": resource_message})
                    save_message(st.session_state.username, "assistant", resource_message)
                    st.session_state[resource_message_shown_key] = True # Set flag after showing


                # --- Final Reset Logic for Assessments ---
                st.session_state.phq9_active = False; st.session_state.phq9_current_q = 0; st.session_state.phq9_answers = {}; st.session_state.phq9_high_alert_q9 = False
                st.session_state.gad7_active = False; st.session_state.gad7_current_q = 0; st.session_state.gad7_answers = {}
                st.session_state.phq9_consent_pending = False
                st.session_state.triggering_prompt = None # Clear prompt after check-in completes
                # Keep predicted_mental_health_status_sklearn until next prediction or logout

                time.sleep(0.5); st.rerun() # Rerun to clear assessment UI and show results
            st.markdown("---")


    # --- Chat Input Area ---
    chat_input_disabled = (
        st.session_state.get("phq9_consent_pending", False) or
        st.session_state.get("phq9_active", False) or
        st.session_state.get("gad7_active", False)
    )
    prompt = st.chat_input("How are you feeling? Type here...", disabled=chat_input_disabled, key="chat_input_main")

    if prompt and not chat_input_disabled:
        logging.info(f"User '{st.session_state.username}' inputted: '{prompt[:50]}...'")
        st.session_state.conversation.append({"role": "user", "content": prompt})
        save_message(st.session_state.username, "user", prompt)
        with st.spinner("Analyzing message..."):
             detected_sentiment_text, detected_sentiment_score = analyze_sentiment(prompt)
             st.session_state.last_sentiment = detected_sentiment_text

        # Trigger check-in offer ONLY if no assessment is active/pending and sentiment is Negative
        if detected_sentiment_text == "Negative" and not st.session_state.get("phq9_active") and not st.session_state.get("gad7_active") and not st.session_state.get("phq9_consent_pending"):
            logging.info(f"Negative sentiment detected for {st.session_state.username}. Offering combined PHQ-9/GAD-7 check-in.")
            st.session_state.phq9_consent_pending = True
            st.session_state.triggering_prompt = prompt # Store the prompt that triggered the check-in
            # The consent UI will be shown on rerun
            st.rerun()
        else:
            # Generate standard empathetic response only if no assessment flow is active/pending
            if not st.session_state.get("phq9_consent_pending") and not st.session_state.get("phq9_active") and not st.session_state.get("gad7_active"):
                with st.spinner("Thinking..."):
                    emotion_context = f"{st.session_state.get('emotion', 'N/A')} ({st.session_state.get('emotion_conf', 'N/A')})"
                    # Get latest scores/severities safely using .get()
                    phq9_score_state = st.session_state.get("phq9_score")
                    phq9_severity_state = st.session_state.get("phq9_severity")
                    phq9_context_latest = f"Latest PHQ-9: Score {phq9_score_state} ({str(phq9_severity_state).title()})" if phq9_score_state is not None else "Latest PHQ-9: Not Assessed Recently"

                    gad7_score_state = st.session_state.get("gad7_score")
                    gad7_severity_state = st.session_state.get("gad7_severity")
                    gad7_context_latest = f"Latest GAD-7: Score {gad7_score_state} ({str(gad7_severity_state).title()})" if gad7_score_state is not None else "Latest GAD-7: Not Assessed Recently"

                    stress_risk = calculate_stress_risk_level(st.session_state)
                    stress_risk_context = f"Calculated Stress Risk: {stress_risk}" if stress_risk != "Unknown" else None

                    # Call generate_empathetic_response in standard mode (phq9_score=None triggers this)
                    assistant_reply = generate_empathetic_response(
                        username=st.session_state.username, triggering_prompt=prompt,
                        emotion_status_with_conf=emotion_context, text_sentiment=detected_sentiment_text,
                        phq9_score=None, phq9_severity=phq9_context_latest, # Pass status strings
                        gad7_score=None, gad7_severity=gad7_context_latest, # Pass status strings
                        stress_risk_info=stress_risk_context, session_state_snapshot=st.session_state
                    )
                    st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
                    save_message(st.session_state.username, "assistant", assistant_reply)
                    st.rerun() # Display new messages

# END OF FULL CODE: hello.py (Corrected Feature Lists and Prediction Logic)