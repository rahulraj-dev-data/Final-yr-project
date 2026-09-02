# START OF FULL CODE: predict_samples.py

import pandas as pd
import joblib # To load the pipeline
import sys
import numpy as np # Might be needed for np.nan or specific numeric types

# --- Configuration (MUST MATCH THE CORRECTED train.py) ---
# --- Path to the pipeline saved by the CORRECTED train.py ---
SAVED_PIPELINE_FILENAME = 'sklearn_knn_mental_health_pipeline.joblib'

# --- Intended Features (MUST MATCH THE CORRECTED train.py) ---
# **>>> VERIFY/ADJUST THIS LIST TO MATCH YOUR CORRECTED train.py <<<**
intended_feature_columns = [
    'final_phq9_score', 'final_gad7_score', 'stress_level',
    'social_interactions', 'emotion_volatility_estimate',
    'avg_sentiment_last_5', 'sleep_quality', 'activity_level',
    'final_emotion', 'final_text_sentiment'
    # Add/remove any other columns to match the list in your corrected train.py
]

# --- Define Sample Data ---
# Create a list of dictionaries. Each dictionary represents one sample.
# Keys MUST exactly match the 'intended_feature_columns' list above.
# Provide plausible values for each feature. Use None for missing values if needed (imputer should handle them).
# **>>> DEFINE YOUR SAMPLE DATA HERE <<<**
sample_data = [
    # Sample 1: Represents a user potentially struggling
    {
        'final_phq9_score': 16,
        'final_gad7_score': 12,
        'stress_level': 8,
        'social_interactions': 2, # Low social interactions
        'emotion_volatility_estimate': 0.6, # High volatility
        'avg_sentiment_last_5': -0.4, # Negative sentiment
        'sleep_quality': 'Poor',
        'activity_level': 'Low',
        'final_emotion': 'Sad',
        'final_text_sentiment': 'Negative'
    },
    # Sample 2: Represents a user potentially coping/thriving
    {
        'final_phq9_score': 3,
        'final_gad7_score': 2,
        'stress_level': 2,
        'social_interactions': 10, # High social interactions
        'emotion_volatility_estimate': 0.1, # Low volatility
        'avg_sentiment_last_5': 0.7, # Positive sentiment
        'sleep_quality': 'Good',
        'activity_level': 'High',
        'final_emotion': 'Happy',
        'final_text_sentiment': 'Positive'
    },
    # Sample 3: Represents a user potentially vulnerable (mixed signals)
    {
        'final_phq9_score': 9,
        'final_gad7_score': 7,
        'stress_level': 5,
        'social_interactions': 5,
        'emotion_volatility_estimate': 0.3,
        'avg_sentiment_last_5': 0.1, # Slightly positive/neutral sentiment
        'sleep_quality': 'Fair',
        'activity_level': 'Medium',
        'final_emotion': 'Neutral', # Neutral emotion
        'final_text_sentiment': 'Negative' # But negative text
    },
    # Add more sample dictionaries here if needed
    # Sample 4: Example with missing values (if your pipeline's imputers are set up)
    # {
    #     'final_phq9_score': 10,
    #     'final_gad7_score': 8,
    #     'stress_level': None, # Missing stress level
    #     'social_interactions': 4,
    #     'emotion_volatility_estimate': 0.4,
    #     'avg_sentiment_last_5': -0.1,
    #     'sleep_quality': 'Fair',
    #     'activity_level': None, # Missing activity level
    #     'final_emotion': 'Neutral',
    #     'final_text_sentiment': 'Neutral'
    # },
]
# --- End Sample Data Definition ---


print("--- Preparing Sample Data ---")
try:
    # Convert the list of dictionaries into a Pandas DataFrame
    # Ensure the columns are in the order expected by the pipeline (matching intended_feature_columns)
    X_samples = pd.DataFrame(sample_data, columns=intended_feature_columns)
    print(f"Created DataFrame with {len(X_samples)} sample(s).")
    print("Sample Data Features:")
    print(X_samples.to_string())

except Exception as e:
    print(f"Error creating DataFrame from sample data: {e}")
    print("Please check if all keys in sample_data dictionaries match intended_feature_columns.")
    sys.exit(1)

# --- Load the Saved Pipeline ---
print(f"\n--- Loading Trained Pipeline from {SAVED_PIPELINE_FILENAME} ---")
try:
    loaded_pipeline = joblib.load(SAVED_PIPELINE_FILENAME)
    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print(f"Error: The pipeline file '{SAVED_PIPELINE_FILENAME}' was not found.")
    print("Make sure you have run the corrected train.py script successfully to create it.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading the pipeline: {e}")
    sys.exit(1)

# --- Make Predictions on Samples ---
print("\n--- Predicting Mental Health Status for Samples ---")
try:
    # Use the loaded pipeline to predict
    predictions = loaded_pipeline.predict(X_samples)

    # Display predictions alongside sample identifiers (index)
    print("\nPredictions:")
    for i, prediction in enumerate(predictions):
        print(f"  Sample {i+1}: Predicted Status = {prediction}")

    # Optional: Display probabilities if the model supports predict_proba (KNN does)
    if hasattr(loaded_pipeline, "predict_proba"):
        print("\nPrediction Probabilities (if available):")
        try:
            probabilities = loaded_pipeline.predict_proba(X_samples)
            # Get class labels from the pipeline
            class_labels = loaded_pipeline.classes_
            prob_df = pd.DataFrame(probabilities, columns=class_labels)
            prob_df.index = [f"Sample {i+1}" for i in range(len(prob_df))]
            print(prob_df)
        except Exception as proba_e:
            print(f"  Could not get probabilities: {proba_e}")


    print("\n--- Prediction Complete ---")

except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging

# END OF FULL CODE: predict_samples.py