# START OF FULL CODE: train.py (Corrected with Explicit Feature Selection)

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier # Correct classifier import
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import joblib # Import joblib for saving the model

# --- Configuration ---
FILE_PATH = 'oversampled_mental_health_dataset.csv'
# --- Update this to the exact name of the target column in your CSV ---
TARGET_COLUMN = 'mental_health_status'
TEST_SIZE = 0.25 # Fraction of data to use for testing
RANDOM_STATE = 42 # For reproducibility
CV_FOLDS = 5 # Number of folds for cross-validation
OUTPUT_PIPELINE_FILENAME = 'sklearn_knn_mental_health_pipeline.joblib'

# --- Load Data ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded data from {FILE_PATH}")
    print(f"Dataset shape: {df.shape}")

    # --- Print Column Names to Help Debugging ---
    print("\nColumns found in the CSV file:")
    print(list(df.columns)) # Print all column names

    # --- Check if TARGET_COLUMN exists ---
    if TARGET_COLUMN not in df.columns:
        print(f"\n--- ERROR ---")
        print(f"The specified TARGET_COLUMN ('{TARGET_COLUMN}') was not found in the CSV.")
        print(f"Please check the 'Columns found' list above and update the TARGET_COLUMN variable in the script.")
        sys.exit(1) # Exit the script

    # --- Basic Data Cleaning ---
    # Drop rows with missing target variable BEFORE feature selection
    initial_rows = df.shape[0]
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"\nShape after dropping rows with missing target: {df.shape} ({initial_rows - df.shape[0]} rows removed)")

    # --- Define Features (X) and Target (y) ---

    # <<< --- CORRECTED SECTION: Explicitly Define Features --- >>>
    # Define the columns you ACTUALLY want to use as features for the model
    # These should match the ones you collect/calculate in hello.py for prediction
    # **>>> ADJUST THIS LIST BASED ON YOUR ACTUAL INTENDED FEATURES <<<**
    intended_feature_columns = [
        'final_phq9_score', 'final_gad7_score', 'stress_level',
        'social_interactions', 'emotion_volatility_estimate',
        'avg_sentiment_last_5', 'sleep_quality', 'activity_level',
        'final_emotion', 'final_text_sentiment'
        # Add/remove any other columns from your CSV that are meant to be features
        # DO NOT include 'username', 'phq9_severity', 'gad7_severity', etc. unless they are actual features
    ]
    print(f"\nIntended features for training: {intended_feature_columns}")

    # Ensure all intended features exist in the dataframe
    missing_intended_features = [col for col in intended_feature_columns if col not in df.columns]
    if missing_intended_features:
        print(f"\n--- ERROR ---")
        print(f"The following intended feature columns are missing from the CSV: {missing_intended_features}")
        print(f"Please check the 'intended_feature_columns' list in the script or the CSV file.")
        sys.exit(1)

    # Select only the intended features for X
    X = df[intended_feature_columns].copy() # Use .copy() to avoid warnings
    y = df[TARGET_COLUMN]
    # <<< --- END OF CORRECTED SECTION --- >>>

    print(f"\nShape of features (X) for training: {X.shape}")
    print(f"\nTarget variable distribution:\n{y.value_counts(normalize=True)}")

    # --- Identify Feature Types (using the columns from the filtered X) ---
    # This will now correctly identify types only from the intended features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"\nNumerical features used ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features used ({len(categorical_features)}): {categorical_features}")

    # --- Preprocessing Steps ---
    # Impute missing numerical values with the median and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Impute missing categorical values with the most frequent value and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # handle_unknown='ignore' is important
    ])

    # Create a preprocessor object using ColumnTransformer
    # This now explicitly uses the identified numerical/categorical features from the INTENDED set
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not explicitly specified (safer)
    )

    # --- Define the KNN Model ---
    knn_model = KNeighborsClassifier() # Default n_neighbors=5

    # --- Create the Full Pipeline ---
    # The pipeline first preprocesses ONLY the specified features, then classifies
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', knn_model)
    ])

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, # Use the explicitly selected X
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # --- Train the KNN Pipeline ---
    print("\n--- Training K-Nearest Neighbors Model ---")
    start_time = time.time()
    # Fit the pipeline on the training data (X contains only intended features)
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training complete. Time taken: {training_time:.2f} seconds")

    # --- Evaluate the Model ---
    print("\n--- Evaluating KNN Model on Test Set ---")
    # Predict using the test set (X_test also contains only intended features)
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # Added zero_division=0
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    # Ensure labels for heatmap match the actual classes in y
    class_labels = sorted(y.unique())
    print(conf_matrix)


    # --- Display Confusion Matrix ---
    try:
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, # Use actual class labels
                    yticklabels=class_labels) # Use actual class labels
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - K-Nearest Neighbors')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except Exception as plot_e:
        print(f"\nWarning: Could not display confusion matrix plot. Error: {plot_e}")


    # --- Save the Trained Pipeline ---
    print(f"\n--- Saving the trained KNN pipeline to {OUTPUT_PIPELINE_FILENAME} ---")
    try:
        joblib.dump(pipeline, OUTPUT_PIPELINE_FILENAME)
        print(f"Pipeline successfully saved.")
    except Exception as e:
        print(f"Error saving pipeline: {e}")

except FileNotFoundError:
    print(f"Error: The file {FILE_PATH} was not found.")
except ModuleNotFoundError as e:
     if 'SimpleImputer' in str(e):
         print("Error: SimpleImputer not found. You might need scikit-learn version 0.20 or later.")
         print("Try installing or upgrading scikit-learn: pip install -U scikit-learn")
     elif 'joblib' in str(e):
          print("Error: joblib not found. You might need to install it.")
          print("Try installing joblib: pip install joblib")
     else:
        print(f"An unexpected error occurred: {e}")
except KeyError as e:
    print(f"Error: Column '{e}' not found during processing. Please check feature names and TARGET_COLUMN.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging

# END OF FULL CODE: train.py (Corrected with Explicit Feature Selection)