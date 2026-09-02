# train_emotion_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# Configuration
# ==============================================================================
DATASET_PATH = 'C:/Users/mrahu/Desktop/js-basics/dataset' # <<< IMPORTANT: SET THIS PATH!
MODEL_SAVE_PATH = 'emotion_model.h5'        # Where to save the best model
HISTORY_PLOT_PATH = 'training_history.png' # Where to save the training plot

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64  # Adjust based on your GPU memory (e.g., 32, 64, 128)
EPOCHS = 100     # Number of training cycles (can increase/decrease based on results)
NUM_CLASSES = 7  # For FER-2013 (angry, disgust, fear, happy, neutral, sad, surprise)
COLOR_MODE = 'grayscale' # 'grayscale' for FER-2013, 'rgb' if using color images
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1) if COLOR_MODE == 'grayscale' else (IMG_HEIGHT, IMG_WIDTH, 3)

# ==============================================================================
# Data Preparation
# ==============================================================================
logging.info("Setting up data generators...")

train_dir = os.path.join(DATASET_PATH, 'train')
test_dir = os.path.join(DATASET_PATH, 'test') # Use your validation or test directory name

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    logging.error(f"Dataset directories not found! Expected 'train' and 'test' folders inside: {DATASET_PATH}")
    exit() # Stop if dataset path is wrong

# --- Data Augmentation and Normalization for Training ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Normalization for Validation/Testing ---
test_datagen = ImageDataGenerator(rescale=1./255)

# --- Create Data Generators ---
try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_mode='categorical', # For multi-class classification
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except FileNotFoundError:
     logging.error(f"Error finding dataset at {DATASET_PATH}. Check path and folder structure.")
     exit()

# Log the class indices - IMPORTANT for matching labels later!
logging.info(f"Training Class Indices: {train_generator.class_indices}")
logging.info(f"Validation Class Indices: {validation_generator.class_indices}")
# Make sure the EMOTION_LABELS list in your final app matches this order.

# Ensure generators found images
if train_generator.samples == 0 or validation_generator.samples == 0:
    logging.error("No images found by the data generators. Check dataset path and subfolder structure.")
    exit()
else:
    logging.info(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    logging.info(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")

# ==============================================================================
# Model Definition (CNN)
# ==============================================================================
logging.info("Building CNN model...")

def build_emotion_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Builds a CNN model for emotion classification."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Softmax for multi-class probability
    ])
    return model

model = build_emotion_model()

# ==============================================================================
# Model Compilation
# ==============================================================================
logging.info("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

logging.info("Model Summary:")
model.summary() # Print model architecture

# ==============================================================================
# Callbacks
# ==============================================================================
logging.info("Setting up callbacks...")

# Save the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Stop training if validation loss doesn't improve for a while
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15, # Number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity.
    verbose=1
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# ==============================================================================
# Training
# ==============================================================================
logging.info("Starting model training...")

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

if steps_per_epoch == 0 or validation_steps == 0:
    logging.warning(f"steps_per_epoch ({steps_per_epoch}) or validation_steps ({validation_steps}) is zero. "
                    f"Check batch size ({BATCH_SIZE}) relative to dataset size "
                    f"({train_generator.samples} train, {validation_generator.samples} val). "
                    f"Consider reducing batch size.")
    # Avoid division by zero; adjust if necessary or exit if dataset too small
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

logging.info("Training finished.")

# ==============================================================================
# Plotting History (Optional)
# ==============================================================================
try:
    logging.info("Plotting training history...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_run = range(len(acc)) # Use actual number of epochs run

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_run, acc, label='Training Accuracy')
    plt.plot(epochs_run, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_run, loss, label='Training Loss')
    plt.plot(epochs_run, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    logging.info(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    # plt.show() # Uncomment to display plot directly if running interactively

except Exception as plot_err:
    logging.warning(f"Could not plot training history: {plot_err}")


logging.info(f"Best model weights were saved to {MODEL_SAVE_PATH} during training (based on validation accuracy).")
logging.info("Training script completed.")