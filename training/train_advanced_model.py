#!/usr/bin/env python3
"""
Advanced Emotion Detection Training Script
Using the custom Performance Model architecture from fer-model-main.

This script will:
1. Load fer2013.csv (48x48 images).
2. Use the custom 'get_performance_model' architecture.
3. Map labels to match the App:
   0:Neutral, 1:Happy, 2:Surprise, 3:Sad, 4:Angry, 5:Disgust, 6:Fear
4. Train with data augmentation and fine-tuning.
5. Save to TFLite format.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import the real model definitions from the local file
try:
    from models import get_performance_model
    print("Loaded custom model architecture from local models.py")
except ImportError:
    print("Error: Could not find models.py in the training folder.")
    sys.exit(1)

# FER2013 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# APP classes: 0=Neutral, 1=Happy, 2=Surprise, 3=Sad, 4=Angry, 5=Disgust, 6=Fear
# Mapping array: index is FER id, value is APP id
FER_TO_APP = np.array([4, 5, 6, 1, 3, 2, 0], dtype=np.int64)

IMG_SIZE = 48
NUM_CLASSES = 7

def load_fer_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("CSV needs 'emotion' and 'pixels' columns (standard fer2013.csv).")

    usage = df["Usage"].str.strip() if "Usage" in df.columns else None
    if usage is not None:
        df = df[usage == "Training"].reset_index(drop=True)

    n = len(df)
    x = np.zeros((n, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    y_app = np.zeros((n,), dtype=np.int64)

    pixels = df["pixels"].astype(str).str.split(expand=False).values
    emotions = df["emotion"].astype(np.int64).values

    for i in range(n):
        arr = np.array(pixels[i], dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)
        x[i] = arr
        e = int(emotions[i])
        if e < 0 or e > 6:
            continue
        y_app[i] = FER_TO_APP[e]

    # Normalize to 0-1
    x = x / 255.0
    return x, y_app

def build_advanced_model() -> keras.Model:
    """Uses the custom Performance Model from models.py"""
    core = get_performance_model(
        leaky_relu_slope=0.1,
        dropout_rate=0.2,
        regularization_rate=0.01,
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        n_classes=NUM_CLASSES,
        logits=True
    )
    
    # Wrap with data augmentation for training
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    outputs = core(x)
    
    # Return the training model and the core model (for export)
    train_model = keras.Model(inputs, outputs, name="emotion_train_advanced")
    return train_model, core

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "fer2013.csv"),
        help="Path to fer2013.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "output", "advanced_emotion_model.tflite"),
        help="Output .tflite path",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"Missing dataset file: {args.data}")
        return 1

    print("Loading Dataset...")
    x, y = load_fer_csv(args.data)
    y_cat = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

    n = len(x)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.9 * n)
    tr, va = idx[:split], idx[split:]

    print("Building Custom Advanced Model...")
    train_model, core_model = build_advanced_model()
    
    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    early = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    print("Training Model...")
    train_model.fit(
        x[tr], y_cat[tr],
        validation_data=(x[va], y_cat[va]),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=[early, reduce],
        verbose=1,
    )

    # Convert to TFLite (use the core model without augmentation)
    print(f"Converting and saving to {args.out}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(core_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    
    with open(args.out, "wb") as f:
        f.write(tflite_bytes)

    print("Done! Advanced custom model is ready.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
