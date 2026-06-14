# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE    = 48
NUM_CLASSES = 7
EMOTIONS    = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


def build_emotion_cnn(input_shape=(48, 48, 1), num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    Deep CNN for facial emotion recognition.
    Architecture: 4 Conv blocks + FC head with heavy regularisation.
    """
    model = Sequential([

        # ── Block 1: Edge detection ──────────────────────────────
        Conv2D(64, (3,3), padding='same', input_shape=input_shape,
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        # ── Block 2: Low-level features ──────────────────────────
        Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        # ── Block 3: Mid-level features ──────────────────────────
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.35),

        # ── Block 4: High-level features ─────────────────────────
        Conv2D(512, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Dropout(0.5),

        # ── Fully Connected Head ──────────────────────────────────
        Dense(512, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(256, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ], name='EmotionCNN')

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile with Adam + categorical crossentropy."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(checkpoint_dir: str = 'models/checkpoints') -> list:
    """
    Training callbacks:
    - ModelCheckpoint : saves best weights automatically
    - EarlyStopping   : stops if val_loss stops improving
    - ReduceLROnPlateau: halves LR when stuck
    - CSVLogger       : logs every epoch to CSV
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv'),
            append=False
        )
    ]
    return callbacks


def print_model_summary(model: tf.keras.Model):
    """Print summary + total parameter count."""
    model.summary()
    total    = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")


if __name__ == '__main__':
    print("🧠 Building EmotionCNN...")
    model = build_emotion_cnn()
    model = compile_model(model)
    print_model_summary(model)
    print("\n✅ Model architecture verified.")

#python src/model.py