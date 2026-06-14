# src/train.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # silence oneDNN info logs

from model import build_emotion_cnn, compile_model, get_callbacks

# ── Hyperparameters ───────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 60        # EarlyStopping will cut this short
LR_INITIAL  = 1e-3
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)


def load_data():
    """Load preprocessed .npy arrays from data/processed/."""
    print("📦 Loading preprocessed data...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')
    X_test  = np.load('data/processed/X_test.npy')
    y_test  = np.load('data/processed/y_test.npy')
    print(f"  Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_augmentor() -> ImageDataGenerator:
    """
    Data augmentation applied ONLY to training images.
    Prevents overfitting by generating slightly modified variants.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights to handle the Disgust class imbalance.
    Converts one-hot y back to integers for sklearn.
    """
    from sklearn.utils.class_weight import compute_class_weight
    y_int = np.argmax(y_train, axis=1)
    classes = np.unique(y_int)
    weights = compute_class_weight('balanced', classes=classes, y=y_int)
    class_weight_dict = dict(zip(classes, weights))
    print("\n⚖️  Class weights (balancing imbalanced Disgust class):")
    emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    for idx, w in class_weight_dict.items():
        print(f"    {emotions[idx]:<10} → {w:.4f}")
    return class_weight_dict


def plot_training_history(history, save_dir='models'):
    """Plot accuracy & loss curves and save to models/."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Acc', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc',   linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  📈 Training curves saved → {path}")


def evaluate_model(model, X_test, y_test):
    """Full evaluation with confusion matrix and per-class report."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    print("\n🔍 Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotions))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions,
                linewidths=0.5)
    ax.set_title('Confusion Matrix — Test Set', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    path = 'models/confusion_matrix.png'
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  🗺️  Confusion matrix saved → {path}")

    return acc


def run_training():
    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Class weights
    class_weights = compute_class_weights(y_train)

    # 3. Build & compile model
    print("\n🧠 Building model...")
    model = build_emotion_cnn()
    model = compile_model(model, learning_rate=LR_INITIAL)
    model.summary()

    # 4. Data augmentation
    augmentor = build_augmentor()
    train_gen = augmentor.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    # 5. Train
    print(f"\n🚀 Starting training | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("   (EarlyStopping active — training will stop automatically)\n")

    steps_per_epoch = len(X_train) // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 6. Plot curves
    plot_training_history(history)

    # 7. Evaluate
    evaluate_model(model, X_test, y_test)

    # 8. Save final model
    os.makedirs('models/saved_model', exist_ok=True)
    save_path = 'models/saved_model/emotion_model.keras'
    model.save(save_path)
    print(f"\n💾 Final model saved → {save_path}")
    print("\n🎉 Training complete!")


if __name__ == '__main__':
    run_training()

    #cd src
#python train.py