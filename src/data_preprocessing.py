# src/data_preprocessing.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE    = 48          # FER2013 images are 48×48 px
NUM_CLASSES = 7
RANDOM_SEED = 42

EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Map folder names → class indices
EMOTION_FOLDERS = {
    'angry':    0,
    'disgust':  1,
    'fear':     2,
    'happy':    3,
    'neutral':  4,
    'sad':      5,
    'surprise': 6
}


def load_images_from_folder(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk through emotion subfolders, load each image as grayscale,
    resize to IMG_SIZE x IMG_SIZE, and return (X, y) arrays.
    """
    images, labels = [], []
    total = 0

    print(f"\n📂 Loading images from: {data_dir}")
    print("-" * 50)

    for folder_name, label_idx in EMOTION_FOLDERS.items():
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"  ⚠️  Skipping missing folder: {folder_name}")
            continue

        count = 0
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('L')          # grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img, dtype=np.float32))
                labels.append(label_idx)
                count += 1
            except Exception as e:
                print(f"  ✗ Could not load {filename}: {e}")

        print(f"  ✓ {EMOTIONS[label_idx]:<10} → {count:>5} images")
        total += count

    print("-" * 50)
    print(f"  Total loaded: {total} images\n")
    return np.array(images), np.array(labels)


def preprocess(X: np.ndarray, y: np.ndarray):
    """
    Normalize pixel values to [0, 1] and reshape for CNN input.
    Returns X shaped (N, 48, 48, 1) and one-hot y shaped (N, 7).
    """
    from tensorflow.keras.utils import to_categorical

    # Normalize
    X = X / 255.0

    # Add channel dimension → (N, 48, 48, 1)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # One-hot encode labels
    y = to_categorical(y, num_classes=NUM_CLASSES)

    return X, y


def get_class_distribution(y_raw: np.ndarray) -> pd.DataFrame:
    """
    Returns a pandas DataFrame summarising class counts and percentages.
    y_raw must be the integer label array (before one-hot encoding).
    """
    counts = pd.Series(y_raw).value_counts().sort_index()
    df = pd.DataFrame({
        'Emotion': [EMOTIONS[i] for i in counts.index],
        'Count':   counts.values,
        'Percent': (counts.values / counts.values.sum() * 100).round(2)
    })
    return df


def plot_class_distribution(df: pd.DataFrame, split_name: str = 'Train'):
    """Bar chart of emotion class distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#E74C3C','#8E44AD','#3498DB','#F1C40F',
              '#95A5A6','#2ECC71','#E67E22']
    bars = ax.bar(df['Emotion'], df['Count'], color=colors, edgecolor='black', linewidth=0.7)

    for bar, pct in zip(bars, df['Percent']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                f'{pct}%', ha='center', va='bottom', fontsize=9)

    ax.set_title(f'FER2013 — {split_name} Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'data/processed/{split_name.lower()}_distribution.png', dpi=150)
    plt.show()
    print(f"  📊 Chart saved → data/processed/{split_name.lower()}_distribution.png")


def plot_sample_images(X_raw: np.ndarray, y_raw: np.ndarray, n: int = 5):
    """Display n sample images per emotion class."""
    fig, axes = plt.subplots(NUM_CLASSES, n, figsize=(n * 2, NUM_CLASSES * 2))
    fig.suptitle('Sample Images per Emotion', fontsize=14, fontweight='bold')

    for class_idx in range(NUM_CLASSES):
        class_images = X_raw[y_raw == class_idx]
        samples = class_images[:n]
        for j, img in enumerate(samples):
            axes[class_idx][j].imshow(img, cmap='gray')
            axes[class_idx][j].axis('off')
            if j == 0:
                axes[class_idx][j].set_ylabel(
                    EMOTIONS[class_idx], fontsize=9, rotation=0,
                    labelpad=45, va='center'
                )
    plt.tight_layout()
    plt.savefig('data/processed/sample_images.png', dpi=150)
    plt.show()
    print("  🖼️  Sample grid saved → data/processed/sample_images.png")


def run_preprocessing():
    """Full pipeline: load → inspect → preprocess → save .npy files."""

    RAW_TRAIN = os.path.join('data', 'raw', 'train')
    RAW_TEST  = os.path.join('data', 'raw', 'test')

    # ── 1. Load raw images ────────────────────────────────────────
    X_train_raw, y_train_raw = load_images_from_folder(RAW_TRAIN)
    X_test_raw,  y_test_raw  = load_images_from_folder(RAW_TEST)

    # ── 2. Class distribution report ──────────────────────────────
    print("\n📊 Training Set Distribution:")
    train_dist = get_class_distribution(y_train_raw)
    print(train_dist.to_string(index=False))
    plot_class_distribution(train_dist, 'Train')

    print("\n📊 Test Set Distribution:")
    test_dist = get_class_distribution(y_test_raw)
    print(test_dist.to_string(index=False))

    # ── 3. Sample image grid ──────────────────────────────────────
    plot_sample_images(X_train_raw, y_train_raw)

    # ── 4. Preprocess ─────────────────────────────────────────────
    print("\n⚙️  Preprocessing...")
    X_train, y_train = preprocess(X_train_raw, y_train_raw)

    # Split off a validation set (10%) from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=y_train          # keep class balance
    )

    X_test, y_test = preprocess(X_test_raw, y_test_raw)

    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}    |  y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

    # ── 5. Save processed arrays ──────────────────────────────────
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_val.npy',   X_val)
    np.save('data/processed/y_val.npy',   y_val)
    np.save('data/processed/X_test.npy',  X_test)
    np.save('data/processed/y_test.npy',  y_test)

    print("\n✅ Preprocessed arrays saved to data/processed/")
    print("   → X_train.npy, y_train.npy")
    print("   → X_val.npy,   y_val.npy")
    print("   → X_test.npy,  y_test.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    run_preprocessing()