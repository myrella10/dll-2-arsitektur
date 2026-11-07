"""
Central configuration module for Waste / Garbage Classification experiment.
Defines hyperparameters, dataset paths, class labels, and reproducibility settings.
"""

import os
import random
import numpy as np
import tensorflow as tf

# Reproducibility Seed
SEED: int = 42

# Image Dimensions
IMG_SIZE_CNN: tuple[int, int] = (150, 150)
IMG_SIZE_TL: tuple[int, int] = (224, 224)

# Training Hyperparameters
BATCH_SIZE: int = 32
EPOCHS: int = 30
LEARNING_RATE_CNN: float = 1e-3
LEARNING_RATE_TL_BASE: float = 1e-3
LEARNING_RATE_TL_FINETUNE: float = 1e-4

# Dataset Metadata (Garbage Classification: 6 Classes)
DATASET_KAGGLE_ID: str = "asdasdasasdas/garbage-classification"
NUM_CLASSES: int = 6
CLASS_NAMES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Directory Paths
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models")

# Mixed Precision Flag for GPU T4 Acceleration
ENABLE_MIXED_PRECISION: bool = True


def set_seed(seed: int = SEED) -> None:
    """Sets random seeds across random, numpy, and tensorflow for full reproducibility.

    Args:
        seed: The integer seed value to enforce.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)


def setup_directories() -> None:
    """Ensures that outputs and models directories exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def configure_mixed_precision(enable: bool = ENABLE_MIXED_PRECISION) -> str:
    """Configures TensorFlow mixed precision policy for faster training on GPU (T4).

    Args:
        enable: Whether to enable mixed_float16.

    Returns:
        The active mixed precision policy name.
    """
    if enable:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        return "mixed_float16"
    else:
        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)
        return "float32"
