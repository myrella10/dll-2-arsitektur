"""
Data loading and preprocessing pipeline for Waste / Garbage Classification.
Provides KaggleHub dataset download, dynamic directory resolution,
stratified train/val/test splitting, and optimized tf.data pipeline with augmentation.
"""

import os
from typing import Optional, Callable, Tuple, List, Union
import tensorflow as tf

from src.config import SEED, BATCH_SIZE, CLASS_NAMES, DATASET_KAGGLE_ID


def download_dataset_kagglehub(kaggle_id: str = DATASET_KAGGLE_ID) -> str:
    """Downloads the Waste / Garbage Classification dataset using kagglehub.

    Args:
        kaggle_id: Kaggle dataset identifier string.

    Returns:
        The local path string where kagglehub downloaded the dataset.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download(kaggle_id)
        print(f"[DataLoader] Successfully downloaded dataset via kagglehub: {path}")
        return str(path)
    except Exception as exc:
        print(f"[DataLoader] kagglehub download failed ({exc}). Falling back to local search.")
        candidate_paths = [
            "/content/dataset",
            "/content/garbage-classification",
            "dataset",
            "datasets",
            "."
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate) and any(
                os.path.isdir(os.path.join(candidate, c)) for c in CLASS_NAMES
            ):
                print(f"[DataLoader] Found existing dataset directory: {candidate}")
                return os.path.abspath(candidate)
        raise RuntimeError(
            f"Could not download dataset via kagglehub and no local fallback directory found: {exc}"
        )


def find_data_directory(dataset_path: str) -> str:
    """Finds the root directory containing class subfolders (cardboard, glass, etc.).

    Args:
        dataset_path: Root folder returned by kagglehub.

    Returns:
        Absolute path to the directory containing category subfolders.
    """
    # Direct check
    if all(os.path.isdir(os.path.join(dataset_path, c)) for c in CLASS_NAMES):
        return dataset_path

    # Recursive search
    for root, dirs, _ in os.walk(dataset_path):
        matches = [c for c in CLASS_NAMES if c in dirs]
        if len(matches) >= 4:  # Found the directory containing class folders
            return root

    raise FileNotFoundError(
        f"Could not find category subfolders ({CLASS_NAMES}) inside '{dataset_path}'"
    )


def get_class_names(data_dir: str) -> List[str]:
    """Retrieves sorted subfolder names representing class labels.

    Args:
        data_dir: Path to directory containing class subfolders.

    Returns:
        Alphabetically sorted list of class names.
    """
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]
    subdirs.sort()
    return subdirs


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """Builds a Keras Sequential layer for data augmentation.

    Returns:
        tf.keras.Sequential model with flip, rotation, zoom, and contrast layers.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(0.15, seed=SEED),
            tf.keras.layers.RandomZoom(0.15, seed=SEED),
            tf.keras.layers.RandomContrast(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )


def build_datasets(
    data_dir: str,
    img_size: Tuple[int, int] = (150, 150),
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
    preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Builds clean, stratified tf.data pipelines for Train (70%), Val (15%), and Test (15%).

    Args:
        data_dir: Path to the directory containing class subfolders.
        img_size: Tuple of (height, width) for image resizing.
        batch_size: Mini-batch size.
        use_augmentation: Whether to apply data augmentation to training data.
        preprocess_fn: Optional normalization/scaling function.
        train_split: Fraction for training data.
        val_split: Fraction for validation data.
        test_split: Fraction for testing data.
        seed: Random seed for reproducible splitting.

    Returns:
        A tuple of (train_ds, val_ds, test_ds).
    """
    # Total holdout fraction for val + test
    holdout_fraction = val_split + test_split

    # 1. Load Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=holdout_fraction,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )

    # 2. Load Holdout (Val + Test) dataset
    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=holdout_fraction,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    # Split holdout evenly into val and test
    holdout_cardinality = tf.data.experimental.cardinality(holdout_ds).numpy()
    val_batches = holdout_cardinality // 2

    val_ds = holdout_ds.take(val_batches)
    test_ds = holdout_ds.skip(val_batches)

    augmentation_model = get_data_augmentation_layer() if use_augmentation else None

    def prepare_pipeline(dataset: tf.data.Dataset, is_training: bool = False) -> tf.data.Dataset:
        if is_training and augmentation_model is not None:
            dataset = dataset.map(
                lambda x, y: (augmentation_model(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if preprocess_fn is not None:
            dataset = dataset.map(
                lambda x, y: (preprocess_fn(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if is_training:
            dataset = dataset.cache().shuffle(buffer_size=1000, seed=seed).prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    train_ds = prepare_pipeline(train_ds, is_training=True)
    val_ds = prepare_pipeline(val_ds, is_training=False)
    test_ds = prepare_pipeline(test_ds, is_training=False)

    return train_ds, val_ds, test_ds
