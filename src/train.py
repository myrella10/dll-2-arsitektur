"""
Training utilities for Intel Image Classification models.
Provides automated callback configuration, timed training loops,
and CSV history export.
"""

import os
import time
from typing import List, Tuple, Union, Optional
import pandas as pd
import tensorflow as tf

from src.config import MODEL_DIR, OUTPUT_DIR


def get_callbacks(
    model_name: str,
    model_dir: str = MODEL_DIR,
    patience_es: int = 5,
    patience_lr: int = 3,
) -> List[tf.keras.callbacks.Callback]:
    """Generates standard training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

    Args:
        model_name: Base name for saved checkpoint files.
        model_dir: Directory where model checkpoints will be stored.
        patience_es: Patience for EarlyStopping.
        patience_lr: Patience for ReduceLROnPlateau.

    Returns:
        List of configured tf.keras callbacks.
    """
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"{model_name}_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 30,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    initial_epoch: int = 0,
) -> Tuple[tf.keras.callbacks.History, float]:
    """Fits the model on training data with validation monitoring and timing.

    Args:
        model: tf.keras.Model to train.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        epochs: Target number of epochs.
        callbacks: Optional list of training callbacks.
        initial_epoch: Epoch at which to start training (useful for fine-tuning resumption).

    Returns:
        A tuple of (history, elapsed_seconds).
    """
    print(f"\n[Training] Starting training for '{model.name}' (Target Epochs: {epochs})...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    elapsed_time = time.time() - start_time
    print(
        f"[Training] Completed training '{model.name}' in "
        f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)."
    )

    # Attach duration to history for convenient access
    history.training_time = elapsed_time
    return history, elapsed_time


def save_training_history(
    history: Union[tf.keras.callbacks.History, dict],
    model_name: str,
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Exports training metrics per epoch to a CSV file.

    Args:
        history: Keras History object or metrics dictionary.
        model_name: Identifier for the model.
        output_dir: Destination folder for the CSV file.

    Returns:
        Pandas DataFrame containing history metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_history.csv")

    raw_dict = history.history if hasattr(history, "history") else history
    df = pd.DataFrame(raw_dict)
    df.index.name = "epoch"
    df.to_csv(csv_path)

    print(f"[Training] Saved training history to {csv_path}")
    return df
