"""
Model architecture definitions for Waste / Garbage Classification.
Implements:
1. Custom CNN from scratch (4 Conv blocks with BatchNorm + ReLU + MaxPool + Dropout).
2. Transfer Learning models (ResNet50V2, MobileNetV2, and DenseNet121).
3. Fine-tuning utilities for unfreezing top layers safely.
4. Model compilation helper with Adam optimizer.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    name: str = "custom_cnn",
) -> tf.keras.Model:
    """Builds a 4-block Custom Convolutional Neural Network from scratch.

    Architecture:
        - 4 Convolutional Blocks:
            Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.2)
            Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.25)
            Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.3)
            Block 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout(0.3)
        - Classification Head:
            GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.4)
            -> Dense(num_classes, softmax, dtype='float32')

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of target categories.
        name: Name of the model.

    Returns:
        A tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="dropout1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="dropout3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.30, name="dropout4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, kernel_initializer="he_normal", name="dense_feature")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.40, name="dropout_dense")(x)

    # Output layer explicitly float32 for mixed precision stability
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_transfer_learning_model(
    base_name: str = "ResNet50V2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    fine_tune: bool = False,
    name: Optional[str] = None,
) -> tf.keras.Model:
    """Builds a Transfer Learning model using pre-trained backbones (ResNet50V2, MobileNetV2, or DenseNet121).

    Args:
        base_name: One of 'ResNet50V2', 'MobileNetV2', or 'DenseNet121'.
        input_shape: Input image shape, typically (224, 224, 3).
        num_classes: Number of target categories.
        fine_tune: Whether base model weights should be trainable initially.
        name: Optional custom model name.

    Returns:
        Constructed tf.keras.Model.
    """
    model_name = name or f"transfer_{base_name.lower()}"
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Instantiate pre-trained backbone
    if base_name == "ResNet50V2":
        base_model = tf.keras.applications.ResNet50V2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif base_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif base_name == "DenseNet121":
        base_model = tf.keras.applications.DenseNet121(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported base_name: {base_name}. Supported: ResNet50V2, MobileNetV2, DenseNet121")

    base_model.trainable = fine_tune

    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_gap")(x)
    x = layers.Dropout(0.35, name="dropout_head")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def get_base_model(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    """Finds the underlying pre-trained base model inside a Transfer Learning model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreezes the last `num_layers` of the pre-trained base model for fine-tuning.

    BatchNormalization layers are frozen to maintain stable pre-trained feature statistics.

    Args:
        model: Transfer learning model containing the pre-trained base.
        num_layers: Number of top layers in the base model to unfreeze.
    """
    base_model = get_base_model(model)
    if base_model is None:
        target_layers = model.layers
    else:
        base_model.trainable = True
        target_layers = base_model.layers

    total_layers = len(target_layers)
    freeze_until = max(0, total_layers - num_layers)

    for i, layer in enumerate(target_layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for layer in target_layers if layer.trainable)
    print(
        f"[Model] Unfroze top {num_layers} layers of base model. "
        f"Trainable layers in base: {trainable_count}/{total_layers}"
    )


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    loss: str = "sparse_categorical_crossentropy",
) -> tf.keras.Model:
    """Compiles a Keras model with Adam optimizer and accuracy metrics.

    Args:
        model: The tf.keras.Model to compile.
        learning_rate: Learning rate for Adam optimizer.
        loss: Loss function string.

    Returns:
        The compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model
