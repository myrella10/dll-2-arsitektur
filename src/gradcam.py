"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation from scratch.
Enables visual interpretability by highlighting image regions most influential
in model classification decisions. Supports Custom CNN and Transfer Learning models.
"""

import os
from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from src.config import OUTPUT_DIR, CLASS_NAMES


def find_last_conv_layer_name(model: tf.keras.Model) -> Tuple[Optional[tf.keras.Model], str]:
    """Automatically finds the target model (base or main) and the name of its last conv layer.

    Args:
        model: tf.keras.Model to inspect.

    Returns:
        A tuple of (target_model_reference, layer_name).
    """
    # Check if model has a nested base model (Transfer Learning)
    submodel = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            submodel = layer
            break

    search_target = submodel if submodel is not None else model

    # Search backwards for a 2D Conv or corresponding activation layer
    for layer in reversed(search_target.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return search_target, layer.name
        # Some architectures end conv backbone with an activation
        if "conv" in layer.name.lower() or "relu" in layer.name.lower() or "activation" in layer.name.lower():
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return search_target, layer.name

    raise ValueError(f"Could not automatically detect a 4D convolutional layer in '{model.name}'.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Computes Grad-CAM heatmap for a given input image array.

    Args:
        img_array: Batch tensor of shape (1, H, W, C).
        model: Trained tf.keras.Model.
        last_conv_layer_name: Name of the last convolutional layer.
        pred_index: Target class index to explain. If None, uses top prediction.

    Returns:
        2D numpy array representing normalized class activation map (0.0 to 1.0).
    """
    # Identify target layer and whether it's inside a nested submodel
    submodel = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            submodel = layer
            break

    if last_conv_layer_name is None:
        target_model, detected_layer_name = find_last_conv_layer_name(model)
        last_conv_layer_name = detected_layer_name
    else:
        target_model = submodel if (submodel and any(l.name == last_conv_layer_name for l in submodel.layers)) else model

    # Case A: Custom CNN / Flat model
    if submodel is None or target_model is model:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

    # Case B: Nested Transfer Learning Model
    else:
        # Create intermediate feature extractor from submodel
        feature_extractor = tf.keras.models.Model(
            inputs=[submodel.inputs],
            outputs=[submodel.get_layer(last_conv_layer_name).output, submodel.output],
        )

        # Collect classifier head layers that follow submodel
        head_layers = []
        found_submodel = False
        for layer in model.layers:
            if found_submodel:
                head_layers.append(layer)
            if layer is submodel:
                found_submodel = True

        with tf.GradientTape() as tape:
            conv_outputs, base_output = feature_extractor(img_array)
            tape.watch(conv_outputs)
            tape.watch(base_output)

            # Pass through classifier head
            x = base_output
            for layer in head_layers:
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by "how important this channel is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain features that positively contribute to the target class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def superimpose_gradcam(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Superimposes Grad-CAM heatmap over the original input image.

    Args:
        img: Original image array with values in [0, 255] or [0, 1].
        heatmap: 2D numpy array heatmap with values in [0, 1].
        alpha: Blending weight for the heatmap overlay.
        colormap_name: Matplotlib colormap name.

    Returns:
        Blended RGB image as uint8 numpy array [0, 255].
    """
    # Normalize original image to uint8 [0, 255]
    if img.max() <= 1.0:
        img_uint8 = np.uint8(255 * img)
    else:
        img_uint8 = np.uint8(img)

    # Resize heatmap to match original image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (img_uint8.shape[0], img_uint8.shape[1])
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # Colormap transformation
    colormap = plt.colormaps.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap_resized)
    colored_heatmap = np.uint8(255 * colored_heatmap[:, :, :3])

    # Blend original image and colored heatmap
    superimposed = np.uint8(colored_heatmap * alpha + img_uint8 * (1.0 - alpha))
    return superimposed


def visualize_gradcam_samples(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str] = CLASS_NAMES,
    last_conv_layer_name: Optional[str] = None,
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> str:
    """Visualizes Grad-CAM heatmaps for a batch of test samples in a clean grid.

    Args:
        model: Trained tf.keras.Model.
        dataset: Evaluation tf.data.Dataset (unbatched or batched).
        class_names: List of category names.
        last_conv_layer_name: Optional target convolutional layer name.
        num_samples: Number of samples to visualize (default 8: 2 rows x 4 cols).
        save_path: Destination path for the saved image.

    Returns:
        Path of the saved visualization plot.
    """
    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"{model.name}_gradcam_grid.png")

    # Collect sample images and labels
    samples_img = []
    samples_label = []
    for images, labels in dataset.take(num_samples):
        for i in range(len(images)):
            samples_img.append(images[i].numpy())
            samples_label.append(int(labels[i].numpy()))
            if len(samples_img) >= num_samples:
                break
        if len(samples_img) >= num_samples:
            break

    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        raw_img = samples_img[i]
        true_cls_idx = samples_label[i]
        true_name = class_names[true_cls_idx]

        # Prepare single image tensor for model input
        input_tensor = np.expand_dims(raw_img, axis=0)
        preds = model.predict(input_tensor, verbose=0)
        pred_cls_idx = int(np.argmax(preds[0]))
        pred_name = class_names[pred_cls_idx]
        confidence = preds[0][pred_cls_idx] * 100

        # Compute Grad-CAM
        heatmap = make_gradcam_heatmap(
            input_tensor, model, last_conv_layer_name=last_conv_layer_name, pred_index=pred_cls_idx
        )
        overlay = superimpose_gradcam(raw_img, heatmap, alpha=0.45)

        # Plot overlay
        axes[i].imshow(overlay)
        axes[i].axis("off")

        # Color title green if correct, red if incorrect
        color = "#2ca02c" if true_cls_idx == pred_cls_idx else "#d62728"
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name} ({confidence:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Grad-CAM Explainability Grid — {model.name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Grad-CAM] Saved explainability grid visualization to {save_path}")
    return save_path
