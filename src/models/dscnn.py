"""
DS-CNN student model architecture.
Depthwise Separable CNN designed for MCU inference (~60KB INT8).

Reference: Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers"
"""

import tensorflow as tf

NUM_CLASSES = 6


def ds_conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """Depthwise separable conv block: depthwise → pointwise → BN → ReLU."""
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_dscnn(input_shape, num_classes=NUM_CLASSES):
    """
    DS-CNN for log-mel spectrograms.
    Target: <100KB INT8, <100ms on ESP32-S3.

    input_shape: (64, T, 1) — log-mel with channel dim
    """
    inputs = tf.keras.Input(shape=input_shape, name="log_mel")

    # Initial standard conv
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # DS-CNN blocks
    x = ds_conv_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ds_conv_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ds_conv_block(x, 128)

    # Global average pooling — no flatten, keeps model small
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="dscnn_student")
    return model


def count_params(model):
    total = sum(tf.size(w).numpy() for w in model.weights)
    print(f"Parameters: {total:,} ({total * 4 / 1024:.1f} KB float32 | ~{total / 1024:.1f} KB INT8)")
    return total


if __name__ == "__main__":
    # Quick test
    import numpy as np
    model = build_dscnn((64, 63, 1))
    model.summary()
    count_params(model)
    x = np.random.rand(1, 64, 63, 1).astype(np.float32)
    print("Output shape:", model(x).shape)
