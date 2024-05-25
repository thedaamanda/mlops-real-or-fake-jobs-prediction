"""
Author: Muhammad Theda Amanda
Usage: Transform the data
"""
import tensorflow as tf

LABEL_KEY = "fraudulent"
FEATURE_KEY = "text"


def transformed_name(key):
    """
    This function returns the name of the transformed feature

    Args:
        key: the name of the feature

    Returns:
        the name of the transformed feature
    """
    return f"{key}_xf"


def preprocessing_fn(inputs):
    """
    This function preprocesses the data before training the model

    Args:
        inputs: A dictionary of input data with the keys

    Returns:
        the transformed data
    """

    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
