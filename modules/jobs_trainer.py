"""
Author: Muhammad Theda Amanda
Usage: Train the model
"""
import os
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from jobs_transform import transformed_name, LABEL_KEY, FEATURE_KEY
from jobs_tuner import (
    early_stopping_callback,
    vectorized_dataset,
    vectorized_layer,
    input_fn,
)

NUM_EPOCHS = 3


def model_builder(vectorizer_layer, hyperparameters):
    """
    This function builds the model

    Args:
        vectorizer_layer: the text vectorizer layer
        hyperparameters: the hyperparameters

    Returns:
        the model
    """
    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorizer_layer(inputs)
    x = layers.Embedding(
        input_dim=5000,
        output_dim=hyperparameters["embed_dims"])(x)
    x = layers.Bidirectional(layers.LSTM(hyperparameters["lstm_units"]))(x)

    for _ in range(hyperparameters["num_hidden_layers"]):
        x = layers.Dense(
            hyperparameters["dense_units"],
            activation=tf.nn.relu)(x)
        x = layers.Dropout(hyperparameters["dropout_rate"])(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparameters["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy()],
    )

    model.summary()

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    This function returns the serving function

    Args:
        model: the model
        tf_transform_output: the transformation output

    Returns:
        the serving function
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """
    This function trains the model

    Args:
        fn_args: the function arguments
    """
    hyperparameters = fn_args.hyperparameters["values"]

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    callbacks = [
        tensorboard_callback,
        early_stopping_callback,
        model_checkpoint_callback
    ]

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        NUM_EPOCHS)

    eval_set = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        NUM_EPOCHS)

    vectorizer_dataset = vectorized_dataset(train_set)

    vectorizer_layer = vectorized_layer()

    vectorizer_layer.adapt(vectorizer_dataset)

    model = model_builder(vectorizer_layer, hyperparameters)

    model.fit(
        x=train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_set,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=NUM_EPOCHS,
        verbose=1,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples",
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures
    )
