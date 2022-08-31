from unittest import TestCase

import tensorflow as tf
import numpy as np


class TestFineTuning(TestCase):
    def test_trainable_param_native_training(self):
        layer = tf.keras.layers.Dense(1, input_shape=(1,))
        model = tf.keras.Sequential([layer])

        data_x = [[1]]
        data_y = [0]

        model.compile(optimizer="adam", loss="mse")
        layer.trainable = False
        initial_weights: list[np.ndarray] = layer.get_weights()
        model.fit(data_x, data_y, epochs=1, batch_size=1, verbose=0)
        new_weights: list[np.ndarray] = layer.get_weights()

        for initial, new in zip(initial_weights, new_weights):
            self.assertFalse(np.allclose(initial, new))

        # model should be recompiled after layer.trainable changed
        model.compile(optimizer="adam", loss="mse")
        initial_weights: list[np.ndarray] = layer.get_weights()
        model.fit(data_x, data_y, epochs=1, batch_size=1, verbose=0)
        new_weights: list[np.ndarray] = layer.get_weights()

        for initial, new in zip(initial_weights, new_weights):
            self.assertTrue(np.allclose(initial, new))

    def test_trainable_param_custom_training(self):
        layer = tf.keras.layers.Dense(1, input_shape=(1,))
        model = tf.keras.Sequential([layer])

        layer.trainable = False
        initial_weights: list[np.ndarray] = layer.get_weights()

        data_x = np.array([[1]])
        data_y = np.array([0])
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

        with tf.GradientTape() as tape:
            logits = model(data_x, training=True)
            loss_value = loss_fn(data_y, logits)

        # model.weights are used instead of model.trainable_weights!!!
        grads = tape.gradient(loss_value, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        new_weights: list[np.ndarray] = layer.get_weights()

        for initial, new in zip(initial_weights, new_weights):
            # trainable param does not prevent variables update, it's just could be used during training
            self.assertFalse(np.allclose(initial, new))
