from unittest import TestCase

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TestSequential(TestCase):
    def test_without_input_shape(self) -> None:
        layer = layers.Dense(1)
        model = keras.Sequential([layer])

        self.assertEqual(len(layer.weights), 0)
        with self.assertRaises(ValueError):
            weights: list[tf.Variable] = model.weights

    def test_with_input(self) -> None:
        layer = layers.Dense(1)
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            layer,
        ])

        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(model.weights), 2)

    def test_with_input_shape(self) -> None:
        layer = layers.Dense(1, input_shape=(4,))
        model = keras.Sequential([
            layer,
        ])

        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(model.weights), 2)

    def test_after_call(self) -> None:
        layer = layers.Dense(1)
        model = keras.Sequential([layer])
        model(tf.zeros((4, 2)))  # weights initialized

        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(model.weights), 2)

    def test_nested(self) -> None:
        base = keras.Sequential([
            keras.Input(shape=(2,)),
            layers.Dense(3),
        ])
        model = keras.Sequential([
            base,
            layers.Dense(1)
        ])

        output: tf.Tensor = model(tf.zeros(shape=(4, 2)))
        self.assertEqual(tuple(output.shape.as_list()), (4, 1))

    def test_intercept_layers(self) -> None:
        model = keras.Sequential([
            layers.Input(shape=(2,)),
            layers.Dense(4, name='first'),
            layers.Dense(3, name='second'),
            layers.Dense(2, name='third'),
            layers.Dense(1, name='fourth')
        ])
        model_intercept = keras.Model(
                inputs=model.get_layer(name="second").input,
                outputs=model.get_layer(name="third").output,
        )

        model_output: tf.Tensor = model(tf.zeros(shape=(4, 2)))
        model_intercept_output: tf.Tensor = model_intercept(tf.zeros(shape=(4, 4)))

        self.assertEqual(tuple(model_output.shape.as_list()), (4, 1))
        self.assertEqual(tuple(model_intercept_output.shape.as_list()), (4, 2))
