from os.path import join
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH

_save_path: str = join(OUTPUT_PATH, 'saved')


class TestKerasNoTraces(TestCase):
    def test_with_traces(self) -> None:
        layer = tf.keras.layers.Dense(1)
        model = tf.keras.Sequential([layer])

        with self.assertRaises(ValueError):
            model.save(_save_path)

    def test_with_traces_with_input(self) -> None:
        layer = tf.keras.layers.Dense(1, input_shape=(1,))
        model = tf.keras.Sequential([layer])
        model.save(_save_path)

        model_reloaded: tf.keras.Sequential = tf.keras.models.load_model(_save_path, compile=False)
        data: tf.Tensor = tf.constant([[42]])
        prediction: tf.Tensor = model(data)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy().all().all())
        self.assertIs(type(model_reloaded), tf.keras.Sequential)

    def test_without_traces(self) -> None:
        layer = tf.keras.layers.Dense(1)
        model = tf.keras.Sequential([layer])

        model.save(_save_path, save_traces=False)
        with self.assertRaises(ValueError):
            model_reloaded: tf.keras.Sequential = tf.keras.models.load_model(_save_path)

    def test_without_traces_input_shape(self) -> None:
        layer = tf.keras.layers.Dense(1, input_shape=(1,))
        model = tf.keras.Sequential([layer])
        model.save(_save_path)

        model_reloaded: tf.keras.Sequential = tf.keras.models.load_model(_save_path, compile=False)
        data: tf.Tensor = tf.constant([[42]])
        prediction: tf.Tensor = model(data)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy().all().all())
        self.assertIs(type(model_reloaded), tf.keras.Sequential)
