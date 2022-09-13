from os.path import join
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from tensorflow_.suppress_tf_warning import SuppressTFWarnings

_save_path: str = join(OUTPUT_PATH, 'saved')


class TestKerasTraces(TestCase):
    def test_with_traces(self) -> None:
        layer = tf.keras.layers.Dense(1)
        model = tf.keras.Sequential([layer])
        data: tf.Tensor = tf.constant([[42]])
        prediction: tf.Tensor = model(data)

        with SuppressTFWarnings():
            model.save(_save_path)
        model_reloaded: tf.keras.Sequential = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy().all().all())
        self.assertIs(type(model_reloaded), tf.keras.Sequential)

    def test_without_traces(self) -> None:
        layer = tf.keras.layers.Dense(1)
        model = tf.keras.Sequential([layer])
        data: tf.Tensor = tf.constant([[42]])
        prediction: tf.Tensor = model(data)

        with SuppressTFWarnings():
            model.save(_save_path)
        model_reloaded: tf.keras.Sequential = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy().all().all())
        self.assertIs(type(model_reloaded), tf.keras.Sequential)
