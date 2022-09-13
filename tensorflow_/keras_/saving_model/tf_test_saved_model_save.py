from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from tensorflow_.suppress_tf_warning import SuppressTFWarnings


class FlexibleDense(tf.keras.layers.Layer):
    _out_features: int
    _weights: tf.Variable
    _biases: tf.Variable

    def __init__(self, out_features: int, **kwargs):
        super().__init__(**kwargs)
        self._out_features = out_features

    def build(self, input_shape):
        self._weights = tf.Variable(
                tf.random.normal([input_shape[-1], self._out_features]), name='weights')
        self._biases = tf.Variable(tf.zeros([self._out_features]), name='biases')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self._weights) + self._biases

    def get_config(self) -> dict[str, Any]:
        return {"out_features": self._out_features}


class Sequential(tf.keras.Model):
    _dense_1: FlexibleDense
    _dense_2: FlexibleDense

    def __init__(self, name=None):
        super().__init__(name=name)

        self._dense_1 = FlexibleDense(out_features=3)
        self._dense_2 = FlexibleDense(out_features=2)

    # noinspection PyCallingNonCallable
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._dense_1(x)
        return self._dense_2(x)

    def get_config(self):
        return {"name": self.name, **super().get_config()}


_save_path: str = join(OUTPUT_PATH, 'saved')


class TestModelSaving(TestCase):
    def test_save_raw_model(self) -> None:
        model = Sequential(name='the_model')
        with SuppressTFWarnings():
            tf.saved_model.save(model, _save_path)

        model_reloaded: Any = tf.saved_model.load(_save_path)
        with self.assertRaises(TypeError):
            # pylint: disable=unused-variable
            prediction: tf.Tensor = model_reloaded(tf.constant([[2.0, 2.0, 2.0]]))

    def test_compiled_model(self) -> None:
        model = Sequential(name='the_model')
        model.compile()
        with SuppressTFWarnings():
            tf.saved_model.save(model, _save_path)

        model_reloaded: Any = tf.saved_model.load(_save_path)
        with self.assertRaises(TypeError):
            # pylint: disable=unused-variable
            prediction: tf.Tensor = model_reloaded(tf.constant([[2.0, 2.0, 2.0]]))

    def test_save_built_model(self) -> None:
        model = Sequential(name='the_model')
        model.build(input_shape=(1, 3))

        with self.assertRaises(ValueError):
            tf.saved_model.save(model, _save_path)

    def test_save_model(self) -> None:
        data: tf.Tensor = tf.constant([[2.0, 2.0, 2.0]])
        model = Sequential(name='the_model')
        # noinspection PyCallingNonCallable
        prediction_original: tf.Tensor = model(data)

        tf.saved_model.save(model, _save_path)
        model_reloaded: Any = tf.saved_model.load(_save_path)

        prediction_reloaded: tf.Tensor = model_reloaded(data)
        # noinspection PyTypeChecker
        prediction_comparison: tf.Tensor = prediction_reloaded == prediction_original

        self.assertTrue(prediction_comparison.numpy().all())
        self.assertIsNot(type(model_reloaded), Sequential)  # the graph only has been retrieved
