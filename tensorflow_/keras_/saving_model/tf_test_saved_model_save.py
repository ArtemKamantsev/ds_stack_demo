from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH


class FlexibleDense(tf.keras.layers.Layer):
    __out_features: int
    __weights: tf.Variable
    __biases: tf.Variable

    def __init__(self, out_features: int, **kwargs):
        super().__init__(**kwargs)
        self.__out_features = out_features

    def build(self, input_shape):
        self.__weights = tf.Variable(
                tf.random.normal([input_shape[-1], self.__out_features]), name='weights')
        self.__biases = tf.Variable(tf.zeros([self.__out_features]), name='biases')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self.__weights) + self.__biases

    def get_config(self) -> dict[str, Any]:
        return {"out_features": self.__out_features}


class Sequential(tf.keras.Model):
    __dense_1: FlexibleDense
    __dense_2: FlexibleDense

    def __init__(self, name=None):
        super().__init__(name=name)

        self.__dense_1 = FlexibleDense(out_features=3)
        self.__dense_2 = FlexibleDense(out_features=2)

    # noinspection PyCallingNonCallable
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.__dense_1(x)
        return self.__dense_2(x)

    def get_config(self):
        return {"name": self.name, **super().get_config()}


_save_path: str = join(OUTPUT_PATH, 'saved')


class TestModelSaving(TestCase):
    def test_save_raw_model(self) -> None:
        model = Sequential(name='the_model')
        tf.saved_model.save(model, _save_path)

        model_reloaded: Any = tf.saved_model.load(_save_path)
        with self.assertRaises(TypeError):
            # pylint: disable=unused-variable
            prediction: tf.Tensor = model_reloaded(tf.constant([[2.0, 2.0, 2.0]]))

    def test_compiled_model(self) -> None:
        model = Sequential(name='the_model')
        model.compile()
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
