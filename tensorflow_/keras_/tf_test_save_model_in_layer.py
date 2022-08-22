from unittest import TestCase

import tensorflow as tf


class FlexibleDense(tf.keras.Model):
    out_features: int
    weights_local: tf.Variable
    biases_local: tf.Variable

    def __init__(self, out_features: int, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.weights_local = tf.Variable(
                tf.random.normal([input_shape[-1], self.out_features]), name='weights')
        self.biases_local = tf.Variable(tf.zeros([self.out_features]), name='biases')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self.weights_local) + self.biases_local


class Sequential(tf.keras.layers.Layer):
    dense_1: FlexibleDense
    dense_2: FlexibleDense

    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDense(out_features=3)
        self.dense_2 = FlexibleDense(out_features=2)

    # noinspection PyCallingNonCallable
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.dense_1(x)
        return self.dense_2(x)


_save_path: str = './output/saved'


class TestModelSaving(TestCase):
    def test_save_model(self) -> None:
        data: tf.Tensor = tf.constant([[2.0, 2.0, 2.0]])
        model = Sequential(name='the_model')
        # noinspection PyCallingNonCallable
        prediction_original = model(data)

        self.assertIsNotNone(tf.math.reduce_sum(prediction_original).numpy())

        tf.saved_model.save(model, _save_path)
        model_reloaded = tf.saved_model.load(_save_path)

        with self.assertRaises(ValueError):
            prediction_reloaded = model_reloaded(data)
