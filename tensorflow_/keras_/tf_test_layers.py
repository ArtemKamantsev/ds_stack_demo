from random import randint
from unittest import TestCase

import tensorflow as tf


class TestLayers(TestCase):
    def test_lstm_parameters_count(self) -> None:
        n_units: int = randint(10, 100)
        features_count: int = 42
        predicted_params_count = 4 * ((n_units + features_count) * n_units + n_units)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(n_units, input_shape=(None, features_count)),
        ])

        self.assertEqual(predicted_params_count, model.count_params())

    def test_sublayer_weights_initialization(self) -> None:
        initializer = tf.keras.initializers.GlorotUniform()

        class InverseBias(tf.keras.layers.Layer):
            _biases: tf.Variable

            def build(self, input_shape):
                self._biases = tf.Variable(tf.zeros((input_shape[-1])), name='biases')

            def call(self, inputs: tf.Tensor) -> tf.Tensor:
                return inputs + self._biases

        class IrregularDense(tf.keras.layers.Layer):
            _out_features: int
            sublayer: InverseBias
            _weights: tf.Variable

            def __init__(self, out_features: int, **kwargs):
                super().__init__(**kwargs)
                self._out_features = out_features
                self.sublayer = InverseBias()

            def build(self, input_shape):
                self._weights = self.add_weight(
                        shape=(input_shape[-1], self._out_features),
                        name='weights',
                        initializer=initializer,
                )

            def call(self, inputs: tf.Tensor) -> tf.Tensor:
                value: tf.Tensor = self.sublayer(inputs)
                return tf.matmul(value, self._weights)

        layer = IrregularDense(1, input_shape=(1,))
        model = tf.keras.Sequential([layer])

        self.assertEqual(len(layer.sublayer.weights), 1)
        self.assertEqual(tuple(layer.sublayer.weights[0].shape.as_list()), (1,))

    def test_training_param(self) -> None:
        class DifferentBehaviourLayer(tf.keras.layers.Layer):
            def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
                if training:
                    return tf.constant([1.0])

                return tf.constant([0.0])

        layer = DifferentBehaviourLayer(input_shape=(1,))
        model = tf.keras.Sequential([layer])
        model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.mean_squared_error,
        )

        x = [[0.0]]
        y = [1.0]

        target_loss: float = (x[0][0] - y[0]) ** 2
        history = model.fit(x, y, epochs=1, batch_size=1, verbose=0)
        eval_loss = model.evaluate(x, y, batch_size=1, verbose=0)

        self.assertAlmostEqual(history.history['loss'][0], 0.0)
        self.assertAlmostEqual(eval_loss, target_loss)



