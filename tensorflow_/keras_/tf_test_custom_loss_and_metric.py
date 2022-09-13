from typing import Callable
from unittest import TestCase

import tensorflow as tf

from tensorflow_.suppress_tf_warning import SuppressTFWarnings


class TestCustomLossAndMetric(TestCase):
    _constant: float
    _target: float
    x: tf.Tensor
    y: tf.Tensor

    def setUp(self) -> None:
        self._constant: float = 40.0
        self._target: float = 42.0

        self._x = tf.constant([[self._constant]])
        self._y = tf.constant([self._target])

    @staticmethod
    def __build_model(main_layer: tf.keras.layers.Layer | None = None,
                      loss: tf.keras.losses.Loss | Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
                      metrics: list[tf.keras.metrics.Metric] | None = None) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=(1,))
        layer = main_layer(input_layer) if main_layer is not None else input_layer
        model = tf.keras.Model(inputs=input_layer, outputs=layer)
        model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=loss,
                metrics=metrics,
        )

        return model

    def __evaluate_model(self, model: tf.keras.Model) -> float | list[float]:
        with SuppressTFWarnings():
            model.fit(self._x, self._y, epochs=1, batch_size=1, verbose=0)

            return model.evaluate(self._x, self._y, batch_size=1, verbose=0)

    def test_dummy_model(self) -> None:
        target_loss: float = (self._target - self._constant) ** 2
        target_metric: float = self._target - self._constant

        model: tf.keras.Model = self.__build_model(loss=tf.keras.losses.mean_squared_error,
                                                   metrics=[tf.keras.metrics.MeanAbsoluteError()])
        loss: float
        metric: float
        loss, metric = self.__evaluate_model(model)

        self.assertAlmostEqual(loss, target_loss)
        self.assertAlmostEqual(metric, target_metric)

    def test_custom_loss_function(self) -> None:
        def custom_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return tf.math.reduce_mean(tf.square(y_true - y_pred)) + self._constant

        target_loss: float = (self._target - self._constant) ** 2 + self._constant

        model: tf.keras.Model = self.__build_model(loss=custom_mean_squared_error)
        loss: float = self.__evaluate_model(model)

        self.assertAlmostEqual(loss, target_loss)

    def test_custom_loss_class(self) -> None:
        class CustomMSE(tf.keras.losses.Loss):
            additional_term: float

            def __init__(self, additional_term: float):
                super().__init__(name='custom_mse')
                self.additional_term = additional_term

            # labels/predictions count corresponds to batch_size
            def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                return tf.math.reduce_mean(tf.square(y_true - y_pred)) + self.additional_term

        target_loss: float = (self._target - self._constant) ** 2 + self._constant

        model: tf.keras.Model = self.__build_model(loss=CustomMSE(self._constant))
        loss: float = self.__evaluate_model(model)

        self.assertAlmostEqual(loss, target_loss)

    def test_custom_metric(self):
        class CustomMeanAbsoluteError(tf.keras.metrics.Metric):
            additional_term: float

            def __init__(self, additional_term: float, **kwargs):
                super().__init__(name='custom_mean_absolute_error', **kwargs)
                self.mean_absolute_error = self.add_weight(name="cmae", initializer="zeros")
                self.additional_term = additional_term

            def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> None:
                values: tf.Tensor = y_true - y_pred
                if sample_weight is not None:
                    sample_weight = tf.cast(sample_weight, "float32")
                    values *= sample_weight

                self.mean_absolute_error.assign_add(tf.reduce_mean(values) + self.additional_term)

            def result(self) -> tf.Tensor:
                return self.mean_absolute_error

            def reset_state(self) -> None:
                # The state of the metric will be reset at the start of each epoch.
                self.mean_absolute_error.assign(0.0)

        term: float = 1.0
        target_metric: float = self._target - self._constant + term

        model: tf.keras.Model = self.__build_model(loss=tf.keras.losses.mean_squared_error,
                                                   metrics=[CustomMeanAbsoluteError(term)])
        metric: float
        _, metric = self.__evaluate_model(model)

        self.assertAlmostEqual(metric, target_metric)

    def test_additional_loss(self):
        term: float = 1.0

        class AdditionalLossLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                self.add_loss(term)
                return inputs  # Pass-through layer.

        target_loss: float = term
        # loss function could be omitted as loss created inside layers
        model: tf.keras.Model = self.__build_model(main_layer=AdditionalLossLayer())
        self.assertEqual(len(model.losses), 1)

        loss: float = self.__evaluate_model(model)

        self.assertAlmostEqual(loss, target_loss)
        self.assertEqual(len(model.losses), 1)

    def test_additional_loss_functional(self):
        term: float = 1.0
        target_loss: float = (self._target - self._constant) ** 2 + term

        model: tf.keras.Model = self.__build_model(loss=tf.keras.losses.mean_squared_error, )
        self.assertEqual(len(model.losses), 0)

        model.add_loss(lambda: term)
        loss: float = self.__evaluate_model(model)

        self.assertAlmostEqual(loss, target_loss)
        self.assertEqual(len(model.losses), 1)

    def test_additional_metric(self):
        term: float = 1.0

        class AdditionalMetricLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                self.add_metric(term, name='additional_term', aggregation='mean')
                return inputs  # Pass-through layer.

        target_metric: float = term
        model: tf.keras.Model = self.__build_model(main_layer=AdditionalMetricLayer(),
                                                   loss=tf.keras.losses.mean_squared_error)
        self.assertEqual(len(model.metrics), 1)

        metric: float
        _, metric = self.__evaluate_model(model)

        self.assertAlmostEqual(metric, target_metric)
        self.assertEqual(len(model.metrics), 2)

    def test_additional_metric_functional(self):
        term: float = 1.0
        target_metric: float = self._constant + term

        model = self.__build_model(loss=tf.keras.losses.mean_squared_error)
        model.add_metric(model.outputs[0] + term, name='additional_term', aggregation='mean')
        metric: float
        _, metric = self.__evaluate_model(model)

        self.assertAlmostEqual(metric, target_metric)
