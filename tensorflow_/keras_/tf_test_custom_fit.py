from unittest import TestCase

import numpy as np
import tensorflow as tf


class TestCustomFit(TestCase):
    def test_custom_train_step(self) -> None:
        class CustomModel(tf.keras.Model):
            @tf.function  # won't give any performance gain on 1 run, but in general should be used
            def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> dict[str, float]:
                batch_data: tf.Tensor
                batch_inputs: tf.Tensor
                batch_data, batch_labels = inputs
                y_pred: tf.Tensor = batch_data

                self.compiled_loss(batch_labels, y_pred, regularization_losses=self.losses)
                self.compiled_metrics.update_state(batch_labels, y_pred)

                result: dict[str, float] = {m.name: m.result() for m in self.metrics}
                result['custom_metric'] = 42.0

                return result

        model = CustomModel()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        data: np.ndarray = np.array([[0]])
        labels: np.ndarray = np.array([2])
        history: dict[str, list[float]] = model.fit(data, labels, epochs=1, batch_size=1, verbose=0).history

        self.assertAlmostEqual(history['loss'][0], 4.0)
        self.assertAlmostEqual(history['mae'][0], 2.0)
        self.assertAlmostEqual(history['custom_metric'][0], 42.0)

    def test_sample_weights(self):
        class CustomModel(tf.keras.Model):
            @tf.function
            def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
                _, _, batch_sample_weights = inputs

                return {'sample_weights': batch_sample_weights}

        model = CustomModel()
        model.compile()

        data: np.ndarray = np.array([[0]])
        labels: np.ndarray = np.array([2])
        with self.assertRaises(ValueError):
            # 'fit' checks for size of 'sample_weight'
            model.fit(data, labels, sample_weight=np.array([1, 2]), epochs=1, batch_size=1, verbose=0)

        sample_weights: np.ndarray = np.array([1])
        history: dict[str, np.ndarray] = model.fit(data, labels, sample_weight=sample_weights, epochs=1, batch_size=1,
                                                   verbose=0).history

        self.assertEqual(history['sample_weights'][0], sample_weights)

    def test_class_weights(self):
        class CustomModel(tf.keras.Model):
            @tf.function
            def train_step(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
                _, _, batch_class_sample_weight = inputs

                return {'class_sample_weight': batch_class_sample_weight}

        model = CustomModel()
        model.compile()

        data: np.ndarray = np.array([[0]])
        labels: np.ndarray = np.array([0])
        sample_weights: np.ndarray = np.array([2])
        class_weights: dict[int, int] = {
            0: 3,
            1: 4,
        }
        class_sample_weight: np.ndarray = sample_weights * [class_weights[lbl] for lbl in labels]
        history: dict[str, np.ndarray] = model.fit(data, labels,
                                                   sample_weight=sample_weights,
                                                   class_weight=class_weights,
                                                   epochs=1,
                                                   batch_size=1,
                                                   verbose=0
                                                   ).history

        self.assertEqual(history['class_sample_weight'][0], class_sample_weight)
