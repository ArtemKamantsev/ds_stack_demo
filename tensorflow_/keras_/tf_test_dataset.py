from unittest import TestCase

import tensorflow as tf

from tensorflow_.suppress_tf_warning import SuppressTFWarnings


class TestDataset(TestCase):
    def test_steps_per_epoch_param(self):
        input_layer = tf.keras.Input(shape=(1,))
        model = tf.keras.Model(inputs=input_layer, outputs=input_layer)
        model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.mean_squared_error,
        )

        epochs: int = 4
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(([[0], [1], [2], [3]], [0, 1, 2, 3])).batch(1)
        # history: dict[str: list[float]] = model.fit(dataset, epochs=epochs, verbose=0).history
        # self.assertEqual(len(history['loss']), epochs)

        with SuppressTFWarnings():
            # trying to get more then epochs * steps_per_epoch batches from dataset
            history: dict[str: list[float]] = model.fit(dataset, epochs=epochs, verbose=0, steps_per_epoch=2).history
        self.assertLess(len(history['loss']), epochs)
