from unittest import TestCase

import tensorflow as tf


class TestPreprocessing(TestCase):
    def test_adapt(self) -> None:
        data: list[str] = [
            'w1 w2',
            'w3 W2',
        ]
        layer = tf.keras.layers.TextVectorization()
        layer.adapt(data)
        vectorized_text = layer(data)

        self.assertTrue((vectorized_text == [[4, 2],
                                             [3, 2]]).numpy().all().all())

    def test_manual_configuration(self):
        data: list[str] = [
            'w1 w2',
            'w3 W2',
        ]
        layer = tf.keras.layers.TextVectorization(vocabulary=['w1', 'w2'])
        vectorized_text = layer(data)

        self.assertTrue((vectorized_text == [[2, 3],
                                             [1, 3]]).numpy().all().all())  # 1 is reserved as OOV token
