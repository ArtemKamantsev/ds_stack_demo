from unittest import TestCase

import numpy as np
import tensorflow as tf


# pylint: disable=protected-access
class TestMaskingAndPadding(TestCase):
    def test_padding(self) -> None:
        sequence_list: list[list[int]] = [
            [1],
            [1, 2],
            [1, 2, 3],
        ]
        sequence_padded: list[list[int]] = tf.keras.preprocessing.sequence.pad_sequences(
                sequence_list, padding='post'
        )

        self.assertTrue((sequence_padded == np.array([[1, 0, 0],
                                                      [1, 2, 0],
                                                      [1, 2, 3]])).all().all())

    def test_masking(self) -> None:
        sequence_list: list[list[int]] = [
            [1],
            [1, 2],
            [1, 2, 3],
        ]
        sequence_padded: list[list[int]] = tf.keras.preprocessing.sequence.pad_sequences(
                sequence_list, padding='post'
        )

        embedding = tf.keras.layers.Embedding(input_dim=4, output_dim=16, mask_zero=True)
        masked_embedding: tf.Tensor = embedding(sequence_padded)

        sequence_padded_3d: tf.Tensor = tf.cast(
                tf.tile(tf.expand_dims(sequence_padded, axis=-1), [1, 1, 10]), tf.float32
        )
        masking_layer = tf.keras.layers.Masking()
        masked_masking_layer: tf.Tensor = masking_layer(sequence_padded_3d)

        self.assertTrue((masked_embedding._keras_mask.numpy() == [[True, False, False],
                                                                  [True, True, False],
                                                                  [True, True, True]]).all().all())
        # noinspection PyUnresolvedReferences
        self.assertTrue((masked_embedding._keras_mask == masked_masking_layer._keras_mask).numpy().all().all())

    # noinspection PyCallingNonCallable
    def test_mask_custom(self) -> None:
        class ReduceMask3D(tf.keras.layers.Layer):
            def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor | None:
                if mask is None:
                    return None

                mask_pattern = [i % 2 != 0 for i in range(inputs.shape[1])]

                return tf.tile(tf.constant([mask_pattern]), (inputs.shape[0], 1))

            def call(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
                return tf.reduce_sum(inputs * tf.cast(mask, tf.float32), axis=2)

        class MaskConsumer2D(tf.keras.layers.Layer):
            def call(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
                return tf.reduce_sum(inputs * tf.cast(mask, tf.float32), axis=1)

        data_3d: np.ndarray = np.array([[[1, 2], [3, 4]]], dtype=float)
        mask_3d: list[list[list[bool]]] = [[[False, True], [True, True]]]
        output1: tf.Tensor = ReduceMask3D()(data_3d, mask=mask_3d)
        output2: tf.Tensor = MaskConsumer2D()(output1)

        self.assertTrue((output1.numpy() == [[2, 7]]).all().all())
        self.assertTrue((output1._keras_mask.numpy() == [[False, True]]).all().all())
        self.assertTrue((output2.numpy() == [7]).all())
        self.assertFalse(hasattr(output2, '_keras_mask'))
