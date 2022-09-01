from numbers import Number
from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf


# pylint: disable=abstract-method
class CustomModule(tf.Module):
    __weights: tf.Variable

    def __init__(self, factor: Number):
        super().__init__()
        self.__weights = tf.Variable(factor)

    @tf.function
    def multiply(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs * self.__weights


_save_path: str = './output/saved'


class TestModelSaving(TestCase):
    def test_saved_without_evaluation(self) -> None:
        data: tf.Tensor = tf.constant([1, 2, 3])
        model_original = CustomModule(2)

        tf.saved_model.save(model_original, _save_path)
        model_reloaded: Any = tf.saved_model.load(_save_path)
        with self.assertRaises(ValueError):  # Found zero restored functions for caller function.
            # pylint: disable=unused-variable
            prediction_reloaded_signature1: tf.Tensor = model_reloaded.multiply(data)

    def test_save_with_evaluation(self) -> None:
        data_signature1: tf.Tensor = tf.constant([1, 2, 3])
        data_signature2: tf.Tensor = tf.constant([1, 2])
        model_original = CustomModule(2)

        prediction_original_signature1: tf.Tensor = model_original.multiply(data_signature1)
        tf.saved_model.save(model_original, _save_path)
        prediction_original_signature2: tf.Tensor = model_original.multiply(data_signature2)
        # noinspection PyTypeChecker
        prediction_comparison_signature2: tf.Tensor = prediction_original_signature2 == [2, 4]
        self.assertTrue(prediction_comparison_signature2.numpy().all())

        model_reloaded: Any = tf.saved_model.load(_save_path)
        prediction_reloaded_signature1: tf.Tensor = model_reloaded.multiply(data_signature1)
        with self.assertRaises(ValueError):  # can not use data with different shape on reloaded model
            # pylint: disable=unused-variable
            prediction_reloaded_signature2: tf.Tensor = model_reloaded.multiply(data_signature2)
        # noinspection PyTypeChecker
        prediction_comparison_signature1: tf.Tensor = prediction_reloaded_signature1 == prediction_original_signature1

        self.assertIsInstance(model_original, CustomModule)
        # Reloaded model is an internal TensorFlow user object without any of the class knowledge.
        self.assertNotIsInstance(model_reloaded, CustomModule)
        self.assertTrue(prediction_comparison_signature1.numpy().all())

    def test_save_model_restore_from_checkpoint(self):
        data: tf.Tensor = tf.constant([1, 2, 3])
        model_original = CustomModule(2)

        prediction_original: tf.Tensor = model_original.multiply(data)
        tf.saved_model.save(model_original, _save_path)

        model_restored = CustomModule(-1)
        checkpoint_restore = tf.train.Checkpoint(model=model_restored)
        checkpoint_restore.restore(join(_save_path, 'variables', 'variables'))
        prediction_restored = model_restored.multiply(data)

        prediction_comparison: tf.Tensor = prediction_restored != prediction_original
        self.assertTrue(prediction_comparison.numpy().all())
