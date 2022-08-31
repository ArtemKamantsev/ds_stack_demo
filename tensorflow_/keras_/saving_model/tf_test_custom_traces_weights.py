from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from .models import CustomModelWithWeights

_save_path: str = join(OUTPUT_PATH, 'saved')


class TestCustomTracesWeights(TestCase):
    # noinspection PyCallingNonCallable
    def test_with_traces(self) -> None:
        model = CustomModelWithWeights(42)
        data: tf.Tensor = tf.constant([21.0])
        prediction: tf.Tensor = model(data)

        model.save(_save_path)
        model_reloaded: Any = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
        self.assertIsNot(type(model_reloaded), CustomModelWithWeights)  # only calculation graph actually was reloaded

    # noinspection PyCallingNonCallable
    def test_with_traces_restore_class(self) -> None:
        model = CustomModelWithWeights(42)
        data: tf.Tensor = tf.constant([21.0])
        prediction: tf.Tensor = model(data)

        model.save(_save_path)
        model_reloaded: CustomModelWithWeights = tf.keras.models.load_model(_save_path,
                                                                            compile=False,
                                                                            custom_objects={
                                                                                'CustomModelWithWeights': CustomModelWithWeights
                                                                            })
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
        self.assertIs(type(model_reloaded), CustomModelWithWeights)  # full model was reloaded
        # noinspection PyUnresolvedReferences
        self.assertTrue((model_reloaded._shift == model._shift).numpy())
        self.assertEqual(tuple(model_reloaded._weights.shape.as_list()), tuple(model._weights.shape.as_list()))

    # noinspection PyCallingNonCallable
    def test_without_traces(self) -> None:
        model = CustomModelWithWeights(42)
        data: tf.Tensor = tf.constant([42.0])
        prediction: tf.Tensor = model(data)

        model.save(_save_path, save_traces=False)
        model_reloaded: CustomModelWithWeights = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor
        with self.assertRaises(ValueError):
            prediction_reloaded = model_reloaded(data)

        model_reloaded: CustomModelWithWeights = tf.keras.models.load_model(_save_path,
                                                                            compile=False,
                                                                            custom_objects={
                                                                                'CustomModelWithWeights': CustomModelWithWeights,
                                                                            },
                                                                            )
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
        self.assertIs(type(model_reloaded), CustomModelWithWeights)
