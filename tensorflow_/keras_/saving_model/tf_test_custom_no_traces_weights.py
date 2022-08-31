from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from .models import CustomModelWithWeights

_save_path: str = join(OUTPUT_PATH, 'saved')


# For some reason, model with 'build' function cannot be restored without traces correctly
class TestCustomNoTracesWeights(TestCase):
    def test_with_traces(self) -> None:
        model = CustomModelWithWeights(42.0)

        with self.assertRaises(ValueError):
            model.save(_save_path)

    def test_without_traces(self) -> None:
        model = CustomModelWithWeights(42.0)
        model.save(_save_path, save_traces=False)
        model_reloaded: Any = tf.keras.models.load_model(_save_path,
                                                         compile=False,
                                                         custom_objects={
                                                             'CustomModelWithWeights': CustomModelWithWeights})

        self.assertIsNot(type(model_reloaded), CustomModelWithWeights)
        with self.assertRaises(ValueError):
            prediction_reloaded: tf.Tensor = model_reloaded(tf.constant([42.0]))

    def test_without_traces_built(self) -> None:
        model = CustomModelWithWeights(42.0)
        model.build((1,))
        model.save(_save_path, save_traces=False)
        model_reloaded: Any = tf.keras.models.load_model(_save_path,
                                                         compile=False,
                                                         custom_objects={
                                                             'CustomModelWithWeights': CustomModelWithWeights
                                                         },
                                                         )

        self.assertIsNot(type(model_reloaded), CustomModelWithWeights)
        with self.assertRaises(ValueError):
            prediction_reloaded: tf.Tensor = model_reloaded(tf.constant([42.0]))
