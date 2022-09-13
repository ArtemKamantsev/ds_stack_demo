from os.path import join
from typing import Any
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from .models import SimplestCustomModel, CustomModelWithParam

_save_path: str = join(OUTPUT_PATH, 'saved')


class TestCustomTracesNoWeight(TestCase):
    # noinspection PyCallingNonCallable
    def test_with_traces(self) -> None:
        model = SimplestCustomModel()
        # pylint: disable=duplicate-code
        data: tf.Tensor = tf.constant(42)
        prediction: tf.Tensor = model(data)

        model.save(_save_path)
        model_reloaded: Any = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
        self.assertIsNot(type(model_reloaded), SimplestCustomModel)  # only calculation graph actually was reloaded

    # noinspection PyCallingNonCallable
    def test_with_traces_restore_class(self) -> None:
        model = SimplestCustomModel()
        data: tf.Tensor = tf.constant(42)
        prediction: tf.Tensor = model(data)

        model.save(_save_path)
        model_reloaded: SimplestCustomModel = tf.keras.models.load_model(_save_path,
                                                                         compile=False,
                                                                         custom_objects={
                                                                             'SimplestCustomModel': SimplestCustomModel
                                                                         })
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
        self.assertIs(type(model_reloaded), SimplestCustomModel)  # full model was reloaded

    def test_with_traces_restore_class_with_param(self) -> None:
        model = CustomModelWithParam(42)
        data: tf.Tensor = tf.constant(42)
        # noinspection PyCallingNonCallable
        model(data)

        model.save(_save_path)
        with self.assertRaises(TypeError):
            # Despite calculation graph could be loaded instead of model, exception is raised
            # pylint: disable=unused-variable
            model_reloaded: CustomModelWithParam = \
                tf.keras.models.load_model(_save_path,
                                           compile=False,
                                           custom_objects={
                                               'CustomModelWithParam': CustomModelWithParam
                                           },
                                           )

    # noinspection PyCallingNonCallable
    def test_without_traces(self) -> None:
        model = SimplestCustomModel()
        data: tf.Tensor = tf.constant(42)
        prediction: tf.Tensor = model(data)

        model.save(_save_path, save_traces=False)
        model_reloaded: SimplestCustomModel = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor
        with self.assertRaises(ValueError):
            prediction_reloaded = model_reloaded(data)

        model_reloaded: SimplestCustomModel = tf.keras.models.load_model(_save_path,
                                                                         compile=False,
                                                                         custom_objects={
                                                                             'SimplestCustomModel': SimplestCustomModel,
                                                                         },
                                                                         )
        prediction_reloaded: tf.Tensor = model_reloaded(data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())
