from os.path import join
from unittest import TestCase

import tensorflow as tf

from constants import OUTPUT_PATH
from tensorflow_.suppress_tf_warning import SuppressTFWarnings
from .models import SimplestCustomModel, CustomModelWithParam, CustomModelWithParamConfig

_save_path: str = join(OUTPUT_PATH, 'saved')


class TestCustomNoTracesNoWeight(TestCase):
    def test_with_traces(self) -> None:
        model = SimplestCustomModel()

        with self.assertRaises(ValueError):
            with SuppressTFWarnings():
                model.save(_save_path)

    # noinspection PyCallingNonCallable
    def test_without_traces(self) -> None:
        model = SimplestCustomModel()
        model.save(_save_path, save_traces=False)
        test_data: tf.Tensor = tf.constant(42)
        prediction: tf.Tensor = model(test_data)

        model_reloaded: SimplestCustomModel = tf.keras.models.load_model(_save_path, compile=False)
        prediction_reloaded: tf.Tensor
        with self.assertRaises(ValueError):
            prediction_reloaded = model_reloaded(test_data)

        # you should provide custom_objects arg loading models with custom components
        model_reloaded = tf.keras.models.load_model(_save_path,
                                                    compile=False,
                                                    custom_objects={'SimplestCustomModel': SimplestCustomModel})
        prediction_reloaded: tf.Tensor = model_reloaded(test_data)

        # noinspection PyUnresolvedReferences
        self.assertTrue((prediction == prediction_reloaded).numpy())

    # noinspection PyCallingNonCallable, PyUnresolvedReferences
    def test_without_traces_substitute_fake_class(self) -> None:
        class FakeCustomModel(tf.keras.Model):
            def call(self, x: tf.Tensor) -> tf.Tensor:
                return x * tf.constant(2)

        model = SimplestCustomModel()
        model.save(_save_path, save_traces=False)
        model_reloaded: FakeCustomModel = tf.keras.models.load_model(_save_path,
                                                                     compile=False,
                                                                     custom_objects={
                                                                         'SimplestCustomModel': FakeCustomModel
                                                                     },
                                                                     )
        test_data: tf.Tensor = tf.constant(42)
        prediction: tf.Tensor = model(test_data)
        prediction_reloaded: tf.Tensor = model_reloaded(test_data)

        self.assertFalse((prediction == prediction_reloaded).numpy())
        # noinspection PyTypeChecker
        self.assertTrue((prediction * 2 == prediction_reloaded).numpy())

    def test_without_traces_with_param(self) -> None:
        model = CustomModelWithParam(42)
        # pylint: disable=duplicate-code
        model.save(_save_path, save_traces=False)
        with self.assertRaises(TypeError):
            # pylint: disable=unused-variable
            model_reloaded: CustomModelWithParam = \
                tf.keras.models.load_model(_save_path,
                                           compile=False,
                                           custom_objects={
                                               'CustomModelWithParam': CustomModelWithParam
                                           },
                                           )

    def test_without_traces_with_param_config(self) -> None:
        model = CustomModelWithParamConfig(42)
        model.save(_save_path, save_traces=False)
        model_reloaded: CustomModelWithParamConfig = \
            tf.keras.models.load_model(_save_path,
                                       compile=False,
                                       custom_objects={
                                           'CustomModelWithParamConfig': CustomModelWithParamConfig
                                       },
                                       )
        self.assertEqual(model_reloaded.value.numpy(), 43)
