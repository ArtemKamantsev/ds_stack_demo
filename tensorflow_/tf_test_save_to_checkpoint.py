from unittest import TestCase

import tensorflow as tf


class FlexibleDenseModule(tf.Module):
    is_built: bool
    out_features: int
    weights: tf.Variable
    bias: tf.Variable

    def __init__(self, out_features: int, name: str | None = None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        # Create variables on first call.
        if not self.is_built:
            self.weights = tf.Variable(
                    tf.random.normal([inputs.shape[-1], self.out_features]), name='weights')
            self.bias = tf.Variable(tf.zeros([self.out_features]), name='bias')
            self.is_built = True

        y = tf.matmul(inputs, self.weights) + self.bias
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    dense_1: FlexibleDenseModule
    dense_2: FlexibleDenseModule

    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.dense_1(inputs)
        return self.dense_2(inputs)


class TestSavingToCheckpoint(TestCase):
    def test_checkpoint(self) -> None:
        model_original = SequentialModule(name="the_model")
        data: tf.Tensor = tf.constant([[2.0, 2.0, 2.0]])
        prediction_original: tf.Tensor = model_original(data)

        checkpoint_path: str = "../output/checkpoint"
        checkpoint = tf.train.Checkpoint(model=model_original)
        checkpoint.write(checkpoint_path)

        variables_saved: list[tuple[str, list[int]]] = tf.train.list_variables(checkpoint_path)
        variable_names: tuple[str]
        variable_shapes: tuple[list[int] | tuple[int, ...], ...]
        variable_names, variable_shapes = zip(*variables_saved)
        variable_shapes = tuple(tuple(shape) for shape in variable_shapes)

        self.assertEqual(variable_names, ('_CHECKPOINTABLE_OBJECT_GRAPH',
                                          'model/dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE',
                                          'model/dense_1/weights/.ATTRIBUTES/VARIABLE_VALUE',
                                          'model/dense_2/bias/.ATTRIBUTES/VARIABLE_VALUE',
                                          'model/dense_2/weights/.ATTRIBUTES/VARIABLE_VALUE'))
        self.assertEqual(variable_shapes, ((), (3,), (3, 3), (2,), (3, 2)))

        model_restored = SequentialModule()
        checkpoint_restore = tf.train.Checkpoint(model=model_restored)
        checkpoint_restore.restore("../output/checkpoint")
        prediction_restored = model_restored(data)

        # noinspection PyTypeChecker
        prediction_comparison: tf.Tensor = prediction_restored == prediction_original
        self.assertTrue(prediction_comparison.numpy().all())
