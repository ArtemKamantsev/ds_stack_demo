from unittest import TestCase

import tensorflow as tf


class TestVariables(TestCase):
    def test_operations(self) -> None:
        variable: tf.Variable = tf.Variable([0, 1, 2])
        result: tf.Tensor = tf.math.reduce_sum(variable)

        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.numpy(), 3)

    def test_reshape(self) -> None:
        variable: tf.Variable = tf.Variable([0, 1])
        reshaped: tf.Tensor = tf.reshape(variable, (1, 2))

        self.assertIsInstance(reshaped, tf.Tensor)  # Reshaping creates a new tensor; it does not reshape the variable.
        self.assertEqual(tuple(reshaped.shape.as_list()), (1, 2))

    def test_assign(self) -> None:
        variable: tf.Variable = tf.Variable([0, 1])
        variable.assign([1, 2])

        self.assertTrue((variable.numpy() == [1, 2]).all())

        # shapes should be equal:
        with self.assertRaises(ValueError):
            variable.assign([0, 1, 2])

        with self.assertRaises(ValueError):
            variable.assign([0])

        with self.assertRaises(ValueError):
            variable.assign([[0, 1]])
