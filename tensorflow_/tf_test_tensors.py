from unittest import TestCase

import tensorflow as tf


class TestTensors(TestCase):
    def test_shape(self) -> None:
        tensor: tf.Tensor = tf.zeros([3, 2, 4, 5])
        shape: tf.TensorShape = tensor.shape
        shape_tensor: tf.Tensor = tf.shape(tensor)

        self.assertIsInstance(shape, tf.TensorShape)
        self.assertEqual(tuple(shape.as_list()), (3, 2, 4, 5))
        self.assertIsInstance(shape_tensor, tf.Tensor)
        self.assertTrue((shape_tensor.numpy() == (3, 2, 4, 5)).all())

    def test_reshape(self) -> None:
        tensor: tf.Tensor = tf.zeros((4,))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            tf.reshape(tensor, [3, -1])

    def test_cast(self) -> None:
        tensor_float: tf.Tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float32)
        tensor_int: tf.Tensor = tf.cast(tensor_float, dtype=tf.int32)

        self.assertTrue((tensor_int.numpy() == [2, 3, 4]).all())

    def test_ragged_tensor_shape(self) -> None:
        ragged_list: list[list[int]] = [[0, 1, 2],
                                        [3, 4]]
        with self.assertRaises(ValueError):
            tf.constant(ragged_list)

        tensor: tf.Tensor = tf.ragged.constant(ragged_list)
        shape: tf.TensorShape = tensor.shape

        self.assertIsInstance(shape, tf.TensorShape)
        self.assertEqual(tuple(shape.as_list()), (2, None))

    def test_string_split(self) -> None:
        tensor_string: tf.Tensor = tf.constant('first second')
        tensor_string_list: tf.Tensor = tf.strings.split(tensor_string, sep=' ')

        self.assertEqual(tensor_string.ndim, 0)
        self.assertIsInstance(tensor_string_list, tf.Tensor)
        self.assertEqual(tuple(tensor_string_list.numpy().astype(str)), ('first', 'second'))

    def test_string_list_split(self) -> None:
        tensor_string_list: tf.Tensor = tf.constant(['first second', 'first second third'])
        tensor_ragged_string_list: tf.RaggedTensor = tf.strings.split(tensor_string_list, sep=' ')

        self.assertEqual(tuple(tensor_string_list.shape.as_list()), (2,))
        self.assertIsInstance(tensor_ragged_string_list, tf.RaggedTensor)
        self.assertEqual(tuple(tensor_ragged_string_list.shape.as_list()), (2, None))

    def test_sparse_tensor(self) -> None:
        sparse_tensor: tf.SparseTensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                                                values=[1, 2],
                                                                dense_shape=[3, 4])
        dense_tensor: tf.Tensor = tf.sparse.to_dense(sparse_tensor)

        self.assertTrue((dense_tensor.numpy() == [[1, 0, 0, 0],
                                                  [0, 0, 2, 0],
                                                  [0, 0, 0, 0]]).all().all())
