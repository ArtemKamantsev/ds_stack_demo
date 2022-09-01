from unittest import TestCase

import tensorflow as tf


class TestAutodiff(TestCase):
    def test_default_watch_behaviour(self) -> None:
        trainable: tf.Variable = tf.Variable(3.0)
        not_trainable: tf.Variable = tf.Variable(3.0, trainable=False)
        tensor: tf.Tensor = tf.constant(3.0)

        with tf.GradientTape() as tape:
            y = trainable ** 2 + not_trainable ** 2 + tensor ** 2

        trainable_grad: tf.Tensor
        not_trainable_grad: tf.Tensor
        tensor_grad: tf.Tensor
        trainable_grad, not_trainable_grad, tensor_grad = tape.gradient(y, [trainable, not_trainable, tensor])

        self.assertIsNotNone(trainable_grad)
        self.assertEqual(trainable_grad.numpy(), 6.0)  # dy/d(trainable) = 2 * trainable
        self.assertIsNone(not_trainable_grad)
        self.assertIsNone(tensor_grad)

    def test_modified_watch_behaviour(self) -> None:
        variable1: tf.Variable = tf.Variable(3.0)
        variable2: tf.Variable = tf.Variable(3.0)
        tensor: tf.Tensor = tf.constant(3.0)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variable2)
            tape.watch(tensor)
            y = variable1 ** 2 + variable2 ** 2 + tensor ** 2

        variable1_grad: tf.Tensor
        variable2_grad: tf.Tensor
        tensor_grad: tf.Tensor
        variable1_grad, variable2_grad, tensor_grad = tape.gradient(y, [variable1, variable2, tensor])

        self.assertIsNone(variable1_grad)
        self.assertIsNotNone(variable2_grad)
        self.assertEqual(variable2_grad.numpy(), 6.0)  # dy/d(variable2) = 2 * variable2
        self.assertIsNotNone(tensor_grad)
        self.assertEqual(tensor_grad.numpy(), 6.0)  # dy/d(tensor) = 2 * tensor

    def test_int_variable(self) -> None:
        trainable_int: tf.Variable = tf.Variable(3)

        with tf.GradientTape() as tape:
            y = trainable_int ** 2
        trainable_int_grad: tf.Tensor = tape.gradient(y, trainable_int)

        self.assertIsNone(trainable_int_grad)

    def test_gradient_through_stateful_object(self) -> None:
        variable1: tf.Variable = tf.Variable(3.0)
        variable2: tf.Variable = tf.Variable(0.0)

        with tf.GradientTape(persistent=True) as tape:
            variable2.assign_add(variable1)
            y = variable2 ** 2

        grad: tf.Tensor = tape.gradient(y, variable1)
        self.assertIsNone(grad)

        grad = tape.gradient(y, variable1, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.assertEqual(grad.numpy(), 0)

    def test_non_gradient_registered(self) -> None:
        image: tf.Variable = tf.Variable([[[0.5, 0.0, 0.0]]])
        delta: tf.Variable = tf.Variable(0.1)

        with tf.GradientTape(persistent=True) as tape:
            new_image: tf.Tensor = tf.image.adjust_contrast(image, delta)

        with self.assertRaises(LookupError):
            # pylint: disable=unused-variable
            grad: list[tf.Tensor] = tape.gradient(new_image, [image, delta])

    def test_several_gradients_computation(self):
        variable: tf.Variable = tf.Variable(3.0)
        with tf.GradientTape() as tape_unpersistent:
            intermediate: tf.Tensor = variable ** 2
            result: tf.Tensor = intermediate ** 2

        # By default, the resources held by a GradientTape are released
        # as soon as the GradientTape.gradient method is called.
        self.assertEqual(tape_unpersistent.gradient(result, intermediate).numpy(), 18.0)
        with self.assertRaises(RuntimeError):
            result: tf.Tensor = tape_unpersistent.gradient(result, variable)

        with tf.GradientTape(persistent=True) as tape_persistent:
            intermediate: tf.Tensor = variable ** 2
            result: tf.Tensor = intermediate ** 2

        self.assertEqual(tape_persistent.gradient(result, intermediate).numpy(), 18.0)
        self.assertEqual(tape_persistent.gradient(intermediate, variable).numpy(), 6.0)

        del tape_persistent  # resources are released when the GradientTape(persistent=True) object is garbage collected

    def test_multiple_targets(self):
        variable: tf.Variable = tf.Variable(3.0)

        with tf.GradientTape() as tape:
            target1: tf.Tensor = variable ** 2
            target2: tf.Tensor = 2 * variable

        grad: tf.Tensor = tape.gradient([target1, target2], variable)
        self.assertEqual(grad.numpy(), 8.0)

    def test_multi_output_target(self):
        variable: tf.Variable = tf.Variable(3.0)

        with tf.GradientTape() as tape:
            target_multi_output: tf.Tensor = variable * [2, 3]

        grad: tf.Tensor = tape.gradient(target_multi_output, variable)
        self.assertEqual(grad.numpy(), 5.0)

    def test_differentiate_with_flow_control(self):
        tensor: tf.Tensor = tf.constant(1.0)
        variable1: tf.Variable = tf.Variable(2.0)
        variable2: tf.Variable = tf.Variable(2.0)

        with tf.GradientTape() as tape:
            tape.watch(tensor)
            if tensor > 0.0:
                result = 2 * variable1
            else:
                result = variable2 ** 2

        variable1_grad: tf.Tensor
        variable2_grad: tf.Tensor
        tensor_grad: tf.Tensor
        variable1_grad, variable2_grad, tensor_grad = tape.gradient(result, [variable1, variable2, tensor])

        self.assertIsNotNone(variable1_grad)
        self.assertIsNone(variable2_grad)  # variable2 hasn't been used
        self.assertIsNone(tensor_grad)  # tensor hasn't been used
