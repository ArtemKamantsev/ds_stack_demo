from typing import Any

import tensorflow as tf


class SimplestCustomModel(tf.keras.Model):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x


class CustomModelWithParam(tf.keras.Model):
    value: tf.Tensor

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = tf.constant(value)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.value


class CustomModelWithParamConfig(CustomModelWithParam):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.value

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({'value': self.value.numpy()})

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'CustomModelWithParamConfig':
        config['value'] += 1

        return cls(**config)


class CustomModelWithWeights(tf.keras.Model):
    shift: tf.Tensor
    weights_: tf.Variable

    def __init__(self, shift: float, **kwargs):
        super().__init__(**kwargs)
        self.shift = tf.constant(shift, dtype=tf.float32)

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.weights_ = tf.Variable(tf.ones(input_shape, dtype=tf.float32), name='weights')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * self.weights_ + self.shift

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({'shift': self.shift.numpy()})

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'CustomModelWithWeights':
        return cls(**config)
