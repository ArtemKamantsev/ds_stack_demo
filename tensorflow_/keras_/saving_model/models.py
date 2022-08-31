import tensorflow as tf
from typing import Any


class SimplestCustomModel(tf.keras.Model):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x


class CustomModelWithParam(tf.keras.Model):
    _value: tf.Tensor

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self._value = tf.constant(value)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self._value


class CustomModelWithParamConfig(CustomModelWithParam):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self._value

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({'value': self._value.numpy()})

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'CustomModelWithParamConfig':
        config['value'] += 1

        return cls(**config)


class CustomModelWithWeights(tf.keras.Model):
    _shift: tf.Tensor
    _weights: tf.Variable

    def __init__(self, shift: float, **kwargs):
        super().__init__(**kwargs)
        self._shift = tf.constant(shift, dtype=tf.float32)

    def build(self, input_shape: tuple[int, ...]) -> None:
        self._weights = tf.Variable(tf.ones(input_shape, dtype=tf.float32), name='weights')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * self._weights + self._shift

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({'shift': self._shift.numpy()})

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'CustomModelWithWeights':
        return cls(**config)
