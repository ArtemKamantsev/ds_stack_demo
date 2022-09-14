from collections.abc import Collection, Sequence
from numbers import Number
from typing import Callable, Any
from unittest import TestCase

import numpy as np


class RandomImage:
    shape: tuple[int, int]

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        random_image: np.ndarray = np.random.rand(*self.shape)
        random_image *= 254
        random_image += 1

        return random_image.astype(dtype)


# np.lib.mixins.NDArrayOperatorsMixin overrides __add__, __lt__ and so on via usage u-functions
class ValueHolder(np.lib.mixins.NDArrayOperatorsMixin):
    value: int

    def __init__(self, value: int):
        self.value = value


# u-function custom handling example
class RegularValue(ValueHolder):
    def __array_ufunc__(self, ufunc: Callable[..., int], method: str,
                        *inputs: list[Any], **kwargs: dict[Any, Any]) -> 'RegularValue':
        if method == '__call__':
            scalars: list[Number] = []
            for input_ in inputs:
                if isinstance(input_, Number):
                    scalars.append(input_)
                elif isinstance(input_, ValueHolder):
                    scalars.append(input_.value)
                else:
                    return NotImplemented
            return self.__class__(ufunc(*scalars, **kwargs))

        return NotImplemented


class TrickyValue(ValueHolder):
    def __array_ufunc__(self, ufunc: Callable[..., int], method: str,
                        *inputs: Sequence[Any], **kwargs: dict[Any, Any]) -> 'TrickyValue':
        if method == '__call__':
            scalars: list[Number] = []
            for input_ in inputs:
                if isinstance(input_, Number):
                    scalars.append(input_)
                elif isinstance(input_, ValueHolder):
                    scalars.append(input_.value)
                else:
                    return NotImplemented

            # noinspection PyTypeChecker
            scalars = [2 * value for value in scalars]
            return self.__class__(ufunc(*scalars, **kwargs))

        return NotImplemented


HANDLED_FUNCTIONS: dict[Callable, Callable[..., 'ValueWithFunctions']] = {}


# non u-function custom handling example
class ValueWithFunctions(ValueHolder):
    def __array_function__(self, func: Callable, types: Collection[type],
                           args: Sequence[Any], kwargs: dict[Any, Any]) -> Any:
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle ValueWithFunctions objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(np_function: Callable):
    """Register an __array_function__ implementation for ValueWithFunctions objects."""

    def decorator(func: Callable[..., ValueWithFunctions]):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.sum)
def square(value: ValueWithFunctions) -> ValueWithFunctions:
    return ValueWithFunctions(value.value ** 2)


class TestCustomArrayContainers(TestCase):
    def test_treat_as_ndarray(self) -> None:
        random_image: RandomImage = RandomImage((2, 2))
        # noinspection PyTypeChecker
        image_array: np.ndarray = np.asarray(random_image, dtype=int)

        self.assertTrue(isinstance(image_array, np.ndarray))
        self.assertEqual(image_array.shape, (2, 2))
        self.assertEqual(image_array.dtype, int)
        self.assertGreater(image_array.max(initial=-1), 1)
        self.assertLess(image_array.max(initial=-1), 255)

    def test_use_with_np_function(self) -> None:
        random_image: RandomImage = RandomImage((2, 2))
        random_image_scaled: np.ndarray = np.multiply(random_image, 1 / 255)
        self.assertTrue(isinstance(random_image_scaled, np.ndarray))
        self.assertEqual(random_image_scaled.shape, random_image.shape)
        self.assertEqual(random_image_scaled.dtype, float)
        # pylint: disable=unexpected-keyword-arg
        self.assertLess(random_image_scaled.max(initial=-1), 1)

    def test_pass_through_np_u_function(self) -> None:
        regular_value: RegularValue = RegularValue(17)

        # noinspection PyTypeChecker
        # pylint: disable=no-member
        result: RegularValue = np.add(regular_value, 1)
        self.assertTrue(isinstance(result, RegularValue))
        self.assertEqual(result.value, 18)

        result = np.multiply(result, 2)
        self.assertTrue(isinstance(result, RegularValue))
        self.assertEqual(result.value, 36)

        result: RegularValue = result / 2
        self.assertTrue(isinstance(result, RegularValue))
        self.assertEqual(result.value, 18)

        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            result = np.sum(regular_value)

    def test_pass_custom_np_u_function_priorities(self) -> None:
        value_regular: RegularValue = RegularValue(1)
        value_tricky: TrickyValue = TrickyValue(2)

        # noinspection PyTypeChecker
        result_regular: RegularValue = value_regular + 1
        self.assertTrue(isinstance(result_regular, RegularValue))
        self.assertEqual(result_regular.value, 2)

        # noinspection PyTypeChecker
        result_tricky: TrickyValue = value_tricky + 1
        self.assertTrue(isinstance(result_tricky, TrickyValue))
        self.assertEqual(result_tricky.value, 6)

        # Type of result depends on order of operands
        # Because non default implementation of ufunction application is taken from the first argument having such
        result_regular = value_regular + value_tricky
        self.assertTrue(isinstance(result_regular, RegularValue))
        self.assertEqual(result_regular.value, 3)

        result_tricky = value_tricky + value_regular
        self.assertTrue(isinstance(result_tricky, TrickyValue))
        self.assertEqual(result_tricky.value, 6)

    def test_pass_through_np_function(self) -> None:
        person: ValueWithFunctions = ValueWithFunctions(5)
        # noinspection PyTypeChecker
        # pylint: disable=no-member
        person_age_squared: ValueWithFunctions = np.sum(person)

        self.assertTrue(isinstance(person_age_squared, ValueWithFunctions))
        self.assertEqual(person_age_squared.value, 25)
