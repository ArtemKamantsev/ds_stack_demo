from types import TracebackType

import absl.logging
import tensorflow as tf


class SuppressTFWarnings:
    def __enter__(self) -> None:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        absl.logging.set_verbosity(absl.logging.ERROR)

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
        absl.logging.set_verbosity(absl.logging.WARNING)
