from itertools import chain

from .test_skipping import *
from .custom_test_case import *
from . import test_skipping
from . import custom_test_case

# need this to avoid exporting submodule names using * syntax
__all__ = list(chain(custom_test_case.__all__, test_skipping.__all__))
