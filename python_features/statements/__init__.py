from itertools import chain

from .test_del_statement import *
from .test_global_statement import *
from .test_nonlocal_statement import *
from . import test_del_statement, test_global_statement, test_nonlocal_statement

# need this to avoid exporting submodule names using * syntax
__all__ = list(chain(test_del_statement.__all__, test_global_statement.__all__, test_nonlocal_statement.__all__))
