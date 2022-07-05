from itertools import chain

from . import test_import_package_module, test_multiple_directories_package
from .test_import_package_module import *
from .test_multiple_directories_package import *

# need this to avoid exporting submodule names using * syntax
__all__ = list(chain(test_import_package_module.__all__, test_multiple_directories_package.__all__))
