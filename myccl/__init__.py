# flake8: noqa

# Patch for deprecated alias in Numpy >= 1.20.0 (used in ISiTGR & FAST-PT).
# Deprecation cycle starts in Numpy 1.20 and ends in Numpy 1.24.
from packaging.version import parse
import numpy
numpy.int = int if parse(numpy.__version__) >= parse("1.20.0") else numpy.int
del parse, numpy

from .core import *
from .parameters import *
from .interpolate import *
from .integrate import  *
from .pk2d import *
from .tk3d import *
from .pspec import *
from .neutrinos import *
from .musigma import *
from .tracers import *
from .errors import *
from .background import *
from .power import *
from . import halos
