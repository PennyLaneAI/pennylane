# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This package provides a wrapped version of autograd.numpy, such that
it works with the PennyLane :class:`~.tensor` class.
"""
# pylint: disable=wrong-import-position,wildcard-import,undefined-variable
from autograd import numpy as _np
from autograd.numpy import *

from .wrapper import wrap_arrays, extract_tensors

wrap_arrays(_np.__dict__, globals())

# Delete the unwrapped fft, linalg, random modules
# so that we can re-import our wrapped versions.
del fft
del linalg
del random

from . import fft
from . import linalg
from . import random

from .tensor import tensor, NonDifferentiableError

__doc__ = "NumPy with automatic differentiation support, provided by Autograd and PennyLane."
