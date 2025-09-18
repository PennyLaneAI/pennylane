# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Overview
--------

The PennyLane NumPy subpackage provides a differentiable wrapper around NumPy, that enables
backpropagation through standard NumPy code.

This version of NumPy **must** be used when using PennyLane with the :doc:`Autograd interface
</introduction/interfaces/numpy>`:

>>> from pennylane import numpy as np

.. note::

    If using other interfaces, such as :doc:`TensorFlow </introduction/interfaces/tf>` :doc:`PyTorch
    </introduction/interfaces/torch>`, or :doc:`JAX </introduction/interfaces/jax>`, then the
    PennyLane-provided NumPy should not be used; instead, simply use the standard NumPy import.

This package is a wrapper around ``autograd.numpy``; for details on all available functions,
please refer to the `Autograd
docs <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md>`__.

PennyLane additionally extends Autograd with the following classes,
errors, and functions:

.. autosummary::
    :toctree: api
    :nosignatures:
    :template: autosummary/class_no_inherited.rst

    ~wrap_arrays
    ~extract_tensors
    ~tensor_wrapper
    ~tensor
    ~NonDifferentiableError

Caveats
-------

This package is a wrapper around ``autograd.numpy``, and therefore comes with several caveats
inherited from Autograd:

**Do not use:**

- Assignment to arrays, such as ``A[0, 0] = x``.

..

- Implicit casting of lists to arrays, for example ``A = np.sum([x, y])``.
  Make sure to explicitly cast to a NumPy array first, i.e.,
  ``A = np.sum(np.array([x, y]))`` instead.

..

- ``A.dot(B)`` notation. Use ``np.dot(A, B)`` or ``A @ B`` instead.

..

- In-place operations such as ``a += b``. Use ``a = a + b`` instead.

..

- Some ``isinstance`` checks, like ``isinstance(x, np.ndarray)`` or ``isinstance(x, tuple)``,
  without first doing ``from autograd.builtins import isinstance, tuple``.

For more details, please consult the `Autograd
docs <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md>`__.

"""
# pylint: disable=wrong-import-position,undefined-variable

from autograd import numpy as _np
from autograd.numpy import *

from pennylane.exceptions import NonDifferentiableError
from .wrapper import extract_tensors, tensor_wrapper, wrap_arrays

wrap_arrays(_np.__dict__, globals())

# Delete the unwrapped fft, linalg, random modules
# so that we can re-import our wrapped versions.
del fft
del linalg
del random

from . import fft, linalg, random
from .tensor import asarray as _asarray
from .tensor import tensor

asarray = tensor_wrapper(_asarray)
