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
This package provides a wrapped version of autograd.numpy.random, such that
it works with the PennyLane :class:`~.tensor` class.
"""
import semantic_version

from autograd.numpy import random as _random
from numpy import __version__ as np_version
from numpy.random import MT19937, PCG64, Philox, SFC64  # pylint: disable=unused-import

from .wrapper import wrap_arrays, tensor_wrapper

wrap_arrays(_random.__dict__, globals())


np_version_spec = semantic_version.SimpleSpec(">=0.17.0")
if np_version_spec.match(semantic_version.Version(np_version)):
    # pylint: disable=too-few-public-methods
    # pylint: disable=missing-class-docstring
    class Generator(_random.Generator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.__doc__ = "PennyLane wrapped NumPy Generator object\n" + super().__doc__

            for name in dir(_random.Generator):
                if name[0] != "_":
                    self.__dict__[name] = tensor_wrapper(getattr(super(), name))

    # pylint: disable=missing-function-docstring
    def default_rng(seed=None):
        # Mostly copied from NumPy, but uses our Generator instead

        if hasattr(seed, "capsule"):  # I changed this line
            # We were passed a BitGenerator, so just wrap it up.
            return Generator(seed)
        if isinstance(seed, Generator):
            # Pass through a Generator.
            return seed
        # Otherwise we need to instantiate a new BitGenerator and Generator as
        # normal.
        return Generator(PCG64(seed))

    default_rng.__doc__ = (
        "PennyLane duplicated generator constructor\n" + _random.default_rng.__doc__
    )
