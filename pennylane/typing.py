# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains different PennyLane types."""
import contextlib

# pylint: disable=import-outside-toplevel, too-few-public-methods
import sys
from typing import Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox


class _Typing:
    """Class containing useful PennyLane types."""

    @property
    def TensorLike(self):
        """Returns a ``Union`` of all tensor-like types, which includes any scalar or sequence
        that can be interpreted as a pennylane tensor, including lists and tuples. Any argument
        accepted by ``pnp.array`` is tensor-like.

        **Examples**
        For ``python >= 3.10``, ``typing.TensorLike`` can be used in ``isinstance`` checks:

        >>> from pennylane import typing
        >>> isinstance(4, typing.TensorLike)
        True
        >>> isinstance([2, 6, 8], typing.TensorLike)
        True
        >>> isinstance(torch.tensor([1, 2, 3]), typing.TensorLike)
        True
        """
        tensor_like = Union[int, float, bool, complex, bytes, str, np.ndarray, ArrayBox]

        if "jax" in sys.modules:
            # Need to add this because (for an unknown reason) when executing
            # ``pytest tests/ -m torch`` without jax installed, ``sys.modules`` contains a "jax" module.
            with contextlib.suppress(ImportError):
                from jax.numpy import ndarray

                tensor_like = Union[tensor_like, ndarray]
        if "torch" in sys.modules:
            from torch import Tensor as torchTensor

            tensor_like = Union[tensor_like, torchTensor]
        if "tensorflow" in sys.modules or "tensorflow-macos" in sys.modules:
            from tensorflow import Tensor as tfTensor
            from tensorflow import Variable

            tensor_like = Union[tensor_like, tfTensor, Variable]
        return tensor_like


typing = _Typing()
