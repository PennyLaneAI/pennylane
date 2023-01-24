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
from typing import Union

import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox

Tensor = Union[int, float, bool, complex, bytes, str, np.ndarray, ArrayBox]

with contextlib.suppress(ImportError):
    import jax.numpy as jnp

    Tensor = Union[Tensor, jnp.ndarray]
with contextlib.suppress(ImportError):
    import torch

    Tensor = Union[Tensor, torch.Tensor]
with contextlib.suppress(ImportError):
    import tensorflow as tf

    Tensor = Union[Tensor, tf.Tensor, tf.Variable]
