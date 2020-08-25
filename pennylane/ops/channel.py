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
This module contains the available built-in noisy
quantum channels supported by PennyLane, as well as their conventions.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import numpy as np

from pennylane.operation import AnyWires, Channel


class QubitChannel(Channel):
    r"""QubitChannel(K_list, wires)
    Apply an arbitrary fixed quantum channel.

    Kraus matrices that represent the fixed channel are provided
    as a list of numpy arrays.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        K_list (list(array[complex])): list of Kraus matrices
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "R"
    grad_method = None

    @classmethod
    def _kraus_matrices(cls, *params):
        K_list = params[0]
        K_dag = [K.conj().T for K in K_list]
        Kraus_sum = np.sum(np.array([a @ b for a, b in zip(K_dag, K_list)]), axis=0)

        if Kraus_sum != np.eye(2** num_wires):
            raise ValueError("Only trace preserving channels can be applied.")

        if not all(K.shape == K_list[0].shape for K in K_list):
            raise ValueError("All Kraus matrices must have the same shape.")

        if not all(K.shape[0] == K.shape[1] for K in K_list):
            raise ValueError("Only channels with equal input and output Hilbert space dimensions can be applied.")

        return K_list

__qubit_channels__ = {"QubitChannel"}

