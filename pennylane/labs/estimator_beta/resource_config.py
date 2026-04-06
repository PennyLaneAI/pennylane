# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains the LabsResourceConfig class, which tracks the configuration for resource estimation"""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_config import ResourceConfig

from .ops import (
    ch_resource_decomp,
    hadamard_controlled_resource_decomp,
    paulirot_controlled_resource_decomp,
    mcx_one_clean_aux_resource_decomp,
)
from .templates import selectpaulirot_controlled_resource_decomp


class LabsResourceConfig(ResourceConfig):
    """Sets the values of precisions and custom decompositions when estimating resources for a
    quantum workflow (see also :class:`~.pennylane.estimator.resource_config.ResourceConfig`).

    .. note::

        The ``LabsResourceConfig`` class inherits from :class:`~.pennylane.estimator.resource_config.ResourceConfig`
        and comes preloaded with many optimized custom resource decompositions and custom symbolic resource
        decompositions. The preloaded decompositions can be accessed via the following attributes:
        ``config.custom_decomps``, ``config.pow_custom_decomps``, ``config.adj_custom_decomps``, and
        ``config.ctrl_custom_decomps``.

    The precisions and custom decompositions of resource operators can be
    modified using the :meth:`~.pennylane.labs.estimator_beta.resource_config.LabsResourceConfig.set_precision`
    and :meth:`~.pennylane.labs.estimator_beta.resource_config.LabsResourceConfig.set_decomp` functions of the
    :code:`LabsResourceConfig` class.

    **Example**

    This example shows how to set a custom precision value for every instance of the :code:`RX` gate.

    .. code-block:: pycon

        >>> import pennylane.labs.estimator_beta as qre
        >>> my_config = qre.LabsResourceConfig()
        >>> my_config.set_precision(qre.RX, precision=1e-5)
        >>> res = qre.estimate(
        ...     qre.RX(),
        ...     gate_set={"RZ", "T", "Hadamard"},
        ...     config=my_config,
        ... )
        >>> print(res)
        --- Resources: ---
         Total wires: 1
           algorithmic wires: 1
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 28
           'T': 28

    The :code:`LabsResourceConfig` can also be used to set custom decompositions. The following example
    shows how to define a custom decomposition for the ``RX`` gate.

    .. code-block:: pycon

        >>> def custom_RX_decomp(precision):  # RX = H @ RZ @ H
        ...     h = qre.Hadamard.resource_rep()
        ...     rz = qre.RZ.resource_rep(precision)
        ...     return [qre.GateCount(h, 2), qre.GateCount(rz, 1)]
        >>>
        >>> my_config = qre.LabsResourceConfig()
        >>> my_config.set_decomp(qre.RX, custom_RX_decomp)
        >>> res = qre.estimate(
        ...     qre.RX(precision=None),
        ...     gate_set={"RZ", "T", "Hadamard"},
        ...     config=my_config,
        ... )
        >>> print(res)
        --- Resources: ---
         Total wires: 1
           algorithmic wires: 1
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 3
           'RZ': 1,
           'Hadamard': 2

    """

    def __init__(self):
        super().__init__()

        # Add modified decomps here:
        custom_decomps = {
            qre.CH: ch_resource_decomp,
            qre.MultiControlledX: mcx_one_clean_aux_resource_decomp,
        }
        pow_custom_decomps = {}
        adj_custom_decomps = {}
        ctrl_custom_decomps = {
            qre.PauliRot: paulirot_controlled_resource_decomp,
            qre.SelectPauliRot: selectpaulirot_controlled_resource_decomp,
            qre.Hadamard: hadamard_controlled_resource_decomp,
        }

        self._custom_decomps = custom_decomps
        self._pow_custom_decomps = pow_custom_decomps
        self._adj_custom_decomps = adj_custom_decomps
        self._ctrl_custom_decomps = ctrl_custom_decomps
