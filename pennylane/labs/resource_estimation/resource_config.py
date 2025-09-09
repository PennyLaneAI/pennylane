# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains the base class for qubit management"""

from __future__ import annotations

from collections.abc import Callable

from pennylane.labs.resource_estimation.ops.op_math.controlled_ops import (
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
)
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    ResourceRX,
    ResourceRY,
    ResourceRZ,
)
from pennylane.labs.resource_estimation.resource_operator import ResourceOperator
from pennylane.labs.resource_estimation.templates import (
    ResourceAliasSampling,
    ResourceMPSPrep,
    ResourcePrepTHC,
    ResourceQROMStatePreparation,
    ResourceQubitizeTHC,
    ResourceQubitUnitary,
    ResourceSelectPauliRot,
    ResourceSelectTHC,
)


class ResourceConfig:
    r"""A container to track the configuration for errors, precisions, and custom decompositions for the
    resource estimation pipeline.
    """

    def __init__(self) -> None:
        self.errors_and_precisions = {
            ResourceRX: {"precision": 1e-9},
            ResourceRY: {"precision": 1e-9},
            ResourceRZ: {"precision": 1e-9},
            ResourceCRX: {"precision": 1e-9},
            ResourceCRY: {"precision": 1e-9},
            ResourceCRZ: {"precision": 1e-9},
            ResourceSelectPauliRot: {"precision": 1e-9},
            ResourceQubitUnitary: {"precision": 1e-9},
            ResourceQROMStatePreparation: {"precision": 1e-9},
            ResourceMPSPrep: {"precision": 1e-9},
            ResourceAliasSampling: {"precision": 1e-9},
            ResourceQubitizeTHC: {"rotation_precision": 15, "coeff_precision": 15},
            ResourceSelectTHC: {"rotation_precision": 15, "coeff_precision": 15},
            ResourcePrepTHC: {"rotation_precision": 15, "coeff_precision": 15},
        }
        self._custom_decomps = {}
        self._adj_custom_decomps = {}
        self._ctrl_custom_decomps = {}
        self._pow_custom_decomps = {}

    def __str__(self):
        return f"ResourceConfig(errors_and_precisions = {self.errors_and_precisions}, custom_decomps = {self._custom_decomps}, {self._adj_custom_decomps}, {self._ctrl_custom_decomps}, {self._pow_custom_decomps})"

    def __repr__(self) -> str:
        return f"ResourceConfig(errors_and_precisions = {self.errors_and_precisions}, custom_decomps = {self._custom_decomps}, {self._adj_custom_decomps}, {self._ctrl_custom_decomps}, {self._pow_custom_decomps})"

    def set_single_qubit_rotation_error(self, error: float):
        r"""Sets the synthesis error for all single-qubit rotation gates.

        This is a convenience method to update the synthesis error tolerance,
        :math:`\precision`, for all standard single-qubit rotation gates and their
        controlled versions at once. The synthesis error dictates the precision
        for compiling rotation gates into a discrete gate set, which in turn
        affects the number of gates required.

        This method updates the ``precision`` value for the following operators:
        - :class:`~.ResourceRX`
        - :class:`~.ResourceRY`
        - :class:`~.ResourceRZ`
        - :class:`~.ResourceCRX`
        - :class:`~.ResourceCRY`
        - :class:`~.ResourceCRZ`

        Args:
            error (float): The desired synthesis error tolerance. A smaller
                value corresponds to a higher precision compilation, which may
                increase the required gate counts.

        **Example**

        .. code-block:: python

            from pennylane.labs.resource_estimation import ResourceConfig
            from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import ResourceRX

            config = ResourceConfig()
            print(f"Default RX error: {config.errors_and_precisions[ResourceRX]['precision']}")

            config.set_single_qubit_rotation_error(1e-5)
            print(f"Updated RX error: {config.errors_and_precisions[ResourceRX]['precision']}")

        .. code-block:: pycon

            Default RX error: 1e-09
            Updated RX error: 1e-05
        """
        self.errors_and_precisions[ResourceRX]["precision"] = error
        self.errors_and_precisions[ResourceCRX]["precision"] = error
        self.errors_and_precisions[ResourceRY]["precision"] = error
        self.errors_and_precisions[ResourceCRY]["precision"] = error
        self.errors_and_precisions[ResourceRZ]["precision"] = error
        self.errors_and_precisions[ResourceCRZ]["precision"] = error

    def set_decomp(
        self, op_type: type[ResourceOperator], decomp_func: Callable, type: str = None
    ) -> None:
        """Set a custom function to override the default resource decomposition.

        Args:
            cls (Type[ResourceOperator]): the operator class whose decomposition is being overriden.
            decomp_func (Callable): the new resource decomposition function to be set as default.
            type (str): the decomposition type to override e.g. "adj" or "ctrl"

        .. note::

            The new decomposition function should have the same signature as the one it replaces.
            Specifically, the signature should match the :code:`resource_keys` of the base resource
            operator class being overriden.

        **Example**

        .. code-block:: python

            from pennylane.labs import resource_estimation as plre

            def custom_res_decomp(**kwargs):
                h = plre.resource_rep(plre.ResourceHadamard)
                s = plre.resource_rep(plre.ResourceS)
                return [plre.GateCount(h, 1), plre.GateCount(s, 2)]

        .. code-block:: pycon

            >>> print(plre.estimate_resources(plre.ResourceX(), gate_set={"Hadamard", "Z", "S"}))
            --- Resources: ---
            Total qubits: 1
            Total gates : 4
            Qubit breakdown:
              clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
            Gate breakdown:
              {'Hadamard': 2, 'S': 2}
            >>> config = plre.ResourceConfig()
            >>> config.set_decomp(plre.ResourceX, custom_res_decomp)
            >>> print(plre.estimate_resources(plre.ResourceX(), gate_set={"Hadamard", "Z", "S"}, config=config))
            --- Resources: ---
            Total qubits: 1
            Total gates : 3
            Qubit breakdown:
              clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
            Gate breakdown:
              {'S': 1, 'Hadamard': 2}
        """

        if type == "adj":
            self._adj_custom_decomps[op_type] = decomp_func
        elif type == "ctrl":
            self._ctrl_custom_decomps[op_type] = decomp_func
        elif type == "pow":
            self._pow_custom_decomps[op_type] = decomp_func
        else:
            self._custom_decomps[op_type] = decomp_func
