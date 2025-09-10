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
r"""This module contains the ResourceConfig class, which tracks the configuration for resource estimation"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from .ops.op_math.controlled_ops import (
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
)
from .ops.qubit.parametric_ops_single_qubit import (
    ResourceRX,
    ResourceRY,
    ResourceRZ,
)
from .resource_operator import ResourceOperator
from .templates import (
    ResourceAliasSampling,
    ResourceMPSPrep,
    ResourcePrepTHC,
    ResourceQROMStatePreparation,
    ResourceQubitizeTHC,
    ResourceQubitUnitary,
    ResourceSelectPauliRot,
    ResourceSelectTHC,
)


class DecompositionType(StrEnum):
    """Specifies the type of decomposition to override."""

    ADJOINT = "adj"
    CONTROLLED = "ctrl"
    POW = "pow"
    BASE = "base"


class ResourceConfig:
    """A container to track the configuration for precisions, and custom decompositions for the
    resource estimation pipeline.
    """

    def __init__(self) -> None:
        DEFAULT_PRECISION = 1e-9
        DEFAULT_BIT_PRECISION = 15
        self.resource_op_precisions = {
            ResourceRX: {"precision": DEFAULT_PRECISION},
            ResourceRY: {"precision": DEFAULT_PRECISION},
            ResourceRZ: {"precision": DEFAULT_PRECISION},
            ResourceCRX: {"precision": DEFAULT_PRECISION},
            ResourceCRY: {"precision": DEFAULT_PRECISION},
            ResourceCRZ: {"precision": DEFAULT_PRECISION},
            ResourceSelectPauliRot: {"precision": DEFAULT_PRECISION},
            ResourceQubitUnitary: {"precision": DEFAULT_PRECISION},
            ResourceQROMStatePreparation: {"precision": DEFAULT_PRECISION},
            ResourceMPSPrep: {"precision": DEFAULT_PRECISION},
            ResourceAliasSampling: {"precision": DEFAULT_PRECISION},
            ResourceQubitizeTHC: {
                "rotation_precision": DEFAULT_BIT_PRECISION,
                "coeff_precision": DEFAULT_BIT_PRECISION,
            },
            ResourceSelectTHC: {
                "rotation_precision": DEFAULT_BIT_PRECISION,
                "coeff_precision": DEFAULT_BIT_PRECISION,
            },
            ResourcePrepTHC: {
                "rotation_precision": DEFAULT_BIT_PRECISION,
                "coeff_precision": DEFAULT_BIT_PRECISION,
            },
        }
        self._custom_decomps = {}
        self._adj_custom_decomps = {}
        self._ctrl_custom_decomps = {}
        self._pow_custom_decomps = {}

    def __str__(self) -> str:
        decomps = [op.__name__ for op in self._custom_decomps]
        adj_decomps = [f"Adjoint({op.__name__})" for op in self._adj_custom_decomps]
        ctrl_decomps = [f"Controlled({op.__name__})" for op in self._ctrl_custom_decomps]
        pow_decomps = [f"Pow({op.__name__})" for op in self._pow_custom_decomps]

        all_op_strings = decomps + adj_decomps + ctrl_decomps + pow_decomps
        op_names = ", ".join(all_op_strings)

        dict_items_str = ",\n".join(
            f"    {key.__name__}: {value!r}" for key, value in self.resource_op_precisions.items()
        )

        formatted_dict = f"{{\n{dict_items_str}\n}}"

        return (
            f"ResourceConfig(\n"
            f"  precisions = {formatted_dict},\n"
            f"  custom decomps = [{op_names}]\n)"
        )

    def __repr__(self) -> str:
        return f"ResourceConfig(precisions = {self.resource_op_precisions}, custom_decomps = {self._custom_decomps}, adj_custom_decomps = {self._adj_custom_decomps}, ctrl_custom_decomps = {self._ctrl_custom_decomps}, pow_custom_decomps = {self._pow_custom_decomps})"

    def set_single_qubit_rotation_precision(self, precision: float) -> None:
        r"""Sets the synthesis precision for all single-qubit rotation gates.

        This is a convenience method to update the synthesis precision tolerance,
        :math:`\eps`, for all standard single-qubit rotation gates and their
        controlled versions at once. The synthesis precision dictates the precision
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
            precision (float): The desired synthesis precision tolerance. A smaller
                value corresponds to a higher precision compilation, which may
                increase the required gate counts.

        **Example**

        .. code-block:: python

            from pennylane.labs.resource_estimation import ResourceConfig
            from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import ResourceRX

            config = ResourceConfig()
            print(f"Default RX precision: {config.resource_op_precisions[ResourceRX]['precision']}")

            config.set_single_qubit_rotation_precision(1e-5)
            print(f"Updated RX precision: {config.resource_op_precisions[ResourceRX]['precision']}")

        .. code-block:: pycon

            Default RX precision: 1e-09
            Updated RX precision: 1e-05
        """
        self.resource_op_precisions[ResourceRX]["precision"] = precision
        self.resource_op_precisions[ResourceCRX]["precision"] = precision
        self.resource_op_precisions[ResourceRY]["precision"] = precision
        self.resource_op_precisions[ResourceCRY]["precision"] = precision
        self.resource_op_precisions[ResourceRZ]["precision"] = precision
        self.resource_op_precisions[ResourceCRZ]["precision"] = precision

    def set_decomp(
        self,
        op_type: type[ResourceOperator],
        decomp_func: Callable,
        decomp_type: DecompositionType | None = "base",
    ) -> None:
        """Set a custom function to override the default resource decomposition.

        Args:
            cls (Type[ResourceOperator]): the operator class whose decomposition is being overriden.
            decomp_func (Callable): the new resource decomposition function to be set as default.
            decomp_type (None | DecompositionType): the decomposition type to override. Options are
                "adj", "pow", "controlled",
                and "base". Default is "base".

        .. note::

            The new decomposition function, ``decomp_func``, should have the same signature as the one it replaces.
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

        if decomp_type == DecompositionType.ADJOINT:
            self._adj_custom_decomps[op_type] = decomp_func
        elif decomp_type == DecompositionType.CONTROLLED:
            self._ctrl_custom_decomps[op_type] = decomp_func
        elif decomp_type == DecompositionType.POW:
            self._pow_custom_decomps[op_type] = decomp_func
        else:
            self._custom_decomps[op_type] = decomp_func
