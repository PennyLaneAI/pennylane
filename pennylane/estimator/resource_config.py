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
from typing import TYPE_CHECKING

from pennylane.estimator.ops.op_math.controlled_ops import CRX, CRY, CRZ
from pennylane.estimator.ops.qubit.matrix_ops import QubitUnitary
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ
from pennylane.estimator.templates import (
    AliasSampling,
    MPSPrep,
    PrepTHC,
    QROMStatePreparation,
    QubitizeTHC,
    SelectPauliRot,
    SelectTHC,
)
from pennylane.estimator.templates.trotter import TrotterVibrational, TrotterVibronic

if TYPE_CHECKING:
    from pennylane.estimator.resource_operator import ResourceOperator


class DecompositionType(StrEnum):
    """Specifies the type of decomposition to override."""

    ADJOINT = "adj"
    CONTROLLED = "ctrl"
    POW = "pow"
    BASE = "base"


class ResourceConfig:
    """Sets the values of precisions and custom decompositions when estimating resources for a
    quantum workflow.

    The precisions and custom decompositions of resource operators can be
    modified using the :meth:`~.pennylane.estimator.resource_config.ResourceConfig.set_precision`
    and :meth:`~.pennylane.estimator.resource_config.ResourceConfig.set_decomp` functions of the
    :code:`ResourceConfig` class.

    **Example**

    This example shows how to set a custom precision value for every instance of the :code:`RX` gate.

    .. code-block:: pycon

        >>> import pennylane.estimator as qre
        >>> my_config = qre.ResourceConfig()
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

    The :code:`ResourceConfig` can also be used to set custom decompositions. The following example
    shows how to define a custom decomposition for the ``RX`` gate.

    .. code-block:: pycon

        >>> def custom_RX_decomp(precision):  # RX = H @ RZ @ H
        ...     h = qre.Hadamard.resource_rep()
        ...     rz = qre.RZ.resource_rep(precision)
        ...     return [qre.GateCount(h, 2), qre.GateCount(rz, 1)]
        >>>
        >>> my_config = qre.ResourceConfig()
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

    def __init__(self) -> None:
        _DEFAULT_PRECISION = 1e-9
        _DEFAULT_BIT_PRECISION = 15
        _DEFAULT_PHASEGRAD_PRECISION = 1e-6
        self.resource_op_precisions = {
            RX: {"precision": _DEFAULT_PRECISION},
            RY: {"precision": _DEFAULT_PRECISION},
            RZ: {"precision": _DEFAULT_PRECISION},
            CRX: {"precision": _DEFAULT_PRECISION},
            CRY: {"precision": _DEFAULT_PRECISION},
            CRZ: {"precision": _DEFAULT_PRECISION},
            SelectPauliRot: {"precision": _DEFAULT_PRECISION},
            QubitUnitary: {"precision": _DEFAULT_PRECISION},
            AliasSampling: {"precision": _DEFAULT_PRECISION},
            MPSPrep: {"precision": _DEFAULT_PRECISION},
            QROMStatePreparation: {"precision": _DEFAULT_PRECISION},
            SelectTHC: {"rotation_precision": _DEFAULT_BIT_PRECISION},
            PrepTHC: {"coeff_precision": _DEFAULT_BIT_PRECISION},
            QubitizeTHC: {
                "coeff_precision": _DEFAULT_BIT_PRECISION,
                "rotation_precision": _DEFAULT_BIT_PRECISION,
            },
            TrotterVibronic: {
                "phase_grad_precision": _DEFAULT_PHASEGRAD_PRECISION,
                "coeff_precision": 1e-3,
            },
            TrotterVibrational: {
                "phase_grad_precision": _DEFAULT_PHASEGRAD_PRECISION,
                "coeff_precision": 1e-3,
            },
        }
        self._custom_decomps = {}
        self._adj_custom_decomps = {}
        self._ctrl_custom_decomps = {}
        self._pow_custom_decomps = {}

    @property
    def custom_decomps(self) -> dict[type[ResourceOperator], Callable]:
        """Returns the dictionary of custom base decompositions."""
        return self._custom_decomps

    @property
    def adj_custom_decomps(self) -> dict[type[ResourceOperator], Callable]:
        """Returns the dictionary of custom adjoint decompositions."""
        return self._adj_custom_decomps

    @property
    def ctrl_custom_decomps(self) -> dict[type[ResourceOperator], Callable]:
        """Returns the dictionary of custom controlled decompositions."""
        return self._ctrl_custom_decomps

    @property
    def pow_custom_decomps(self) -> dict[type[ResourceOperator], Callable]:
        """Returns the dictionary of custom power decompositions."""
        return self._pow_custom_decomps

    def __str__(self) -> str:
        decomps = [op.__name__ for op in self.custom_decomps]
        adj_decomps = [f"Adjoint({op.__name__})" for op in self.adj_custom_decomps]
        ctrl_decomps = [f"Controlled({op.__name__})" for op in self.ctrl_custom_decomps]
        pow_decomps = [f"Pow({op.__name__})" for op in self.pow_custom_decomps]

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
        return f"ResourceConfig(precisions = {self.resource_op_precisions}, custom_decomps = {self.custom_decomps}, adj_custom_decomps = {self.adj_custom_decomps}, ctrl_custom_decomps = {self.ctrl_custom_decomps}, pow_custom_decomps = {self.pow_custom_decomps})"

    def set_precision(self, op_type: type[ResourceOperator], precision: float) -> None:
        r"""Sets the precision for a given resource operator.

        This method updates the precision value for operators that use a single
        tolerance parameter (e.g., for synthesis error). It will raise an error
        if you attempt to set the precision for an operator that is not
        configurable or uses bit-precisions. A negative precision will also raise an error.

        Args:
            op_type (type[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): the operator class for which
                to set the precision
            precision (float): The desired precision tolerance. A smaller
                value corresponds to a higher precision compilation, which may
                increase the required gate counts. Must be greater than 0.

        Raises:
            ValueError: If ``op_type`` is not a configurable operator or if setting
                the precision for it is not supported, or if ``precision`` is negative.

        **Example**

        .. code-block:: python

            import pennylane.estimator as qre

            config = qre.ResourceConfig()

            # Check the default precision
            default = config.resource_op_precisions[qre.SelectPauliRot]['precision']
            print(f"Default precision for SelectPauliRot: {default}")

            # Set a new precision
            config.set_precision(qre.SelectPauliRot, precision=1e-5)
            new = config.resource_op_precisions[qre.SelectPauliRot]['precision']
            print(f"New precision for SelectPauliRot: {new}")

        .. code-block:: pycon

            Default precision for SelectPauliRot: 1e-09
            New precision for SelectPauliRot: 1e-05
        """
        if precision < 0:
            raise ValueError(f"Precision must be a non-negative value, but got {precision}.")

        if op_type not in self.resource_op_precisions:
            configurable_ops = sorted(
                [
                    op.__name__
                    for op, params in self.resource_op_precisions.items()
                    if "precision" in params
                ]
            )
            raise ValueError(
                f"{op_type.__name__} is not a configurable operator. "
                f"Configurable operators are: {', '.join(configurable_ops)}"
            )

        if "precision" not in self.resource_op_precisions[op_type]:
            raise ValueError(f"Setting precision for {op_type.__name__} is not supported.")

        self.resource_op_precisions[op_type]["precision"] = precision

    def set_single_qubit_rot_precision(self, precision: float) -> None:
        r"""Sets the synthesis precision for all single-qubit rotation gates.

        This is a convenience method to update the synthesis precision tolerance
        for all standard single-qubit rotation gates (and their
        controlled versions) at once. The synthesis precision dictates the precision
        for compiling rotation gates into a discrete gate set, which in turn
        affects the number of gates required.

        This method updates the ``precision`` value for the following operators:
        :class:`~.pennylane.estimator.RX`, :class:`~.pennylane.estimator.RY`,
        :class:`~.pennylane.estimator.RZ`, :class:`~.pennylane.estimator.CRX`,
        :class:`~.pennylane.estimator.CRY`, :class:`~.pennylane.estimator.CRZ`.

        Args:
            precision (float): The desired synthesis precision tolerance. A smaller
                value corresponds to a higher precision compilation, which may
                increase the required gate counts. Must be greater than ``0``.

        Raises:
            ValueError: If ``precision`` is a negative value.

        **Example**

        .. code-block:: python

            import pennylane.estimator as qre

            config = qre.ResourceConfig()
            rot_ops = [qre.RX, qre.RY, qre.RZ, qre.CRX, qre.CRY, qre.CRZ]
            print([config.resource_op_precisions[op]['precision'] for op in rot_ops])

            config.set_single_qubit_rot_precision(1e-5)
            print([config.resource_op_precisions[op]['precision'] for op in rot_ops])

        .. code-block:: pycon

            [1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09]
            [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05]
        """
        if precision < 0:
            raise ValueError(f"Precision must be a non-negative value, but got {precision}.")

        for op in [RX, RY, RZ, CRX, CRY, CRZ]:
            self.resource_op_precisions[op]["precision"] = precision

    def set_decomp(
        self,
        op_type: type[ResourceOperator],
        decomp_func: Callable,
        decomp_type: DecompositionType | None = DecompositionType.BASE,
    ) -> None:
        """Sets a custom function to override the default resource decomposition.

        Args:
            op_type (type[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): the operator class whose decomposition is being overriden.
            decomp_func (Callable): the new resource decomposition function to be set as default.
            decomp_type (None | DecompositionType): the decomposition type to override. Options are
                ``"adj"``, ``"pow"``, ``"ctrl"``, and ``"base"``. Default is ``"base"``.

        Raises:
            ValueError: If ``decomp_type`` is not a valid decomposition type.

        .. note::

            The new decomposition function ``decomp_func`` should have the same signature as the one it replaces.
            Specifically, the signature should match the :code:`resource_keys` of the base resource
            operator class being overriden.

        **Example**

        .. code-block:: python

            import pennylane.estimator as qre

            def custom_res_decomp(**kwargs):
                h = qre.resource_rep(qre.Hadamard)
                s = qre.resource_rep(qre.S)
                return [qre.GateCount(h, 1), qre.GateCount(s, 2)]

        .. code-block:: pycon

            >>> print(qre.estimate(qre.X(), gate_set={"Hadamard", "Z", "S"}))
            --- Resources: ---
             Total wires: 1
                algorithmic wires: 1
                allocated wires: 0
                 zero state: 0
                 any state: 0
             Total gates : 4
              'S': 2,
              'Hadamard': 2
            >>> config = qre.ResourceConfig()
            >>> config.set_decomp(qre.X, custom_res_decomp)
            >>> print(qre.estimate(qre.X(), gate_set={"Hadamard", "Z", "S"}, config=config))
            --- Resources: ---
             Total wires: 1
                algorithmic wires: 1
                allocated wires: 0
                 zero state: 0
                 any state: 0
             Total gates : 3
              'S': 2,
              'Hadamard': 1
        """
        if decomp_type is None:
            decomp_type = DecompositionType("base")
        else:
            decomp_type = DecompositionType(decomp_type)

        if decomp_type == DecompositionType.ADJOINT:
            self._adj_custom_decomps[op_type] = decomp_func
        elif decomp_type == DecompositionType.CONTROLLED:
            self._ctrl_custom_decomps[op_type] = decomp_func
        elif decomp_type == DecompositionType.POW:
            self._pow_custom_decomps[op_type] = decomp_func
        elif decomp_type is None or decomp_type == DecompositionType.BASE:
            self._custom_decomps[op_type] = decomp_func
