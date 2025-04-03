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
"""Contains DatasetAttribute definition for pennylane operators, and lists
of operators."""

import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Generic, Type, TypeVar

import numpy as np

from pennylane import ops as supported_ops
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager

from ._wires import wires_to_json

Op = TypeVar("Op", bound=Operator)


class DatasetOperator(Generic[Op], DatasetAttribute[HDF5Group, Op, Op]):
    """``DatasetAttribute`` for ``pennylane.operation.Operator`` classes.

    Supports all operator types that meet the following conditions:
        - The ``__init__()`` method matches the signature of ``Operator.__init__``,
            or any additional arguments are optional and do not affect the value of
            the operator
        - The ``data`` and ``wires`` attributes will produce an identical copy of
            operator if passed into the classes' ``__init__()`` method. Generally,
            this means ``__init__()`` do not mutate the ``identifiers`` and ``wires``
            arguments.
        - Hyperparameters are not used or are automatically derived by ``__init__()``.

    """

    type_id = "operator"

    @classmethod
    @lru_cache(1)
    def supported_ops(cls) -> frozenset[Type[Operator]]:
        """Set of supported operators."""
        return frozenset(
            (
                # pennylane/ops/qubit/arithmetic_qml.py
                supported_ops.QubitCarry,
                supported_ops.QubitSum,
                # pennylane/ops/op_math/linear_combination.py
                supported_ops.LinearCombination,
                # pennylane/ops/op_math - prod.py, s_prod.py, sum.py
                supported_ops.Prod,
                supported_ops.SProd,
                supported_ops.Sum,
                # pennylane/ops/qubit/matrix_supported_ops.py
                supported_ops.QubitUnitary,
                supported_ops.DiagonalQubitUnitary,
                # pennylane/ops/qubit/non_parametric_qml.py
                supported_ops.Hadamard,
                supported_ops.PauliX,
                supported_ops.PauliY,
                supported_ops.PauliZ,
                supported_ops.X,
                supported_ops.Y,
                supported_ops.Z,
                supported_ops.T,
                supported_ops.S,
                supported_ops.SX,
                supported_ops.CNOT,
                supported_ops.CH,
                supported_ops.SWAP,
                supported_ops.ECR,
                supported_ops.SISWAP,
                supported_ops.CSWAP,
                supported_ops.CCZ,
                supported_ops.Toffoli,
                supported_ops.WireCut,
                # pennylane/ops/qubit/observables.py
                supported_ops.Hermitian,
                supported_ops.Projector,
                # pennylane/ops/qubit/parametric_ops_multi_qubit.py
                supported_ops.MultiRZ,
                supported_ops.IsingXX,
                supported_ops.IsingYY,
                supported_ops.IsingZZ,
                supported_ops.IsingXY,
                supported_ops.PSWAP,
                supported_ops.CPhaseShift00,
                supported_ops.CPhaseShift01,
                supported_ops.CPhaseShift10,
                # pennylane/ops/qubit/parametric_ops_single_qubit.py
                supported_ops.RX,
                supported_ops.RY,
                supported_ops.RZ,
                supported_ops.PhaseShift,
                supported_ops.Rot,
                supported_ops.U1,
                supported_ops.U2,
                supported_ops.U3,
                # pennylane/ops/qubit/qchem_supported_ops.py
                supported_ops.SingleExcitation,
                supported_ops.SingleExcitationMinus,
                supported_ops.SingleExcitationPlus,
                supported_ops.DoubleExcitation,
                supported_ops.DoubleExcitationMinus,
                supported_ops.DoubleExcitationPlus,
                supported_ops.OrbitalRotation,
                supported_ops.FermionicSWAP,
                # pennylane/ops/special_unitary.py
                supported_ops.SpecialUnitary,
                # pennylane/ops/state_preparation.py
                supported_ops.BasisState,
                supported_ops.StatePrep,
                supported_ops.QubitDensityMatrix,
                # pennylane/ops/qutrit/matrix_obs.py
                supported_ops.QutritUnitary,
                # pennylane/ops/qutrit/non_parametric_supported_ops.py
                supported_ops.TShift,
                supported_ops.TClock,
                supported_ops.TAdd,
                supported_ops.TSWAP,
                # pennylane/ops/qutrit/observables.py
                supported_ops.THermitian,
                # pennylane/ops/channel.py
                supported_ops.AmplitudeDamping,
                supported_ops.GeneralizedAmplitudeDamping,
                supported_ops.PhaseDamping,
                supported_ops.DepolarizingChannel,
                supported_ops.BitFlip,
                supported_ops.ResetError,
                supported_ops.PauliError,
                supported_ops.PhaseFlip,
                supported_ops.ThermalRelaxationError,
                # pennylane/ops/cv.py
                supported_ops.Rotation,
                supported_ops.Squeezing,
                supported_ops.Displacement,
                supported_ops.Beamsplitter,
                supported_ops.TwoModeSqueezing,
                supported_ops.QuadraticPhase,
                supported_ops.ControlledAddition,
                supported_ops.ControlledPhase,
                supported_ops.Kerr,
                supported_ops.CrossKerr,
                supported_ops.InterferometerUnitary,
                supported_ops.CoherentState,
                supported_ops.SqueezedState,
                supported_ops.DisplacedSqueezedState,
                supported_ops.ThermalState,
                supported_ops.GaussianState,
                supported_ops.FockState,
                supported_ops.FockStateVector,
                supported_ops.FockDensityMatrix,
                supported_ops.CatState,
                supported_ops.NumberOperator,
                supported_ops.TensorN,
                supported_ops.QuadX,
                supported_ops.QuadP,
                supported_ops.QuadOperator,
                supported_ops.PolyXP,
                supported_ops.FockStateProjector,
                # pennylane/ops/identity.py
                supported_ops.Identity,
                # pennylane/ops/op_math/controlled_supported_ops.py
                supported_ops.ControlledQubitUnitary,
                supported_ops.ControlledPhaseShift,
                supported_ops.CRX,
                supported_ops.CRY,
                supported_ops.CRZ,
                supported_ops.CRot,
                supported_ops.CZ,
                supported_ops.CY,
            )
        )

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Op) -> HDF5Group:
        return self._ops_to_hdf5(bind_parent, key, [value])

    def hdf5_to_value(self, bind: HDF5Group) -> Op:
        return self._hdf5_to_ops(bind)[0]

    def _ops_to_hdf5(
        self, bind_parent: HDF5Group, key: str, value: Sequence[Operator]
    ) -> HDF5Group:
        """Serialize op sequence ``value``, and create nested sequences for any
        composite ops in ``value``.

        Since operators are commonly used in larger composite operations, we handle
        sequences of operators as the default case. This allows for performant (in
        time and space) serialization of large and nested operator sums, products, etc.
        """
        bind = bind_parent.create_group(key)

        op_wire_labels = []
        op_class_names = []
        for i, op in enumerate(value):
            op_key = f"op_{i}"
            if isinstance(op, (supported_ops.Prod, supported_ops.SProd, supported_ops.Sum)):
                op = op.simplify()
            if type(op) not in self.supported_ops():
                raise TypeError(
                    f"Serialization of operator type '{type(op).__name__}' is not supported."
                )

            if isinstance(op, supported_ops.LinearCombination):
                coeffs, ops = op.terms()
                ham_grp = self._ops_to_hdf5(bind, op_key, ops)
                ham_grp["hamiltonian_coeffs"] = coeffs
                op_wire_labels.append("null")
            elif isinstance(op, (supported_ops.Prod, supported_ops.Sum)):
                self._ops_to_hdf5(bind, op_key, op.operands)
                op_wire_labels.append("null")
            elif isinstance(op, supported_ops.SProd):
                coeffs, ops = op.terms()
                sprod_grp = self._ops_to_hdf5(bind, op_key, ops)
                sprod_grp["sprod_scalar"] = coeffs
                op_wire_labels.append("null")
            else:
                bind[op_key] = op.data if len(op.data) else h5py.Empty("f")
                op_wire_labels.append(wires_to_json(op.wires))

            op_class_names.append(type(op).__name__)

        bind["op_wire_labels"] = op_wire_labels
        bind["op_class_names"] = op_class_names

        return bind

    def _hdf5_to_ops(self, bind: HDF5Group) -> list[Operator]:
        """Load list of serialized ops from ``bind``."""
        ops = []

        names_bind = bind["op_class_names"]
        wires_bind = bind["op_wire_labels"]
        op_class_names = [] if names_bind.shape == (0,) else names_bind.asstr()
        op_wire_labels = [] if wires_bind.shape == (0,) else wires_bind.asstr()

        with QueuingManager.stop_recording():
            for i, op_class_name in enumerate(op_class_names):
                op_key = f"op_{i}"
                op_cls = self._supported_ops_dict()[op_class_name]
                if op_cls is supported_ops.LinearCombination:
                    ops.append(
                        supported_ops.LinearCombination(
                            coeffs=list(bind[op_key]["hamiltonian_coeffs"]),
                            observables=self._hdf5_to_ops(bind[op_key]),
                        )
                    )
                elif op_cls in (supported_ops.Prod, supported_ops.Sum):
                    ops.append(op_cls(*self._hdf5_to_ops(bind[op_key])))
                elif op_cls is supported_ops.SProd:
                    ops.append(
                        supported_ops.s_prod(
                            scalar=bind[op_key]["sprod_scalar"][0],
                            operator=self._hdf5_to_ops(bind[op_key])[0],
                        )
                    )
                else:
                    wire_labels = json.loads(op_wire_labels[i])
                    op_data = bind[op_key]
                    if op_data.shape is not None:
                        params = np.zeros(shape=op_data.shape, dtype=op_data.dtype)
                        op_data.read_direct(params)
                        ops.append(op_cls(*params, wires=wire_labels))
                    else:
                        ops.append(op_cls(wires=wire_labels))

        return ops

    @classmethod
    @lru_cache(1)
    def _supported_ops_dict(cls) -> dict[str, Type[Operator]]:
        """Returns a dict mapping ``Operator`` subclass names to the class."""
        ops_dict = {op.__name__: op for op in cls.supported_ops()}
        ops_dict["Hamiltonian"] = supported_ops.LinearCombination
        ops_dict["Tensor"] = supported_ops.Prod
        return ops_dict
