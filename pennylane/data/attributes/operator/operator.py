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
from typing import Generic, TypeVar

import numpy as np

from pennylane import ops as qops
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
    def supported_ops(cls) -> frozenset[type[Operator]]:
        """Set of supported operators."""
        return frozenset(
            (
                # pennylane/ops/qubit/arithmetic_qml.py
                qops.QubitCarry,
                qops.QubitSum,
                # pennylane/ops/op_math/linear_combination.py
                qops.LinearCombination,
                # pennylane/ops/op_math - prod.py, s_prod.py, sum.py
                qops.Prod,
                qops.SProd,
                qops.Sum,
                # pennylane/ops/qubit/matrix_ops.py
                qops.QubitUnitary,
                qops.DiagonalQubitUnitary,
                # pennylane/ops/qubit/non_parametric_qml.py
                qops.Hadamard,
                qops.PauliX,
                qops.PauliY,
                qops.PauliZ,
                qops.X,
                qops.Y,
                qops.Z,
                qops.T,
                qops.S,
                qops.SX,
                qops.CNOT,
                qops.CH,
                qops.SWAP,
                qops.ECR,
                qops.SISWAP,
                qops.CSWAP,
                qops.CCZ,
                qops.Toffoli,
                qops.WireCut,
                # pennylane/ops/qubit/observables.py
                qops.Hermitian,
                qops.Projector,
                # pennylane/ops/qubit/parametric_ops_multi_qubit.py
                qops.MultiRZ,
                qops.IsingXX,
                qops.IsingYY,
                qops.IsingZZ,
                qops.IsingXY,
                qops.PSWAP,
                qops.CPhaseShift00,
                qops.CPhaseShift01,
                qops.CPhaseShift10,
                # pennylane/ops/qubit/parametric_ops_single_qubit.py
                qops.RX,
                qops.RY,
                qops.RZ,
                qops.PhaseShift,
                qops.Rot,
                qops.U1,
                qops.U2,
                qops.U3,
                # pennylane/ops/qubit/qchem_ops.py
                qops.SingleExcitation,
                qops.SingleExcitationMinus,
                qops.SingleExcitationPlus,
                qops.DoubleExcitation,
                qops.DoubleExcitationMinus,
                qops.DoubleExcitationPlus,
                qops.OrbitalRotation,
                qops.FermionicSWAP,
                # pennylane/ops/special_unitary.py
                qops.SpecialUnitary,
                # pennylane/ops/state_preparation.py
                qops.BasisState,
                qops.StatePrep,
                qops.QubitDensityMatrix,
                # pennylane/ops/qutrit/matrix_obs.py
                qops.QutritUnitary,
                # pennylane/ops/qutrit/non_parametric_ops.py
                qops.TShift,
                qops.TClock,
                qops.TAdd,
                qops.TSWAP,
                # pennylane/ops/qutrit/observables.py
                qops.THermitian,
                # pennylane/ops/channel.py
                qops.AmplitudeDamping,
                qops.GeneralizedAmplitudeDamping,
                qops.PhaseDamping,
                qops.DepolarizingChannel,
                qops.BitFlip,
                qops.ResetError,
                qops.PauliError,
                qops.PhaseFlip,
                qops.ThermalRelaxationError,
                # pennylane/ops/cv.py
                qops.Rotation,
                qops.Squeezing,
                qops.Displacement,
                qops.Beamsplitter,
                qops.TwoModeSqueezing,
                qops.QuadraticPhase,
                qops.ControlledAddition,
                qops.ControlledPhase,
                qops.Kerr,
                qops.CrossKerr,
                qops.InterferometerUnitary,
                qops.CoherentState,
                qops.SqueezedState,
                qops.DisplacedSqueezedState,
                qops.ThermalState,
                qops.GaussianState,
                qops.FockState,
                qops.FockStateVector,
                qops.FockDensityMatrix,
                qops.CatState,
                qops.NumberOperator,
                qops.TensorN,
                qops.QuadX,
                qops.QuadP,
                qops.QuadOperator,
                qops.PolyXP,
                qops.FockStateProjector,
                # pennylane/ops/identity.py
                qops.Identity,
                # pennylane/ops/op_math/controlled_ops.py
                qops.ControlledQubitUnitary,
                qops.ControlledPhaseShift,
                qops.CRX,
                qops.CRY,
                qops.CRZ,
                qops.CRot,
                qops.CZ,
                qops.CY,
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
            if isinstance(op, (qops.Prod, qops.SProd, qops.Sum)):
                op = op.simplify()
            if type(op) not in self.supported_ops():
                raise TypeError(
                    f"Serialization of operator type '{type(op).__name__}' is not supported."
                )

            if isinstance(op, qops.LinearCombination):
                coeffs, ops = op.terms()
                ham_grp = self._ops_to_hdf5(bind, op_key, ops)
                ham_grp["hamiltonian_coeffs"] = coeffs
                op_wire_labels.append("null")
            elif isinstance(op, (qops.Prod, qops.Sum)):
                self._ops_to_hdf5(bind, op_key, op.operands)
                op_wire_labels.append("null")
            elif isinstance(op, qops.SProd):
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
                if op_cls is qops.LinearCombination:
                    ops.append(
                        qops.LinearCombination(
                            coeffs=list(bind[op_key]["hamiltonian_coeffs"]),
                            observables=self._hdf5_to_ops(bind[op_key]),
                        )
                    )
                elif op_cls in (qops.Prod, qops.Sum):
                    ops.append(op_cls(*self._hdf5_to_ops(bind[op_key])))
                elif op_cls is qops.SProd:
                    ops.append(
                        qops.s_prod(
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
    def _supported_ops_dict(cls) -> dict[str, type[Operator]]:
        """Returns a dict mapping ``Operator`` subclass names to the class."""
        ops_dict = {op.__name__: op for op in cls.supported_ops()}
        ops_dict["Hamiltonian"] = qops.LinearCombination
        ops_dict["Tensor"] = qops.Prod
        return ops_dict
