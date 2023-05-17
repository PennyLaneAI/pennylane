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
"""Contains AttributeType definition for subclasses of ``pennylane.operation.Operator``."""

import json
from functools import lru_cache
from typing import Dict, Generic, MutableMapping, Tuple, Type, TypeVar

import numpy as np

from pennylane import Hamiltonian, ops
from pennylane.data.attributes.array import DatasetArray
from pennylane.data.attributes.list import DatasetList
from pennylane.data.attributes.wires import DatasetWires
from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.typing_util import HDF5Group
from pennylane.operation import Operator, Tensor
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word
import pennylane as qml


@lru_cache(1)
def _get_all_operator_classes() -> Tuple[Type[Operator], ...]:
    """This function returns a tuple of every subclass of
    ``pennylane.operation.Operator``."""
    acc = set()

    def rec(cls):
        for subcls in cls.__subclasses__():
            if subcls not in acc:
                acc.add(subcls)
                rec(subcls)

    rec(Operator)

    return tuple(acc - {Hamiltonian, Tensor})


@lru_cache(1)
def _operator_name_to_class_dict() -> Dict[str, Type[Operator]]:
    """Returns a dictionary mapping the type name of each ``pennylane.operation.Operator``
    class to the class."""

    op_classes = _get_all_operator_classes()

    return {op.__qualname__: op for op in op_classes}


Op = TypeVar("Op", bound=Operator)

_SUPPORTED_OPS = (
    # pennylane/ops/qubit/arithmetic_qml.py
    qml.QubitCarry,
    qml.QubitSum,
    # pennylane/ops/qubit/matrix_qml.py
    qml.QubitUnitary,
    qml.DiagonalQubitUnitary,
    # pennylane/ops/qubit/non_parametric_qml.py
    qml.Hadamard,
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.T,
    qml.S,
    qml.SX,
    qml.CNOT,
    qml.CZ,
    qml.CY,
    qml.CH,
    qml.SWAP,
    qml.ECR,
    qml.SISWAP,
    qml.CSWAP,
    qml.CCZ,
    qml.Toffoli,
    qml.WireCut,
    # pennylane/ops/qubit/observables.py
    qml.Hermitian,
    qml.Projector,
    # pennylane/ops/qubit/parametric_ops_controlled.py
    qml.ControlledPhaseShift,
    qml.CPhaseShift00,
    qml.CPhaseShift01,
    qml.CPhaseShift10,
    qml.CRX,
    qml.CRY,
    qml.CRZ,
    qml.CRot,
    # pennylane/ops/qubit/parametric_ops_multi_qubit.py
    qml.MultiRZ,
    qml.IsingXX,
    qml.IsingYY,
    qml.IsingZZ,
    qml.IsingXY,
    qml.PSWAP,
    # pennylane/ops/qubit/parametric_ops_single_qubit.py
    qml.RX,
    qml.RY,
    qml.RZ,
    qml.PhaseShift,
    qml.Rot,
    qml.U1,
    qml.U2,
    qml.U3,
    # pennylane/ops/qubit/qchem_qml.py
    qml.SingleExcitation,
    qml.SingleExcitationMinus,
    qml.SingleExcitationPlus,
    qml.DoubleExcitation,
    qml.DoubleExcitationMinus,
    qml.DoubleExcitationPlus,
    qml.OrbitalRotation,
    # pennylane/ops/special_unitary.py
    qml.SpecialUnitary,
    # pennylane/ops/state_preparation.py
    qml.BasisState,
    qml.QubitStateVector,
    qml.QubitDensityMatrix,
    # pennylane/ops/qutrit/matrix_qml.py
    qml.QutritUnitary,
    # pennylane/ops/qutrit/non_parametric_qml.py
    qml.TShift,
    qml.TClock,
    qml.TAdd,
    qml.TSWAP,
    # pennylane/ops/qutrit/observables.py
    qml.THermitian,
    # pennylane/ops/channel.py
    qml.AmplitudeDamping,
    qml.GeneralizedAmplitudeDamping,
    qml.PhaseDamping,
    qml.DepolarizingChannel,
    qml.BitFlip,
    qml.ResetError,
    qml.PauliError,
    qml.PhaseFlip,
    qml.ThermalRelaxationError,
    # pennylane/ops/cv.py
    qml.Rotation,
    qml.Squeezing,
    qml.Displacement,
    qml.Beamsplitter,
    qml.TwoModeSqueezing,
    qml.QuadraticPhase,
    qml.ControlledAddition,
    qml.ControlledPhase,
    qml.Kerr,
    qml.CrossKerr,
    qml.InterferometerUnitary,
    qml.CoherentState,
    qml.SqueezedState,
    qml.DisplacedSqueezedState,
    qml.ThermalState,
    qml.GaussianState,
    qml.FockState,
    qml.FockStateVector,
    qml.FockDensityMatrix,
    qml.CatState,
    qml.NumberOperator,
    qml.TensorN,
    qml.X,
    qml.P,
    qml.QuadOperator,
    qml.PolyXP,
    qml.FockStateProjector,
    # pennylane/ops/identity.py
    qml.Identity,
)


class DatasetOperator(Generic[Op], AttributeType[HDF5Group, Op, Op]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "operator"

    def __post_init__(self, value: Op, info):
        """Save the class name of the operator ``value`` into the
        attribute info."""
        super().__post_init__(value, info)
        self.info["operator_class"] = type(value).__qualname__

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Operator], ...]:
        return _SUPPORTED_OPS

    def hdf5_to_value(self, bind: HDF5Group) -> Op:
        info = AttributeInfo(bind.attrs)
        op_cls = _operator_name_to_class_dict()[info["operator_class"]]

        wires = DatasetWires(bind=bind["wires"]).get_value()
        params = np.array(bind["params"])

        return op_cls(*params, wires=wires)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Op) -> HDF5Group:
        bind = bind_parent.create_group(key)

        DatasetWires(value.wires, parent_and_key=(bind, "wires"))
        DatasetArray(value.data, parent_and_key=(bind, "params"))

        return bind


class DatasetTensor(DatasetList[Tensor]):
    type_id = "tensor"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Tensor]]:
        return (Tensor,)

    def hdf5_to_value(self, bind: HDF5Group) -> Tensor:
        return Tensor(*(DatasetOperator(bind=obs_bind).get_value() for obs_bind in bind.values()))

    def __post_init__(self, value: Tensor, info):
        return super().__post_init__(value.obs, info)


class DatasetHamiltonian(AttributeType[HDF5Group, Hamiltonian, Hamiltonian]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "hamiltonian"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def hdf5_to_value(self, bind: HDF5Group) -> Hamiltonian:
        wire_map = {w: i for i, w in enumerate(json.loads(bind["wires"][()]))}

        ops = [string_to_pauli_word(pauli_string, wire_map) for pauli_string in bind["ops"]]
        coeffs = list(bind["coeffs"])

        return Hamiltonian(coeffs, ops)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Hamiltonian) -> HDF5Group:
        bind = bind_parent.create_group(key)

        coeffs, ops = value.terms()
        wire_map = {w: i for i, w in enumerate(value.wires)}

        bind["ops"] = [pauli_word_to_string(op, wire_map) for op in ops]
        bind["coeffs"] = coeffs
        bind["wires"] = json.dumps(list(w for w in value.wires))

        return bind
