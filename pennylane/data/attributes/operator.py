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

from pennylane import Hamiltonian
from pennylane.operation import Tensor
from pennylane.data.attributes.array import DatasetArray
from pennylane.data.attributes.list import DatasetList
from pennylane.data.attributes.wires import DatasetWires
from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.operation import Operator
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word
from pennylane import ops


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
    # pennylane/ops/qubit/arithmetic_ops.py
    ops.QubitCarry,
    ops.QubitSum,
    # pennylane/ops/qubit/matrix_ops.py
    ops.QubitUnitary,
    ops.DiagonalQubitUnitary,
    # pennylane/ops/qubit/non_parametric_ops.py
    ops.Hadamard,
    ops.PauliX,
    ops.PauliY,
    ops.PauliZ,
    ops.T,
    ops.S,
    ops.SX,
    ops.CNOT,
    ops.CZ,
    ops.CY,
    ops.CH,
    ops.SWAP,
    ops.ECR,
    ops.SISWAP,
    ops.CSWAP,
    ops.CCZ,
    ops.Toffoli,
    ops.WireCut,
    # pennylane/ops/qubit/observables.py
    ops.Hermitian,
    ops.Projector,
    # pennylane/ops/qubit/parametric_ops_controlled.py
    ops.ControlledPhaseShift,
    ops.CPhaseShift00,
    ops.CPhaseShift01,
    ops.CPhaseShift10,
    ops.CRX,
    ops.CRY,
    ops.CRZ,
    ops.CRot,
    # pennylane/ops/qubit/parametric_ops_multi_qubit.py
    ops.MultiRZ,
    ops.IsingXX,
    ops.IsingYY,
    ops.IsingZZ,
    ops.IsingXY,
    ops.PSWAP,
    # pennylane/ops/qubit/parametric_ops_single_qubit.py
    ops.RX,
    ops.RY,
    ops.RZ,
    ops.PhaseShift,
    ops.Rot,
    ops.U1,
    ops.U2,
    ops.U3,
    # pennylane/ops/qubit/qchem_ops.py
    ops.SingleExcitation,
    ops.SingleExcitationMinus,
    ops.SingleExcitationPlus,
    ops.DoubleExcitation,
    ops.DoubleExcitationMinus,
    ops.DoubleExcitationPlus,
    ops.OrbitalRotation,
    # pennylane/ops/special_unitary.py
    ops.SpecialUnitary,
    # pennylane/ops/state_preparation.py
    ops.BasisState,
    ops.QubitStateVector,
    ops.QubitDensityMatrix,
    # pennylane/ops/qutrit/matrix_ops.py
    ops.QutritUnitary,
    # pennylane/ops/qutrit/non_parametric_ops.py
    ops.TShift,
    ops.TClock,
    ops.TAdd,
    ops.TSWAP,
    # pennylane/ops/qutrit/observables.py
    ops.THermitian,
    # pennylane/ops/channel.py
    ops.AmplitudeDamping,
    ops.GeneralizedAmplitudeDamping,
    ops.PhaseDamping,
    ops.DepolarizingChannel,
    ops.BitFlip,
    ops.ResetError,
    ops.PauliError,
    ops.PhaseFlip,
    ops.ThermalRelaxationError,
    # pennylane/ops/cv.py
    ops.Rotation,
    ops.Squeezing,
    ops.Displacement,
    ops.Beamsplitter,
    ops.TwoModeSqueezing,
    ops.QuadraticPhase,
    ops.ControlledAddition,
    ops.ControlledPhase,
    ops.Kerr,
    ops.CrossKerr,
    ops.InterferometerUnitary,
    ops.CoherentState,
    ops.SqueezedState,
    ops.DisplacedSqueezedState,
    ops.ThermalState,
    ops.GaussianState,
    ops.FockState,
    ops.FockStateVector,
    ops.FockDensityMatrix,
    ops.CatState,
    ops.NumberOperator,
    ops.TensorN,
    ops.X,
    ops.P,
    ops.QuadOperator,
    ops.PolyXP,
    ops.FockStateProjector,
    # pennylane/ops/identity.py
    ops.Identity,
)


class DatasetOperator(Generic[Op], AttributeType[ZarrGroup, Op, Op]):
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

    def zarr_to_value(self, bind: ZarrGroup) -> Op:
        info = AttributeInfo(bind.attrs)
        op_cls = _operator_name_to_class_dict()[info["operator_class"]]

        wires = DatasetWires(bind=bind["wires"]).get_value()
        params = np.array(bind["params"])

        return op_cls(*params, wires=wires)

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Op) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        DatasetWires(value.wires, parent_and_key=(bind, "wires"))
        DatasetArray(value.data, parent_and_key=(bind, "params"))

        return bind


class DatasetTensor(DatasetList[Tensor]):
    type_id = "tensor"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Tensor]]:
        return (Tensor,)

    def zarr_to_value(self, bind: ZarrGroup) -> Tensor:
        return Tensor(*(DatasetOperator(bind=obs_bind).get_value() for obs_bind in bind.values()))

    def __post_init__(self, value: Tensor, info):
        return super().__post_init__(value.obs, info)


class DatasetHamiltonian(AttributeType[ZarrGroup, Hamiltonian, Hamiltonian]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "hamiltonian"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        wire_map = {w: i for i, w in enumerate(json.loads(bind["wires"][()]))}

        ops = [string_to_pauli_word(pauli_string, wire_map) for pauli_string in bind["ops"]]
        coeffs = list(bind["coeffs"])

        return Hamiltonian(coeffs, ops)

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        coeffs, ops = value.terms()
        wire_map = {w: i for i, w in enumerate(value.wires)}

        bind["ops"] = [pauli_word_to_string(op, wire_map) for op in ops]
        bind["coeffs"] = coeffs
        bind["wires"] = json.dumps(list(w for w in value.wires))

        return bind
