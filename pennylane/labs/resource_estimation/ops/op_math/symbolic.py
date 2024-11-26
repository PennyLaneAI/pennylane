# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for symbolic operations."""
from collections import defaultdict

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane import math
from pennylane.labs.resource_estimation.resource_container import _combine_dict, _scale_dict
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.exp import Exp
from pennylane.ops.op_math.pow import PowOperation
from pennylane.ops.op_math.sprod import SProd

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments


class ResourceAdjoint(AdjointOperation, re.ResourceOperator):
    """Resource class for Adjoint"""

    @staticmethod
    def _resource_decomp(base_class, base_params, **kwargs):
        try:
            return base_class.adjoint_resource_decomp(**base_params)
        except re.ResourcesNotDefined:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params)
            for gate, count in decomp.items():
                resources = gate.op_type.adjoint_resource_decomp(**gate.params)
                _scale_dict(resources, count, in_place=True)
                _combine_dict(gate_types, resources, in_place=True)

            return gate_types

    def resource_params(self):
        return {"base_class": type(self.base), "base_params": self.base.resource_params()}

    @classmethod
    def resource_rep(cls, base_class, base_params, **kwargs):
        name = f"Adjoint({base_class.__name__})".replace("Resource", "")
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "base_params": base_params}, name=name
        )

    @staticmethod
    def adjoint_resource_decomp(base_class, base_params, **kwargs):
        return base_class._resource_decomp(**base_params)


class ResourceControlled(ControlledOp, re.ResourceOperator):
    """Resource class for Controlled"""

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ):
        try:
            return base_class.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **base_params
            )
        except re.ResourcesNotDefined:
            pass

        gate_types = defaultdict(int)
        decomp = base_class.resources(**base_params)
        for gate, count in decomp.items():
            resources = gate.op_type.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **gate.params
            )
            _scale_dict(resources, count, in_place=True)
            _combine_dict(gate_types, resources, in_place=True)

            return gate_types

    def resource_params(self):
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
            "num_ctrl_values": len([val for val in self.control_values if val]),
            "num_work_wires": len(self.work_wires),
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ):
        name = f"Controlled({base_class.__name__}, wires={num_ctrl_wires})".replace("Resource", "")
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "num_work_wires": num_work_wires,
            },
            name=name,
        )

    @classmethod
    def controlled_resource_decomp(
        cls,
        outer_num_ctrl_wires,
        outer_num_ctrl_values,
        outer_num_work_wires,
        base_class,
        base_params,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ):
        return cls._resource_decomp(
            base_class,
            base_params,
            outer_num_ctrl_wires + num_ctrl_wires,
            outer_num_ctrl_values + num_ctrl_values,
            outer_num_work_wires + num_work_wires,
        )


class ResourcePow(PowOperation, re.ResourceOperator):
    """Resource class for Pow"""

    @staticmethod
    def _resource_decomp(base_class, z, base_params, **kwargs):
        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            return _scale_dict(base_class.resources(**base_params), z)
        except re.ResourcesNotDefined:
            pass

        return {base_class.resource_rep(): z}

    def resource_params(self):
        return {
            "base_class": type(self.base),
            "z": self.z,
            "base_params": self.base.resource_params(),
        }

    @classmethod
    def resource_rep(cls, base_class, z, base_params, **kwargs):
        name = f"{base_class.__name__}**{z}".replace("Resource", "")
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "z": z, "base_params": base_params}, name=name
        )

    @classmethod
    def pow_resource_decomp(cls, z0, base_class, z, base_params, **kwargs):
        return cls._resource_decomp(base_class, z0 * z, base_params)


class ResourceExp(Exp, re.ResourceOperator):
    """Resource class for Exp"""

    @staticmethod
    def _resource_decomp(base_class, coeff, num_steps, **kwargs):

        while isinstance(base_class, SProd):
            coeff *= base_class.scalar
            base_class = base_class.base

        # Custom exponential operator resources:
        if isinstance(base_class, re.ResourceOperator):
            try:
                return base_class.exp_resource_decomp(coeff, num_steps, **kwargs)
            except re.ResourcesNotDefined:
                pass

        # PauliRot resource decomp:
        if (pauli_sentence := base_class.pauli_rep) and math.real(coeff) == 0:
            if qml.pauli.is_pauli_word(base_class):
                num_wires = len(base_class.wires)
                pauli_word = tuple(pauli_sentence.keys())[0]  # only one term in the sum
                return _resources_from_pauli_word(pauli_word, num_wires)

            scalar = num_steps or 1  # 1st-order Trotter-Suzuki with 'num_steps' trotter steps:
            return _scale_dict(
                _resources_from_pauli_sentence(pauli_sentence), scalar=scalar, in_place=True
            )

        raise re.ResourcesNotDefined

    def resource_params(self):
        return {
            "base_class": self.base,
            "coeff": self.scalar,
            "num_steps": self.num_steps,
        }

    @classmethod
    def resource_rep(cls, base_class, coeff, num_steps, **kwargs):
        name = f"Exp({base_class.__class__.__name__}, {coeff}, num_steps={num_steps})".replace(
            "Resource", ""
        )
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "coeff": coeff, "num_steps": num_steps}, name=name
        )


def _resources_from_pauli_word(pauli_word, num_wires):
    pauli_string = "".join((str(v) for v in pauli_word.values()))

    if len(pauli_string) == 0:
        return {}  # Identity operation has no resources.

    counter = {"X": 0, "Y": 0, "Z": 0}
    for c in pauli_string:
        counter[c] += 1

    num_x = counter["X"]
    num_y = counter["Y"]

    s = re.CompressedResourceOp(re.ResourceS, {})
    h = re.CompressedResourceOp(re.ResourceHadamard, {})
    rz = re.CompressedResourceOp(re.ResourceRZ, {})
    cnot = re.CompressedResourceOp(re.ResourceCNOT, {})

    gate_types = {}
    gate_types[rz] = 1
    gate_types[s] = 2 * num_y
    gate_types[h] = 2 * (num_x + num_y)
    gate_types[cnot] = 2 * (num_wires - 1)

    return gate_types


def _resources_from_pauli_sentence(pauli_sentence):
    gate_types = defaultdict(int)
    rx = re.CompressedResourceOp(re.ResourceRX, {})
    ry = re.CompressedResourceOp(re.ResourceRY, {})
    rz = re.CompressedResourceOp(re.ResourceRZ, {})

    for pauli_word in iter(pauli_sentence.keys()):
        num_wires = len(pauli_word.wires)

        if num_wires == 1:
            pauli_string = "".join((str(v) for v in pauli_word.values()))
            op_type = {"Z": rz, "X": rx, "Y": ry}[pauli_string]
            gate_types[op_type] += 1

            continue

        pw_gates = _resources_from_pauli_word(pauli_word, num_wires)
        _ = _combine_dict(gate_types, pw_gates, in_place=True)

    return gate_types
