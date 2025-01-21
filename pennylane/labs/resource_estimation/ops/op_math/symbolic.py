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
from typing import Dict

import pennylane.labs.resource_estimation as re
from pennylane import math
from pennylane.labs.resource_estimation.resource_container import _combine_dict, _scale_dict
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.exp import Exp
from pennylane.ops.op_math.pow import PowOperation

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments,too-many-positional-arguments


class ResourceAdjoint(AdjointOperation, re.ResourceOperator):
    """Resource class for the Adjoint symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.adjoint_resource_decomp(**base_params)
        except re.ResourcesNotDefined:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params)
                gate_types[rep] = count

            return gate_types

    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params()}

    @classmethod
    def resource_rep(cls, base_class, base_params) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"base_class": base_class, "base_params": base_params})

    @staticmethod
    def adjoint_resource_decomp(base_class, base_params) -> Dict[re.CompressedResourceOp, int]:
        return {base_class.resource_rep(**base_params): 1}

    @staticmethod
    def tracking_name(base_class, base_params) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"Adjoint({base_name})"


class ResourceControlled(ControlledOp, re.ResourceOperator):
    """Resource class for the Controlled symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **base_params
            )
        except re.ResourcesNotDefined:
            pass

        gate_types = defaultdict(int)

        if num_ctrl_values == 0:
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params, num_ctrl_wires, 0, num_work_wires)
                gate_types[rep] = count

            return gate_types

        no_control = cls.resource_rep(base_class, base_params, num_ctrl_wires, 0, num_work_wires)
        x = re.ResourceX.resource_rep()
        gate_types[no_control] = 1
        gate_types[x] = 2 * num_ctrl_values

        return gate_types

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
            "num_ctrl_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "num_work_wires": num_work_wires,
            },
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
    ) -> Dict[re.CompressedResourceOp, int]:
        return {
            cls.resource_rep(
                base_class,
                base_params,
                outer_num_ctrl_wires + num_ctrl_wires,
                outer_num_ctrl_values + num_ctrl_values,
                outer_num_work_wires + num_work_wires,
            ): 1
        }

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires):
        base_name = base_class.tracking_name(**base_params)
        return f"C({base_name},{num_ctrl_wires},{num_ctrl_values},{num_work_wires})"


class ResourcePow(PowOperation, re.ResourceOperator):
    """Resource class for the Pow symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, z, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if z == 0:
            return {re.ResourceIdentity.resource_rep(): 1}

        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params, z)
                gate_types[rep] = count

            return gate_types
        except re.ResourcesNotDefined:
            pass

        return {base_class.resource_rep(**base_params): z}

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "z": self.z,
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, z) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "base_params": base_params, "z": z}
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, base_params, z
    ) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(base_class, base_params, z0 * z): 1}

    @staticmethod
    def tracking_name(base_class, base_params, z) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"Pow({base_name}, {z})"


class ResourceExp(Exp, re.ResourceOperator):
    """Resource class for Exp"""

    @staticmethod
    def _resource_decomp(base_class, base_params, base_pauli_rep, coeff, num_steps, **kwargs):

        # Custom exponential operator resources:
        if issubclass(base_class, re.ResourceOperator):
            try:
                return base_class.exp_resource_decomp(coeff, num_steps, **base_params)
            except re.ResourcesNotDefined:
                pass

        if base_pauli_rep and math.real(coeff) == 0:
            scalar = num_steps or 1  # 1st-order Trotter-Suzuki with 'num_steps' trotter steps:
            return _scale_dict(
                _resources_from_pauli_sentence(base_pauli_rep), scalar=scalar, in_place=True
            )

        raise re.ResourcesNotDefined

    def resource_params(self):
        return _extract_exp_params(self.base, self.scalar, self.num_steps)

    @classmethod
    def resource_rep(cls, base_class, base_params, base_pauli_rep, coeff, num_steps):
        name = cls.tracking_name(base_class, base_params, base_pauli_rep, coeff, num_steps)
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "base_pauli_rep": base_pauli_rep,
                "coeff": coeff,
                "num_steps": num_steps,
            },
            name=name,
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, base_params, base_pauli_rep, coeff, num_steps
    ) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(base_class, base_params, base_pauli_rep, z0 * coeff, num_steps): 1}

    @staticmethod
    def tracking_name(base_class, base_params, base_pauli_rep, coeff, num_steps):
        base_name = (
            base_class.tracking_name(**base_params)
            if issubclass(base_class, re.ResourceOperator)
            else base_class.__name__
        )

        return f"Exp({base_name}, {coeff}, num_steps={num_steps})".replace("Resource", "")


def _extract_exp_params(base_op, scalar, num_steps):
    pauli_rep = base_op.pauli_rep
    isinstance_resource_op = isinstance(base_op, re.ResourceOperator)

    if (not isinstance_resource_op) and (pauli_rep is None):
        raise ValueError(
            f"Cannot obtain resources for the exponential of {base_op}, if it is not a ResourceOperator and it doesn't have a Pauli decomposition."
        )

    base_class = type(base_op)
    base_params = base_op.resource_params() if isinstance_resource_op else {}

    return {
        "base_class": base_class,
        "base_params": base_params,
        "base_pauli_rep": pauli_rep,
        "coeff": scalar,
        "num_steps": num_steps,
    }


def _resources_from_pauli_sentence(pauli_sentence):
    gate_types = defaultdict(int)

    for pauli_word in iter(pauli_sentence.keys()):
        pauli_string = "".join((str(v) for v in pauli_word.values()))
        pauli_rot_gate = re.ResourcePauliRot.resource_rep(pauli_string)
        gate_types[pauli_rot_gate] = 1

    return gate_types
