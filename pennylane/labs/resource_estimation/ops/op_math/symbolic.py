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

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_container import _combine_dict, _scale_dict
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.pow import PowOperation

# pylint: disable=too-many-ancestors,arguments-differ


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


class ResourceControlled(ControlledOp, re.ResourceOperator):
    """Resource class for Controlled"""

    @staticmethod
    def _resource_decomp(base_class, base_params, num_ctrl_wires, **kwargs):
        try:
            return base_class.controlled_resource_decomp(num_ctrl_wires, **base_params)
        except re.ResourcesNotDefined:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params)
            for gate, count in decomp.items():
                resources = gate.op_type.controlled_resource_decomp(**gate.params)
                _scale_dict(resources, count, in_place=True)
                _combine_dict(gate_types, resources, in_place=True)

            return gate_types

    def resource_params(self):
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
            "num_zeros": len([val for val in self.control_values if not val]),
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, num_ctrl_wires, num_zeros, **kwargs):
        name = f"Controlled({base_class.__name__}, wires={num_ctrl_wires})".replace("Resource", "")
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
                "num_zeros": num_zeros,
            },
            name=name,
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
