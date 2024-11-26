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
from pennylane.labs.resource_estimation.resource_container import _combine_dict, _scale_dict
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.pow import PowOperation

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments


class ResourceAdjoint(AdjointOperation, re.ResourceOperator):
    """Resource class for Adjoint"""

    @staticmethod
    def _resource_decomp(base_class, base_params, **kwargs) -> Dict[re.CompressedResourceOp, int]:
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

    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params()}

    @classmethod
    def resource_rep(cls, base_class, base_params, **kwargs) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "base_params": base_params}
        )

    @staticmethod
    def adjoint_resource_decomp(
        base_class, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        return base_class._resource_decomp(**base_params)

    @staticmethod
    def tracking_name(base_class, base_params) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"Adjoint({base_name})"


class ResourceControlled(ControlledOp, re.ResourceOperator):
    """Resource class for Controlled"""

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
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

    def resource_params(self) -> dict:
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
        return cls._resource_decomp(
            base_class,
            base_params,
            outer_num_ctrl_wires + num_ctrl_wires,
            outer_num_ctrl_values + num_ctrl_values,
            outer_num_work_wires + num_work_wires,
        )

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires):
        base_name = base_class.tracking_name(**base_params)
        return f"C({base_name},{num_ctrl_wires},{num_ctrl_values},{num_work_wires})"


class ResourcePow(PowOperation, re.ResourceOperator):
    """Resource class for Pow"""

    @staticmethod
    def _resource_decomp(
        base_class, z, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            return _scale_dict(base_class.resources(**base_params), z)
        except re.ResourcesNotDefined:
            pass

        return {base_class.resource_rep(): z}

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "z": self.z,
            "base_params": self.base.resource_params(),
        }

    @classmethod
    def resource_rep(cls, base_class, z, base_params, **kwargs) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "z": z, "base_params": base_params}
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, z, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        return cls._resource_decomp(base_class, z0 * z, base_params)

    @staticmethod
    def tracking_name(base_class, z, base_params) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"({base_name})**{z}"
