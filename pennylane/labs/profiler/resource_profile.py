# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This file contains the base classes for the profiler feature
"""
from collections import defaultdict
from pennylane.estimator.resource_operator import CompressedResourceOp


class ProfileNode:
    """Represents a node in the tree structure for the resource profile."""
    
    def __init__(
        self,
        cmpr_op: CompressedResourceOp | None = None,
        scalar: int = 1,
        gate_data: dict | None = None,
        children: list["ProfileNode"] | None = None
    ):
        self.cmpr_op = cmpr_op
        self.scalar = scalar
        self.children = children or []  # a list of ProfileNodes
        self.gate_data = gate_data or defaultdict(int)


    @classmethod
    def default_group_func(cls, child_nodes: list["ProfileNode"]):  # Returns a dict[str, tuple(scalar, gate_counts, child_nodes)]
        if child_nodes is None or len(child_nodes) == 0:
            return 
        
        grouped_data = {}
        for node in child_nodes:
            if node.cmpr_op.name not in grouped_data:
                child_nodes = []
                child_nodes.extend(node.children)  # this makes a deep copy 
                grouped_data[node.cmpr_op.name] = (node.scalar, node.gate_data, child_nodes)

            else:
                group_scalar, group_gate_data, grouped_nodes = grouped_data[node.cmpr_op.name]
                grouped_nodes.extend(node.children)

                grouped_data[node.cmpr_op.name] = (
                    group_scalar + node.scalar,
                    add_dicts(group_gate_data, node.gate_data, out=True),
                    grouped_nodes,
                )
        return grouped_data

    @classmethod
    def default_cost_func(cls, gate_data):
        val = 0
        for cmpr_rep, counts in gate_data.items():
            if cmpr_rep.name == "T":
                val += counts

        return val

    @classmethod
    def set_cost_func(cls, gate_names: str | list[str]):

        def custom_cost_func(gate_data):
            val = 0
            if isinstance(gate_names, str):
                gate_names = [gate_names]

            for cmpr_rep, counts in gate_data.items():
                if cmpr_rep.name in gate_names:
                    val += counts

            return val
        
        return custom_cost_func


def export_flame_graph_data(root_node: ProfileNode, group_func = None, cost_func = None):
    if group_func is None:
        group_func = ProfileNode.default_group_func

    if cost_func is None: 
        cost_func = ProfileNode.default_cost_func

    child_nodes = root_node.children
    gate_counts = root_node.gate_data

    export_data = (
        ["circuit"],               # ids,
        ["circuit"],               # names,
        [cost_func(gate_counts)],  # values,
        [""],                      # parents,
    )
    grouped_data = group_func(child_nodes)
    _recurrsive_export_flame_graph(
        group_func, cost_func, grouped_data, "circuit", export_data,
    )
    return export_data


def _recurrsive_export_flame_graph(group_func, cost_func, grouped_data, parent_id, export_data):
    if grouped_data is None:
        return
    
    for name, data in grouped_data.items():
        scalar, gate_data, child_nodes = data
        local_name = name if scalar == 1 else name + f" x{scalar}"
        local_id = parent_id + " - " + local_name
        local_val = cost_func(gate_data)

        (ids, names, values, parents) = export_data
        ids.append(local_id)
        names.append(local_name)
        values.append(local_val)
        parents.append(parent_id)

        sub_grouped_data = group_func(child_nodes)
        _recurrsive_export_flame_graph(
            group_func, cost_func, sub_grouped_data, local_id, export_data,
        )
    return


def add_dicts(base_dict: defaultdict, other_dict: defaultdict, out=False):
    """Add together two defaultdict with integer values"""
    if out:
        all_keys = set(base_dict.keys()) | set(other_dict.keys())
        return defaultdict(int, {k: base_dict[k] + other_dict[k] for k in all_keys})

    for key, val in other_dict.items(): base_dict[key] += val
    return base_dict


def mul_dict(base_dict: defaultdict, scalar: int):
    """Multiply a defaultdict with integer values by an integer"""
    return defaultdict(int, {key: base_dict[key] * scalar for key in base_dict})
