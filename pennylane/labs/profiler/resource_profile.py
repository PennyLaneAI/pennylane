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
from decimal import Decimal
from typing import Callable

from pennylane.labs.estimator_beta import CompressedResourceOp, T


def _get_decimal_str(num):
    if num < 999:
        return str(num)
    return f"{Decimal(num):.3E}"


class ProfileNode:
    r"""A node in the tree (call-graph) representation of a resource profile.

    Each :class:`ProfileNode` stores a single (compressed) resource operator together
    with the aggregated gate counts produced when that operator is recursively decomposed
    down to a target gate set. The children of a node are the :class:`ProfileNode`
    instances obtained from decomposing the operator one level, which allows the profile
    to be traversed as a hierarchical call graph (for example, to render a flame graph).

    Args:
        cmpr_op (CompressedResourceOp | None): the compressed resource operator represented
            by this node. The root node of a profile has no associated operator and uses the
            default value of ``None``.
        scalar (int): the number of times ``cmpr_op`` appears at this level of the
            decomposition (the *local* multiplicity). Defaults to ``1``.
        gate_data (dict | None): a mapping from :class:`~.estimator.CompressedResourceOp`
            to the integer number of times each gate-set operator occurs in the subtree
            rooted at this node. If ``None``, an empty ``defaultdict(int)`` is used.
        children (list[ProfileNode] | None): the child nodes obtained by decomposing
            ``cmpr_op`` one level. If ``None``, an empty list is used.

    **Example**

    ``profile``

    >>> import pennylane.labs.estimator_beta as qre
    >>> from pennylane.labs.profiler import profile
    ... def circuit():
    ...     for w in range(5):
    ...         qre.Hadamard()
    ...         qre.RZ(1e-9)
    ...
    ...     qre.QPE(qre.RX(precision=1e-3), 4)
    ...     qre.QFT(4)
    >>>
    >>> gate_set = {"T", "Hadamard", "CNOT"}
    >>> res_profile, resources = profile(circuit, gate_set)()
    >>> print(resources)
    --- Resources: ---
     Total wires: 5
       algorithmic wires: 5
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 2.041E+3
       'T': 1.972E+3,
       'CNOT': 44,
       'Hadamard': 25
    """

    def __init__(
        self,
        cmpr_op: CompressedResourceOp | None = None,
        scalar: int = 1,
        gate_data: dict | None = None,
        children: list["ProfileNode"] | None = None,
    ):
        self.cmpr_op = cmpr_op
        self.scalar = scalar
        self.children = children or []  # a list of ProfileNodes
        self.gate_data = gate_data or defaultdict(int)

    @classmethod
    def group_by_name(
        cls, child_nodes: list["ProfileNode"]
    ):  # Returns a dict[str, tuple(scalar, gate_counts, child_nodes)]
        r"""Group a list of child nodes by operator name, merging duplicates.

        Nodes that share the same operator name are merged into a single entry by summing
        their scalars (multiplicities), adding their gate-count dictionaries together, and
        concatenating their children. This is the default grouping used when exporting a
        profile to flame-graph data so that repeated operators are collapsed into one block.

        Args:
            child_nodes (list[ProfileNode]): the nodes to group together.

        Returns:
            dict[str, tuple[int, defaultdict, list[ProfileNode]]] | None: a mapping from
            operator name to a ``(scalar, gate_data, children)`` tuple, where ``scalar`` is
            the summed multiplicity, ``gate_data`` is the combined gate-count dictionary and
            ``children`` are the concatenated child nodes. Returns ``None`` when
            ``child_nodes`` is ``None`` or empty.

        **Example**

        >>> from pennylane.labs.profiler.resource_profile import ProfileNode
        >>> ProfileNode.default_group_func([]) is None
        True
        """
        if child_nodes is None or len(child_nodes) == 0:
            return

        grouped_data = {}
        for node in child_nodes:
            name = node.cmpr_op.name
            scalar = node.gate_data[node.cmpr_op] if node.children == [] else node.scalar
            sub_child_nodes = [
                ProfileNode(c.cmpr_op, scalar * c.scalar, c.gate_data, c.children)
                for c in node.children
            ]

            if name not in grouped_data:
                grouped_data[name] = (scalar, node.gate_data, sub_child_nodes)

            else:
                group_scalar, group_gate_data, grouped_child_nodes = grouped_data[name]
                grouped_child_nodes.extend(sub_child_nodes)
                grouped_data[name] = (
                    group_scalar + node.scalar,
                    add_dicts(group_gate_data, node.gate_data, out=True),
                    grouped_child_nodes,
                )

        return grouped_data

    @classmethod
    def group_by_type(
        cls, child_nodes: list["ProfileNode"]
    ):  # Returns a dict[str, tuple(scalar, gate_counts, child_nodes)]
        r"""Group a list of child nodes by operator name, merging duplicates.

        Nodes that share the same operator name are merged into a single entry by summing
        their scalars (multiplicities), adding their gate-count dictionaries together, and
        concatenating their children. This is the default grouping used when exporting a
        profile to flame-graph data so that repeated operators are collapsed into one block.

        Args:
            child_nodes (list[ProfileNode]): the nodes to group together.

        Returns:
            dict[str, tuple[int, defaultdict, list[ProfileNode]]] | None: a mapping from
            operator name to a ``(scalar, gate_data, children)`` tuple, where ``scalar`` is
            the summed multiplicity, ``gate_data`` is the combined gate-count dictionary and
            ``children`` are the concatenated child nodes. Returns ``None`` when
            ``child_nodes`` is ``None`` or empty.

        **Example**

        >>> from pennylane.labs.profiler.resource_profile import ProfileNode
        >>> ProfileNode.default_group_func([]) is None
        True
        """
        if child_nodes is None or len(child_nodes) == 0:
            return

        grouped_data = {}
        for node in child_nodes:
            scalar = node.gate_data[node.cmpr_op] if node.children == [] else node.scalar
            sub_child_nodes = [
                ProfileNode(c.cmpr_op, scalar * c.scalar, c.gate_data, c.children)
                for c in node.children
            ]

            if node.cmpr_op not in grouped_data:
                grouped_data[node.cmpr_op] = (scalar, node.gate_data, sub_child_nodes)

            else:
                group_scalar, group_gate_data, grouped_child_nodes = grouped_data[node.cmpr_op]
                grouped_child_nodes.extend(sub_child_nodes)
                grouped_data[node.cmpr_op] = (
                    group_scalar + node.scalar,
                    add_dicts(group_gate_data, node.gate_data, out=True),
                    grouped_child_nodes,
                )

        return grouped_data

    @classmethod
    def default_cost_func(cls, gate_data: dict) -> int:
        r"""Compute the default cost of a gate-count dictionary.

        The default cost is the total number of ``T`` gates, which is a common figure of
        merit for fault-tolerant resource estimation.

        Args:
            gate_data (dict): a mapping from :class:`~.estimator.CompressedResourceOp` to
                the integer number of times each operator occurs.

        Returns:
            int: the number of ``T`` gates contained in ``gate_data``.

        **Example**

        >>> import pennylane.estimator as qre
        >>> from pennylane.labs.profiler.resource_profile import ProfileNode
        >>> gate_data = {
        ...     qre.X.resource_rep(): 1,
        ...     qre.T.resource_rep(): 5,
        ...     qre.Hadamard.resource_rep(): 2
        ... }
        >>> ProfileNode.default_cost_func(gate_data)
        5
        """
        return gate_data.get(T.resource_rep(), 0)

    @classmethod
    def set_cost_func(cls, gate_names: str | list[str]) -> Callable:
        r"""Create a cost function that counts a custom set of gates.

        This is a factory method that returns a cost function summing the occurrences of the
        operators named in ``gate_names``. The returned callable has the same signature as
        :meth:`default_cost_func` and can be passed to :func:`export_flame_graph_data`.

        Args:
            gate_names (str | list[str]): the name (or names) of the operators whose counts
                should be summed by the returned cost function.

        Returns:
            Callable[[dict], int]: a cost function which, given a gate-count dictionary,
            returns the total number of occurrences of the operators in ``gate_names``.

        **Example**

        >>> import pennylane.estimator as qre
        >>> from pennylane.labs.profiler.resource_profile import ProfileNode
        >>> gate_data = {
        ...     qre.X.resource_rep(): 1,
        ...     qre.T.resource_rep(): 5,
        ...     qre.Hadamard.resource_rep(): 2
        ... }
        >>> my_cost_func = ProfileNode.set_cost_func(["T", "Hadamard"])
        >>> my_cost_func(gate_data)
        7
        """
        target_names = [gate_names] if isinstance(gate_names, str) else gate_names

        def custom_cost_func(gate_data):
            val = 0
            for cmpr_rep, counts in gate_data.items():
                if cmpr_rep.name in target_names:
                    val += counts

            return val

        return custom_cost_func


def export_flame_graph_data(
    root_node: ProfileNode, group_by: str = "type", cost_func: Callable | None = None
):
    r"""Flatten a profile tree into the column data used to render a flame graph.

    The returned data is a tuple of four parallel lists, ``(ids, names, values, parents)``,
    matching the format expected by the icicle/flame-graph traces in plotting libraries such
    as Plotly. Each entry corresponds to a block in the flame graph: ``ids`` are unique
    identifiers, ``names`` are the multiplicity-annotated operator labels,
    ``values`` are the costs computed by ``cost_func`` and ``parents`` link each block to its
    parent via the parent's id. The root block is labelled ``"circuit"``.

    Args:
        root_node (ProfileNode): the root node of the profile tree to export.
        group_func (Callable | None): a function used to group sibling nodes by name at each
            level of the tree. If ``None``, :meth:`ProfileNode.default_group_func` is used.
        cost_func (Callable | None): a function mapping a gate-count dictionary to a numeric
            cost. If ``None``, :meth:`ProfileNode.default_cost_func` is used.

    Returns:
        tuple[list, list, list, list]: the ``(ids, names, values, parents)`` lists describing
        the flame graph.

    **Example**

    >>> from pennylane.labs.profiler import profile, export_flame_graph_data
    >>> import pennylane.labs.estimator_beta as qre
    >>> def circuit():
    ...     qre.CNOT()
    ...     qre.CNOT()
    >>> root, _ = profile(circuit, {"CNOT"})()
    >>> ids, names, values, parents = export_flame_graph_data(root)
    >>> names
    ['circuit', 'CNOT x2']
    """

    if group_by == "name":
        group_func = ProfileNode.group_by_name
    elif group_by == "type":
        group_func = ProfileNode.group_by_type
    else:
        raise ValueError(f"Unknown group_by value {group_by} encountered.")

    if cost_func is None:
        cost_func = ProfileNode.default_cost_func

    child_nodes = root_node.children
    gate_counts = root_node.gate_data

    export_data = (
        ["circuit"],  # ids,
        ["circuit"],  # names,
        [cost_func(gate_counts)],  # values,
        [""],  # parents,
    )
    grouped_data = group_func(child_nodes)

    if group_by == "type":
        _recurrsive_export_flame_graph_by_type(
            group_func,
            cost_func,
            grouped_data,
            "circuit",
            export_data,
        )

    if group_by == "name":
        _recurrsive_export_flame_graph_by_name(
            group_func,
            cost_func,
            grouped_data,
            "circuit",
            export_data,
        )

    return export_data


def _recurrsive_export_flame_graph_by_name(
    group_func, cost_func, grouped_data, parent_id, export_data
):
    if grouped_data is None:
        return

    for name, data in grouped_data.items():
        scalar, gate_data, child_nodes = data
        local_name = name + f" [x{_get_decimal_str(scalar)}]" if scalar > 1 else name

        local_id = parent_id + " - " + local_name
        local_val = cost_func(gate_data)

        ids, names, values, parents = export_data
        ids.append(local_id)
        names.append(local_name)
        values.append(local_val)
        parents.append(parent_id)

        sub_grouped_data = group_func(child_nodes)
        _recurrsive_export_flame_graph_by_name(
            group_func,
            cost_func,
            sub_grouped_data,
            local_id,
            export_data,
        )
    return


def _recurrsive_export_flame_graph_by_type(
    group_func, cost_func, grouped_data, parent_id, export_data
):
    if grouped_data is None:
        return

    for cmpr_op, data in grouped_data.items():
        name = cmpr_op.name
        scalar, gate_data, child_nodes = data
        local_name = name + f" [x{_get_decimal_str(scalar)}]" if scalar > 1 else name

        local_id = parent_id + " - " + str(cmpr_op)
        local_val = cost_func(gate_data)

        ids, names, values, parents = export_data
        ids.append(local_id)
        names.append(local_name)
        values.append(local_val)
        parents.append(parent_id)

        sub_grouped_data = group_func(child_nodes)
        _recurrsive_export_flame_graph_by_type(
            group_func,
            cost_func,
            sub_grouped_data,
            local_id,
            export_data,
        )
    return


def add_dicts(base_dict: defaultdict, other_dict: defaultdict, out=False):
    r"""Add together two dictionaries with integer values.

    Args:
        base_dict (defaultdict): the dictionary that values are added to.
        other_dict (defaultdict): the dictionary whose values are added to ``base_dict``.
        out (bool): if ``True``, a new ``defaultdict`` is returned and the inputs are left
            unchanged. If ``False`` (the default), ``base_dict`` is updated in place and
            returned.

    Returns:
        defaultdict: the sum of the two dictionaries. When ``out`` is ``False`` this is the
        same object as ``base_dict`` (mutated in place); when ``out`` is ``True`` it is a new
        ``defaultdict``.

    **Example**

    >>> from collections import defaultdict
    >>> from pennylane.labs.profiler.resource_profile import add_dicts
    >>> a = defaultdict(int, {"x": 1, "y": 2})
    >>> b = defaultdict(int, {"y": 3, "z": 4})
    >>> sorted(add_dicts(a, b, out=True).items())
    [('x', 1), ('y', 5), ('z', 4)]
    """
    if out:
        all_keys = set(base_dict.keys()) | set(other_dict.keys())
        return defaultdict(int, {k: base_dict.get(k, 0) + other_dict.get(k, 0) for k in all_keys})

    for key, val in other_dict.items():
        base_dict[key] += val
    return base_dict


def mul_dict(base_dict: defaultdict, scalar: int):
    r"""Multiply the integer values of a dictionary by an integer scalar.

    Args:
        base_dict (defaultdict): a dictionary with integer values.
        scalar (int): the integer to multiply each value by.

    Returns:
        defaultdict: a new ``defaultdict`` whose values are those of ``base_dict`` scaled by
        ``scalar``. The input dictionary is left unchanged.

    **Example**

    >>> from collections import defaultdict
    >>> from pennylane.labs.profiler.resource_profile import mul_dict
    >>> d = defaultdict(int, {"x": 2, "y": 3})
    >>> dict(mul_dict(d, 4))
    {'x': 8, 'y': 12}
    """
    return defaultdict(int, {key: base_dict[key] * scalar for key in base_dict})
