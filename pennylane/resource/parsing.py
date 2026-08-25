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
"""
Functions for parsing Catalyst resource JSON data into :class:`~.resource.SpecsResources` objects.

.. warning::

    This module is intended for internal use only and may change or be removed in future releases.
"""

import copy
import itertools
import math
import warnings
from collections import defaultdict
from typing import Any

from .expression import Expression
from .resource import PBCSpecsResources, SpecsResources, num_to_letters


def ceil(value: int | float | Expression) -> int | Expression:
    """Rounds a value up to the nearest integer. Accounts for precision issues."""
    return math.ceil(round(value, 12))  # Tolerance of 12 decimal places


def _generate_display_name_for_symbolic_var(var: str, display_names: dict[str, str]) -> str:
    """Creates a human-readable display name for a symbolic variable.

    Uses the `display_names` dict as a cache for names which already have been generated,
    and generates a new name if one does not exist.

    Args:
        var (str): The raw symbolic variable name, usually a hash of some kind
        display_names (dict[str, str]): A mapping from symbolic variable names to their display
            names in the output. (modified in-place by this function)

    Returns:
        str: The human-readable display name for the symbolic variable.
    """
    if var not in display_names:
        display_names[var] = num_to_letters(len(display_names))
    return display_names[var]


def _update_resource_dict(
    result_dict: dict[str, Any], call_count: int | float | Expression, fn_resources: dict[str, Any]
) -> None:
    """Helper function to update a resource dictionary with the resources from a called function.

    Args:
        result_dict (dict[str, Any]): The resource dictionary to update
        call_count (int | float | Expression): The number of times the called function is invoked.
            For floating point values, this is the average number of times the function is called.
        fn_resources (dict[str, Any]): The resources of the called function
    """
    for label, value in fn_resources.items():
        result_dict[label] += call_count * value


def _mlir_resources_to_specs_resources(
    all_data: dict[str, Any],
    focus: str,
    fn_resources: dict[str, SpecsResources | None],
    display_names: dict[str, str],
) -> None:
    """
    Helper function to convert Catalyst resource JSON data into :class:`~.resource.SpecsResources` objects.

    Recursively resolves the resources for a given function call, combining subroutine resources
    with the appropriate multiplicative factors. Builds out `fn_resources`, a mapping from
    function name to the corresponding :class:`~pennylane.resource.SpecsResources` object.

    .. warning::

        The resulting :class:`~.resource.SpecsResources` objects will always be of the :class:`~.resource.SpecsResources`
        type, even if they should be instances of a subclass. This is necessary to preserve the
        format of the ``extended_fields`` attribute, which may contain additional fields from the
        original JSON.

        It is the responsibility of the caller to convert to a subclass after calling this function
        if needed.

    Args:
        all_data (dict[str, Any]): the full JSON representation of the resource data
        focus (str): the name of the function to resolve resources for in this call
        fn_resources (dict[str, SpecsResources | None]): the mapping from function name to
            resolved :class:`~.resource.SpecsResources` objects. (modified in-place by this function)
        display_names (dict[str, str]): a mapping from symbolic variable names to their display
            names in the output. (modified in-place by this function)
    """

    if focus in fn_resources:
        return

    # Set to None to mark that we are currently resolving this function, which helps with detecting recursion
    fn_resources[focus] = None
    resources = all_data[focus]

    # Process wire data
    num_wires = resources["num_qubits"]["total"]
    if resources["metadata"].get("auto_qubit_management", False):
        warnings.warn(
            f"Specs detected that function '{focus}' uses automatic qubit management. "
            "The number of qubits allocated by this function will not be known at this time, so "
            "the final allocation counts may be inaccurate.",
        )

    # Process quantum operations and measurements
    measurement_processes = defaultdict(int, resources["measurement_processes"])
    quantum_operations = defaultdict(int)
    for gate_size, ops in resources["quantum_operations"].items():
        for gate_name, count in ops.items():
            if gate_name in ("PPM", "PPR-pi/2", "PPR-pi/4", "PPR-pi/8", "PPR-Phi"):
                # Separate out PPMs and PPRs by weight
                gate_name += f"-w{gate_size}"

            quantum_operations[gate_name] += count

    # Extract extended fields
    extended_fields = copy.deepcopy(resources["extended_fields"])

    # Process function calls (both static and dynamic)
    # Recurses through all function calls and combines resources with the appropriate multiplicative factors
    function_calls = resources["function_calls"]
    for called_fn, call_count in itertools.chain(
        function_calls["static"].items(), function_calls["dynamic"].items()
    ):
        if not isinstance(call_count, (int, float)):
            # If there is no numeric call count, we have to treat this as a symbolic variable
            var_name = _generate_display_name_for_symbolic_var(call_count, display_names)

            call_count = Expression({(var_name,): 1})
        if called_fn not in fn_resources:
            _mlir_resources_to_specs_resources(all_data, called_fn, fn_resources, display_names)

        called_fn_resources = fn_resources[called_fn]
        if called_fn_resources is None:
            warnings.warn(
                f"Specs detected recursion during resolution of JSON resource data. "
                f"Function '{focus}' calls '{called_fn}' which is already being resolved. "
                "This recursive call will not be counted, so final results may be inaccurate."
            )
            continue

        num_wires += call_count * called_fn_resources.num_wires
        _update_resource_dict(
            quantum_operations, call_count, called_fn_resources.quantum_operations
        )
        _update_resource_dict(
            measurement_processes, call_count, called_fn_resources.measurement_processes
        )

        # Helper function to handle merging extended fields
        _handle_extended_fields(extended_fields, call_count, called_fn_resources)

    # Construct final specs resource objects
    # NOTE: Sorting these dicts by key ensures that the resulting SpecsResources objects have a
    # deterministic order, which is helpful for testing and readability

    fn_resources[focus] = SpecsResources(
        counts={k: quantum_operations[k] for k in sorted(quantum_operations.keys())},
        measurement_processes={
            k: measurement_processes[k] for k in sorted(measurement_processes.keys())
        },
        num_wires=num_wires,
        circuit_depth=None,  # Can't get depth from MLIR pass results
        extra=extended_fields,
    )


def _handle_extended_fields(
    extended_fields: dict[str, Any],
    call_count: int | float | Expression,
    called_fn_resources: SpecsResources,
) -> None:
    """Helper function to handle extended fields in the resource data.

    .. warning::
        This function modifies the input `extended_fields` argument.

    Args:
        extended_fields (dict[str, Any]): The extended fields from the resource data (modified by function)
        called_fn_resources (SpecsResources): The resources of the called function
        call_count (int | float | Expression): The number of times the called function is invoked.
            For float values, this is the average number of times the function is invoked.
    """

    unknown_fields = []

    for field_name in called_fn_resources.extra:
        match field_name:
            case "pbc_depth":
                pbc_depth = extended_fields.get("pbc_depth", None)
                if pbc_depth is None:
                    extended_fields["pbc_depth"] = {
                        "any_commuting_depth": call_count
                        * called_fn_resources.extra["pbc_depth"]["any_commuting_depth"],
                        "qubit_disjoint_depth": call_count
                        * called_fn_resources.extra["pbc_depth"]["qubit_disjoint_depth"],
                    }
                else:
                    pbc_depth["any_commuting_depth"] += (
                        call_count * called_fn_resources.extra["pbc_depth"]["any_commuting_depth"]
                    )
                    pbc_depth["qubit_disjoint_depth"] += (
                        call_count * called_fn_resources.extra["pbc_depth"]["qubit_disjoint_depth"]
                    )
            case _:
                unknown_fields.append(field_name)

    if unknown_fields:
        warnings.warn(
            f"Specs detected unknown extended fields in the resource data: {unknown_fields}. "
            "These fields will not be propagated correctly, so final results may be inaccurate.",
            UserWarning,
        )


def _convert_to_subclass(res: SpecsResources) -> SpecsResources:
    """
    Converts a :class:`~.resource.SpecsResources` instance to a subclass if possible.

    Ensures that all counts are rounded up to the nearest integer, as required by the
    :class:`~.resource.SpecsResources` class.

    Args:
        res (SpecsResources): The :class:`~.resource.SpecsResources` object to convert.

    Returns:
        SpecsResources: A :class:`~.resource.SpecsResources` object, potentially of
            a subclass type if the original object contained the appropriate extra data.
    """
    kwargs = {
        "counts": {op: ceil(count) for op, count in res.counts.items()},
        "measurement_processes": {
            meas: ceil(count) for meas, count in res.measurement_processes.items()
        },
        "num_wires": ceil(res.num_wires) if res.num_wires is not None else None,
        "circuit_depth": ceil(res.circuit_depth) if res.circuit_depth is not None else None,
    }
    # Copy the extra fields to avoid mutating the original object
    extra = copy.deepcopy(res.extra)

    if "pbc_depth" in extra:
        pbc_depth = extra.pop("pbc_depth")
        kwargs["any_commuting_depth"] = ceil(pbc_depth.pop("any_commuting_depth"))
        kwargs["qubit_disjoint_depth"] = ceil(pbc_depth.pop("qubit_disjoint_depth"))
        # Pylint gets confused by the dynamic updates to kwargs here
        # pylint: disable=missing-kwoa
        return PBCSpecsResources(
            **kwargs,
            extra=extra,
        )
    return SpecsResources(**kwargs, extra=extra)


def parse_resources_json(
    all_data: dict[str, Any],
) -> list[SpecsResources]:
    """Converts JSON resource data from Catalyst into :class:`~.resource.SpecsResources` objects.

    Args:
        all_data (dict[str, Any]): The full JSON representation of the resource data.

    Returns:
        list[SpecsResources]: A list of :class:`~.resource.SpecsResources` objects corresponding to the QNodes in the JSON data.
    """
    resource_data = {}

    for fn_name in all_data.keys():
        _mlir_resources_to_specs_resources(
            all_data, focus=fn_name, fn_resources=resource_data, display_names={}
        )

    if any(resources["metadata"]["has_branches"] for resources in all_data.values()):
        warnings.warn(
            "Specs was unable to determine the branch of a conditional or switch statement."
            " The results will take the maximum resources across all possible branches, serving as an upper bound.",
            UserWarning,
        )

    # Only include information about qnodes, ignoring any extra functions
    return [
        _convert_to_subclass(resource_data[fn])
        for fn, data in all_data.items()
        if data["metadata"]["qnode"]
    ]
