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
"""Helper functions for converting MLIR resource analysis output into SpecsResources objects."""

import copy
import itertools
import json
import os
import re
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pennylane as qp

from .expression import Expression
from .resource import SpecsResources, SymbolicSpecsResources, num_to_letters

# Used for MLIR analysis pass JSON filenames with pass-by-pass specs
_RESOURCE_ANALYSIS_PREFIX = "pennylane_specs_analysis_pass"


def make_level_name_unique(level_name: str, existing_names: Iterable[str]) -> str:
    """Helper function to make a level name unique by appending a suffix if necessary.

    .. warning::

        This function is intended for internal use and may be subject to change without deprecation.

    Args:
        level_name (str): The original level name
        existing_names (Iterable[str]): The set of existing level names to check against

    Returns:
        str: A unique level name

    Example:
        >>> existing = {"cancel-inverses", "merge-rotations", "cancel-inverses-2"}
        >>> make_level_name_unique("cancel-inverses", existing)
        'cancel-inverses-3'
    """
    unique_name = level_name
    counter = 1
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{level_name}-{counter}"
    return unique_name


def _generate_display_name_for_symbolic_var(var: str, display_names: dict[str, str]) -> str:
    if var not in display_names:
        display_names[var] = num_to_letters(len(display_names))
    return display_names[var]


def _mlir_resources_to_specs_resources(
    all_data: dict[str, Any],
    focus: str,
    fn_resources: dict[str, SymbolicSpecsResources | None],
    display_names: dict[str, str],
) -> None:
    """
    Helper function to convert the output of the resource analysis pass into ``SpecsResources`` objects.

    Recursively resolves the resources for a given function call, combining subroutine resources
    with the appropriate multiplicative factors. Builds out `fn_resources`, a mapping from
    function name to the corresponding :class:`~pennylane.resource.SymbolicSpecsResources` object.

    .. note::

        All resources are stored within :class:`~pennylane.resource.SymbolicSpecsResources` objects
        as symbolic expressions, even if all values are concrete and knowable at compile time.
        It is the responsibility of the caller to upcast these to concrete valued
        :class:`~pennylane.resource.SpecsResources` objects if desired.

    Args:
        all_data (dict[str, Any]): the full data output from the MLIR resource analysis
        focus (str): the name of the function to resolve resources for in this call
        fn_resources (dict[str, SymbolicSpecsResources | None]): the mapping from function name to
            resolved `SymbolicSpecsResources` objects. (modified in-place by this function)
        display_names (dict[str, str]): a mapping from symbolic variable names to their display
            names in the output. (modified in-place by this function)
    """

    if focus in fn_resources:
        return

    # Set to None to mark that we are currently resolving this function, which helps with detecting recursion
    fn_resources[focus] = None
    resources = all_data[focus]

    operations = {k: resources["operations"][k] for k in resources["operations"].keys()}

    measurements = defaultdict(
        int, {k: resources["measurements"][k] for k in resources["measurements"].keys()}
    )
    gate_types = defaultdict(int)
    gate_sizes = defaultdict(int)
    num_allocs = resources["num_qubits"]

    if resources.get("auto_qubit_management", False):
        warnings.warn(
            f"Specs detected that function '{focus}' uses automatic qubit management. "
            "The number of qubits allocated by this function will not be known at this time, so "
            "the final allocation counts may be inaccurate.",
        )

    for res_name, count in operations.items():
        match = re.match(r"(.+)\((\d+)\)", res_name)  # Parse out the number of gates from the key
        gate_name, gate_size = match.groups() if match else (res_name, 0)

        if gate_name in ("PPM", "PPR-pi/2", "PPR-pi/4", "PPR-pi/8", "PPR-Phi"):
            # Separate out PPMs and PPRs by weight
            gate_name += f"-w{gate_size}"

        gate_types[gate_name] = +count
        gate_sizes[int(gate_size)] += count

    # Recurse through all function calls and combine resources with the appropriate multiplicative factors
    for called_fn, call_count in itertools.chain(
        resources["function_calls"].items(), resources["var_function_calls"].items()
    ):
        if not isinstance(call_count, int):
            # If there is no integer call count, we have to treat this as a symbolic variable
            var_name = _generate_display_name_for_symbolic_var(call_count, display_names)

            call_count = Expression({(var_name,): 1})
        if called_fn not in fn_resources:
            _mlir_resources_to_specs_resources(all_data, called_fn, fn_resources, display_names)

        called_fn_resources = fn_resources[called_fn]
        if called_fn_resources is None:
            warnings.warn(
                f"Specs detected recursion during resolution of MLIR resource analysis results. "
                f"Function '{focus}' calls '{called_fn}' which is already being resolved. "
                "This recursive call will not be counted, so final results may be inaccurate."
            )
            continue

        num_allocs += call_count * called_fn_resources.num_allocs
        for gate, gate_count in called_fn_resources.gate_types.items():
            gate_types[gate] += call_count * gate_count
        for size, size_count in called_fn_resources.gate_sizes.items():
            gate_sizes[size] += call_count * size_count
        for meas, meas_count in called_fn_resources.measurements.items():
            measurements[meas] += call_count * meas_count

    # Sorting these dicts by key ensures that the resulting SymbolicSpecsResources objects have a deterministic order,
    # which is helpful for testing and readability
    fn_resources[focus] = SymbolicSpecsResources(
        gate_types={k: gate_types[k] for k in sorted(gate_types.keys())},
        gate_sizes={k: gate_sizes[k] for k in sorted(gate_sizes.keys())},
        measurements={k: measurements[k] for k in sorted(measurements.keys())},
        num_allocs=num_allocs,
        depth=None,  # Can't get depth from MLIR pass results
    )


def _get_resources_from_analysis_pass(
    all_data: dict[str, Any],
) -> list[SpecsResources]:
    resource_data = {}

    for fn_name in all_data.keys():
        _mlir_resources_to_specs_resources(
            all_data, focus=fn_name, fn_resources=resource_data, display_names={}
        )

    if any(resources["has_branches"] for resources in all_data.values()):
        warnings.warn(
            "Specs was unable to determine the branch of a conditional or switch statement."
            " The results will take the maximum resources across all possible branches, serving as an upper bound.",
            UserWarning,
        )

    # Only include information about qnodes, ignoring any extra functions
    # The blank substitution will return a concrete SpecsResources if no symbolic variables remain
    return [resource_data[fn].subs() for fn, data in all_data.items() if data["qnode"]]


def _execute_analysis_pass(
    new_qnode,
    compile_options,
    *args,
    **kwargs,
):  # pragma: no cover
    """
    Helper function to compile the QNode with the resource analysis pass inserted, which will output
    the necessary JSON files for MLIR analysis.

    This function will stop compilation before lowering to LLVM, avoiding the typical Catalyst
    compilation strategy.
    """
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests

    # pylint: disable=import-outside-toplevel,protected-access
    try:
        from catalyst import QJIT
    except ImportError as e:
        raise ImportError(
            "Catalyst must be installed to use specs with QJIT-compiled QNodes. "
            "Please install Catalyst and try again."
        ) from e

    new_qjit = QJIT(new_qnode, compile_options=compile_options)

    # Force a compilation, which will output the necessary JSON files
    # This code snippet is adapted from the source code of `QJIT.jit_compile`
    if new_qjit.mlir_module is None:
        new_qjit.workspace = new_qjit._get_workspace()
        new_qjit.jaxed_function = None
        if new_qjit.compiled_function and new_qjit.compiled_function.shared_object:
            new_qjit.compiled_function.shared_object.close()

        new_qjit.jaxpr, new_qjit.out_type, new_qjit.out_treedef, new_qjit.c_sig = new_qjit.capture(
            args, **kwargs
        )

        new_qjit.mlir_module = new_qjit.generate_ir()

    # Force resolution of this property to finish going through all MLIR passes
    if new_qjit.mlir_opt is None:
        raise ValueError(
            "Specs failed to compile the QNode with the specified passes for MLIR analysis."
        )


def resources_from_analysis_pass(
    qjit,
    original_qnode,
    level: int | tuple[int] | list[int],
    num_tape_levels: int,
    level_to_markers: dict[int, list[str]],
    level_to_name: dict[int, str],
    *args,
    **kwargs,
) -> dict[str, SpecsResources | list[SpecsResources]]:  # pragma: no cover
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests
    """
    Helper function to get specs information from MLIR analysis passes inserted at the specified
    levels.

    .. warning::

        This function is intended for internal use and may be subject to change without deprecation.

    Creates a new compile pipeline with extra resources analysis passes inserted at
    the appropriate levels, then compiles the QNode with this pipeline to get the resource
    information from the output JSON files.

    Args:
        qjit (:class:`~catalyst.QJIT`): the QNode to calculate the specifications for.
        original_qnode (:class:`~pennylane.QNode`): the original QNode before any compilation
        level (int | tuple[int] | list[int]): the levels at which to insert resource analysis passes
            for resource counting
        num_tape_levels (int): the number of tape transform levels in the compile pipeline
        level_to_markers (dict[int, list[str]]): mapping from level number to a list of marker names
        level_to_name (dict[int, str]): mapping from level number to the name to use for that level
            in the output. Note that this argument is mutated by this function
        *args: the arguments to pass to the QNode when compiling
        **kwargs: the keyword arguments to pass to the QNode when compiling
    Returns:
        dict[str, SpecsResources | list[SpecsResources]]: A mapping from level name to the
            corresponding resource information.
    """

    # pylint: disable=protected-access,too-many-arguments

    new_qnode = copy.deepcopy(original_qnode)
    iter_pipeline = new_qnode._compile_pipeline
    new_compile_pipeline = qp.CompilePipeline()

    max_level = max(level) if isinstance(level, (list, tuple)) else level
    max_legal_level = len(iter_pipeline)
    fname_to_level = {}

    with tempfile.TemporaryDirectory(
        prefix=f"{_RESOURCE_ANALYSIS_PREFIX}_{os.getpid()}_"
    ) as tmpdirname:
        fname_prefix = f"{tmpdirname}/{_RESOURCE_ANALYSIS_PREFIX}_{time.time_ns()}_level_"

        if num_tape_levels > 0:
            # Account for the inserted lowering pass which comes after all tape transforms
            max_legal_level += 1

            # Add all tape transforms first, which come before any MLIR passes
            new_compile_pipeline += iter_pipeline[: num_tape_levels - 1]
            iter_pipeline = iter_pipeline[num_tape_levels - 1 :]

        if max_level > max_legal_level:
            bad_levels = ", ".join(str(lvl) for lvl in level if lvl > max_legal_level)
            raise ValueError(f"Requested specs levels {bad_levels} not found in MLIR pass list.")

        if num_tape_levels in level:
            fname = f"{fname_prefix}before.json"
            fname_to_level[fname] = (
                num_tape_levels  # num_tape_levels == the level of the lowering pass
            )
            level_to_name[num_tape_levels] = (
                ", ".join(level_to_markers[num_tape_levels])
                if num_tape_levels in level_to_markers
                else "Before MLIR Passes"
            )
            new_compile_pipeline += qp.transform(pass_name="resource-analysis")(
                output_json=True, output_fname=fname
            )

        for i, comp_pass in enumerate(iter_pipeline, start=num_tape_levels + 1):
            if i > max_level:
                break
            new_compile_pipeline += comp_pass
            if i in level:
                fname = f"{fname_prefix}{i}.json"
                level_name = (
                    ", ".join(level_to_markers[i])
                    if i in level_to_markers
                    else comp_pass.pass_name or f"Level {i}"
                )
                level_name = make_level_name_unique(level_name, frozenset(level_to_name.values()))
                fname_to_level[fname] = i
                level_to_name[i] = level_name
                new_compile_pipeline += qp.transform(pass_name="resource-analysis")(
                    output_json=True, output_fname=fname
                )

        new_qnode._compile_pipeline = new_compile_pipeline
        compile_options = copy.deepcopy(qjit.compile_options)
        compile_options.target = "mlir"
        compile_options.lower_to_llvm = False
        if compile_options.pipelines is None:
            # If the user has not explicitly chosen a pipeline, prevent unnecessary work by
            # limiting which passes are applied to just the necessary ones. In this case, only
            # the set of user-specified transforms (the quantum-compilation-stage) are run
            compile_options.pipelines = [("pipe", ["quantum-compilation-stage"])]

        # Partially compile the QNode, producing JSON data with resource info
        _execute_analysis_pass(new_qnode, compile_options, *args, **kwargs)

        results = {}

        for res_file, curr_level in fname_to_level.items():
            res_file = Path(res_file)
            with res_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            cur_level_resources = _get_resources_from_analysis_pass(data)

            if len(cur_level_resources) == 1:
                cur_level_resources = cur_level_resources[0]

            results[level_to_name[curr_level]] = cur_level_resources

    return results
