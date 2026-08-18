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
import json
import os
import tempfile
import time
from pathlib import Path

import pennylane as qp

from ._utils import make_level_name_unique
from .parsing import parse_resources_json
from .resource import SpecsResources

# Used for MLIR analysis pass JSON filenames with pass-by-pass specs
_RESOURCE_ANALYSIS_PREFIX = "pennylane_specs_analysis_pass"


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

    iter_pipeline = copy.deepcopy(original_qnode._compile_pipeline)
    new_compile_pipeline = qp.CompilePipeline()

    max_level = max(level) if isinstance(level, (list, tuple)) else level
    max_legal_level = len(iter_pipeline)
    fname_to_level = {}

    with tempfile.TemporaryDirectory(
        prefix=f"{_RESOURCE_ANALYSIS_PREFIX}_{os.getpid()}_"
    ) as tmpdirname:
        fname_prefix = f"{tmpdirname}/{_RESOURCE_ANALYSIS_PREFIX}_{time.time_ns()}_level_"

        if max_level > max_legal_level:
            bad_levels = ", ".join(str(lvl) for lvl in level if lvl > max_legal_level)
            raise ValueError(f"Requested specs levels {bad_levels} not found in MLIR pass list.")

        if 0 in level:
            fname = f"{fname_prefix}before.json"
            fname_to_level[fname] = 0
            level_to_name[0] = (
                ", ".join(level_to_markers[0]) if 0 in level_to_markers else "Before MLIR Passes"
            )
            new_compile_pipeline += qp.transform(pass_name="resource-analysis")(
                output_json=True, output_fname=fname
            )

        for i, comp_pass in enumerate(iter_pipeline, start=1):
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

        new_qnode = copy.copy(original_qnode)
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

            cur_level_resources = parse_resources_json(data)

            if len(cur_level_resources) == 1:
                cur_level_resources = cur_level_resources[0]

            results[level_to_name[curr_level]] = cur_level_resources

    return results
