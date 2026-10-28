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

from pennylane.core.transforms import CompilePipeline, transform

from ._utils import make_level_name_unique
from .parsing import parse_resources_json
from .resource import SpecsResources

# Used for MLIR analysis pass JSON filenames with pass-by-pass specs
_RESOURCE_ANALYSIS_PREFIX = "pennylane_specs_analysis_pass"


def resources_from_analysis_pass(
    qjit,
    original_qnode,
    level: int | tuple[int] | list[int],
    level_to_markers: dict[int, list[str]],
    level_to_name: dict[int, str],
    *args,
    **kwargs,
) -> dict[str, SpecsResources | list[SpecsResources]]:
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

    # pylint: disable=protected-access,too-many-arguments,too-many-statements
    from catalyst import QJIT  # pylint: disable=import-outside-toplevel

    iter_pipeline = copy.deepcopy(original_qnode._compile_pipeline)
    new_compile_pipeline = CompilePipeline()
    new_post_device_pipeline = CompilePipeline()

    if isinstance(level, int):
        level = [level]
    max_level = max(level)

    max_legal_level = len(iter_pipeline) + 1
    fname_to_level = {}

    if max_level > max_legal_level:
        bad_levels = ", ".join(str(lvl) for lvl in level if lvl > max_legal_level)
        raise ValueError(f"Requested specs levels {bad_levels} not found in MLIR pass list.")

    with tempfile.TemporaryDirectory(
        prefix=f"{_RESOURCE_ANALYSIS_PREFIX}_{os.getpid()}_"
    ) as tmpdirname:
        fname_prefix = f"{tmpdirname}/{_RESOURCE_ANALYSIS_PREFIX}_{time.time_ns()}_level_"

        if 0 in level:
            fname = f"{fname_prefix}before.json"
            fname_to_level[fname] = 0
            level_to_name[0] = (
                ", ".join(level_to_markers[0]) if 0 in level_to_markers else "Before MLIR Passes"
            )
            new_compile_pipeline += transform(pass_name="resource-analysis")(
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
                if comp_pass.pass_name == "resource-estimate":
                    new_compile_pipeline[-1].kwargs["output_json"] = True
                    new_compile_pipeline[-1].kwargs["output_fname"] = fname
                else:
                    new_compile_pipeline += transform(pass_name="resource-analysis")(
                        output_json=True, output_fname=fname
                    )

        compile_options = copy.deepcopy(qjit.compile_options)

        # The Max Legal Level is the device level
        if max_legal_level in level:
            fname = f"{fname_prefix}device.json"
            fname_to_level[fname] = max_legal_level
            level_to_name[max_legal_level] = make_level_name_unique(
                "Device Preprocessing", frozenset(level_to_name.values())
            )
            new_post_device_pipeline += transform(pass_name="resource-analysis")(
                output_json=True, output_fname=fname
            )
        else:
            # If device level not requested, skip device preprocessing
            compile_options.skip_preprocess = True

        new_qnode = original_qnode.update_pipelines(
            new_compile_pipeline, post_device_pipeline=new_post_device_pipeline
        )

        compile_options.target = "mlir_opt"
        # Otherwise, QNodes without arguments would also be compiled when the QJIT object is created
        compile_options.enable_aot_compilation = False
        if compile_options.pipelines is None:
            # If the user has not explicitly chosen a pipeline, prevent unnecessary work by
            # limiting which passes are applied to just the necessary ones. In this case, only
            # the passes up to and including the user-specified transforms (the
            # user-transform-stage) are run
            compile_options.pipelines = [("pipe", ["user-transform-stage"])]

        # Partially compile the QNode, producing JSON data with resource info
        QJIT(new_qnode, compile_options=compile_options).aot_compile(*args, **kwargs)

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
