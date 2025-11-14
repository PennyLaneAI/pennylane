# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the implementation of the `specs` function for the Unified Compiler."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from ..compiler import Compiler
from .specs_collector import ResourcesResult, specs_collect
from .xdsl_conversion import get_mlir_module

if TYPE_CHECKING:
    from catalyst.jit import QJIT


class StopCompilation(Exception):
    """Custom exception to stop compilation early when the desired specs level is reached."""


def mlir_specs(
    qnode: QJIT, level: int | tuple[int] | list[int] | Literal["all"], *args, **kwargs
) -> ResourcesResult | dict[str, ResourcesResult]:
    """Compute the specs used for a circuit at the level of an MLIR pass.

    Args:
        qnode (QNode): The (QJIT'd) qnode to get the specs for
        level (int | tuple[int] | list[int] | "all"): The level of the MLIR pass to get the specs for
        *args: Positional arguments to pass to the QNode
        **kwargs: Keyword arguments to pass to the QNode

    Returns:
        ResourcesResult | dict[str, ResourcesResult]: The resources for the circuit at the specified level
    """
    cache: dict[int, tuple[ResourcesResult, str]] = {}

    if args or kwargs:
        warnings.warn(
            "The `specs` function does not yet support dynamic arguments, so the results may not reflect information provided by the arguments.",
            UserWarning,
        )

    max_level = level
    if max_level == "all":
        max_level = None
    elif isinstance(level, (tuple, list)):
        max_level = max(level)
    elif not isinstance(level, int):
        raise ValueError("The `level` argument must be an int, a tuple/list of ints, or 'all'.")

    def _specs_callback(previous_pass, module, next_pass, pass_level=0):
        """Callback function for gathering circuit specs."""

        pass_instance = previous_pass if previous_pass else next_pass
        ops = specs_collect(module)

        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            ops,
            pass_name if pass_level else "Before MLIR Passes",
        )

        if max_level is not None and pass_level >= max_level:
            raise StopCompilation("Stopping compilation after reaching max specs level.")

    mlir_module = get_mlir_module(qnode, args, kwargs)
    try:
        Compiler.run(mlir_module, callback=_specs_callback)
    except StopCompilation:
        # We use StopCompilation to interrupt the compilation once we reach
        # the desired level
        pass

    if level == "all":
        return {f"{cache[lvl][1]} (MLIR-{lvl})": cache[lvl][0] for lvl in sorted(cache.keys())}

    if isinstance(level, (tuple, list)):
        if any(lvl not in cache for lvl in level):
            missing = [str(lvl) for lvl in level if lvl not in cache]
            raise ValueError(
                f"Requested specs levels {', '.join(missing)} not found in MLIR pass list."
            )
        return {f"{cache[lvl][1]} (MLIR-{lvl})": cache[lvl][0] for lvl in level if lvl in cache}

    # Just one level was specified
    if level not in cache:
        raise ValueError(f"Requested specs level {level} not found in MLIR pass list.")
    return cache[level][0]
