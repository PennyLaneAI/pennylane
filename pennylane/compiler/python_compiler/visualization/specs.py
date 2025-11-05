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
from functools import wraps
from typing import TYPE_CHECKING, Literal

from catalyst import qjit
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

from ..compiler import Compiler
from .specs_collector import specs_collect

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp

    from pennylane.workflow.qnode import QNode


# TODO: This function is identically defined within draw.py
def _get_mlir_module(qnode: QNode, args, kwargs) -> ModuleOp:
    """Ensure the QNode is compiled and return its MLIR module."""
    if hasattr(qnode, "mlir_module") and qnode.mlir_module is not None:
        return qnode.mlir_module

    func = getattr(qnode, "user_function", qnode)
    jitted_qnode = qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(func)
    jitted_qnode.jit_compile(args, **kwargs)
    return jitted_qnode.mlir_module


def mlir_specs(
    qnode: QNode, level: None | int | tuple[int] | list[int] | Literal["all"] = None
) -> callable:
    """Compute the specs used for a circuit at the level of an MLIR pass.

    Args:
        qnode (QNode): The (QJIT'd) qnode to get the specs for
        level (None | int, optional): The (int) level of the MLIR pass to get the specs for

    Returns:
        callable: A callable that returns the specs for the circuit at the specified level
    """
    cache: dict[int, tuple[dict[str, int], str]] = {}

    def _specs_callback(previous_pass, module, next_pass, pass_level=0):
        """Callback function for gathering circuit specs."""

        pass_instance = previous_pass if previous_pass else next_pass
        ops = specs_collect(module)

        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            ops,
            pass_name if pass_level else "No transforms",
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        if args or kwargs:
            warnings.warn(
                "The `specs` function does not yet support dynamic arguments.\n",
                UserWarning,
            )
        mlir_module = _get_mlir_module(qnode, args, kwargs)
        Compiler.run(mlir_module, callback=_specs_callback)

        if not cache:
            return None

        if level == "all":
            return [cache[lvl][0] for lvl in sorted(cache.keys())]
        if isinstance(level, (tuple, list)):
            return [cache[lvl][0] if lvl in cache else None for lvl in level]
        # Just one level was specified
        return cache.get(level, cache[max(cache.keys())])[0]

    return wrapper
