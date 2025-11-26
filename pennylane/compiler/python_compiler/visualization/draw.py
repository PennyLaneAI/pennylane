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
"""This file contains the implementation of the `draw` function for the Unified Compiler."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import TYPE_CHECKING

from catalyst import qjit
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

from pennylane.tape import QuantumScript

from ..compiler import Compiler
from .collector import QMLCollector

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp

    from pennylane.typing import Callable
    from pennylane.workflow.qnode import QNode

# TODO: This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def _get_mlir_module(qnode: QNode, args, kwargs) -> ModuleOp:
    """Ensure the QNode is compiled and return its MLIR module."""
    if hasattr(qnode, "mlir_module") and qnode.mlir_module is not None:
        return qnode.mlir_module

    func = getattr(qnode, "user_function", qnode)
    jitted_qnode = qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(func)
    jitted_qnode.jit_compile(args, **kwargs)
    return jitted_qnode.mlir_module


def draw(qnode: QNode, *, level: None | int = None) -> Callable:
    """
    Draw the QNode at the specified level.

    This function can be used to visualize the QNode at different stages of the transformation pipeline
    when xDSL or Catalyst compilation passes are applied.
    If the specified level is not available, the highest level will be used as a fallback.

    The provided QNode is assumed to be decorated with compilation passes.
    If no passes are applied, the original QNode is visualized.

    Args:
        qnode (.QNode): the input QNode that is to be visualized. The QNode is assumed to be compiled with ``qjit``.
        level (None | int): the level of transformation to visualize. If `None`, the final level is visualized.


    Returns:
        Callable: A wrapper function that visualizes the QNode at the specified level.

    """
    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    def _draw_callback(previous_pass, module, next_pass, pass_level=0):
        """Callback function for circuit drawing."""

        pass_instance = previous_pass if previous_pass else next_pass
        collector = QMLCollector(module)
        ops, meas = collector.collect()
        tape = QuantumScript(ops, meas)
        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            tape.draw(show_matrices=False),
            pass_name if pass_level else "No transforms",
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        if args or kwargs:
            warnings.warn(
                "The `draw` function does not yet support dynamic arguments.\n"
                "To visualize the circuit with dynamic parameters or wires, please use the\n"
                "`compiler.python_compiler.visualization.generate_mlir_graph` function instead.",
                UserWarning,
            )
        mlir_module = _get_mlir_module(qnode, args, kwargs)
        Compiler.run(mlir_module, callback=_draw_callback)

        if not cache:
            return None

        return cache.get(level, cache[max(cache.keys())])[0]

    return wrapper
