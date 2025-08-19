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

from functools import wraps

from catalyst import qjit
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

from pennylane import QNode
from pennylane import draw as qml_draw
from pennylane.tape import QuantumScript
from pennylane.typing import Callable

from ..compiler import Compiler
from .collector import QMLCollector

# This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def draw(qnode: QNode, *, level: None | int = None) -> Callable:
    """
    Draw the QNode at the specified level.

    This function can be used to visualize the QNode at different stages of the transformation pipeline
    when xDSL compilation passes are applied. If the specified level is not available, the highest level
    will be used as a fallback.

    The provided QNode is assumed to be decorated with xDSL compilation passes.
    If no passes are applied, the original QNode is visualized.

    Args:
        qnode (.QNode): the input QNode that is to be visualized. The QNode is assumed to be compiled with ``qjit``.
        level (None | int): the level of transformation to visualize. If `None`, the original QNode is visualized.


    Returns:
        Callable: A wrapper function that visualizes the QNode at the specified level.

    """
    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    def _draw_callback(pass_instance, module, pass_level):
        collector = QMLCollector(module)
        ops, meas = collector.collect()
        tape = QuantumScript(ops, meas)
        cache[pass_level] = (tape.draw(), pass_instance.name if pass_level else "No transforms")

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # We re-compile the qnode to ensure the passes are applied
        # with the args and kwargs provided by the user.
        # TODO: we could integrate the callback mechanism within `qjit`,
        # so that we wouldn't need to recompile the qnode twice.
        mlir_module = qnode.mlir_module if hasattr(qnode, "mlir_module") else None
        if mlir_module is None:
            func = qnode.user_function if hasattr(qnode, "user_function") else qnode
            jitted_qnode = qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(func)
            jitted_qnode.jit_compile(args, **kwargs)
            mlir_module = jitted_qnode.mlir_module

        Compiler.run(mlir_module, callback=_draw_callback)

        if level in cache:
            return cache[level][0]

        if cache:
            # Fallback to the highest level if the requested level is not available.
            # This is done for compatibility with qml.draw.
            return cache[max(cache.keys())][0]

        return qml_draw(qnode.user_function, level=level)

    return wrapper
