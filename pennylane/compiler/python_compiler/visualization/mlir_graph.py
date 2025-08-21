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
"""
This file contains the implementation of the MLIR graph generation for the Unified Compiler framework.
"""

import io
import subprocess
from functools import wraps
from pathlib import Path

from catalyst import qjit
from catalyst.compiler import CompileError, _get_catalyst_cli_cmd
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from xdsl.printer import Printer

from pennylane import QNode
from pennylane.typing import Callable

from ..compiler import Compiler

try:
    from graphviz import Source as GraphSource
    has_graphviz = True
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_graphviz = False


# TODO: This interface can be removed once the _quantum_opt interface
# implemented in catalyst.compiler returns the `stderr` output
def _quantum_opt_stderr(*args, stdin=None, stderr_return=False):
    """Raw interface to quantum-opt"""
    return _catalyst(("--tool", "opt"), *args, stdin=stdin, stderr_return=stderr_return)


def _catalyst(*args, stdin=None, stderr_return=False):
    """Raw interface to catalyst"""
    cmd = _get_catalyst_cli_cmd(*args, stdin=stdin)
    try:
        result = subprocess.run(cmd, input=stdin, check=True, capture_output=True, text=True)
        if stderr_return:
            return result.stdout, result.stderr
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise CompileError(f"catalyst failed with error code {e.returncode}: {e.stderr}") from e


def _mlir_graph_callback(pass_instance, module, pass_level):
    """
    Callback function for MLIR graph generation.
    """
    buffer = io.StringIO()
    Printer(stream=buffer, print_generic_format=True).print_op(module)
    _, dot_graph = _quantum_opt_stderr(
        "--view-op-graph", stdin=buffer.getvalue(), stderr_return=True
    )
    graph = GraphSource(dot_graph)

    out_dir = Path("mlir_generated_graphs")
    out_dir.mkdir(exist_ok=True)

    pass_name = f"after_{pass_instance.name}" if pass_level else "no_transforms"
    out_file = out_dir / f"QNode_level_{pass_level}_{pass_name}.svg"

    with open(out_file, "wb") as f:
        f.write(graph.pipe(format="svg"))


def generate_mlir_graph(qnode: QNode) -> Callable:
    """
    Generate an MLIR graph for the given QNode and saves it to a file.

    This function uses the callback mechanism of the unified compiler framework to generate
    the MLIR graph in between compilation passes. The provided QNode is assumed to be decorated with xDSL compilation passes.
    The ``qjit`` decorator is used to recompile the QNode with the passes and the provided arguments.

    If no compilation passes are applied, the MLIR graph will not be generated.

    Args:
        qnode (.QNode): the input QNode that is to be visualized.


    Returns:
        Callable: A wrapper function that generates the MLIR graph.

    """

    if not has_graphviz:
        raise ImportError(
            "This feature requires graphviz, a library for graph visualization. "
            "It can be installed with:\n\npip install graphviz"
        )  # pragma: no cover

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # We re-compile the qnode to ensure the passes are applied
        # with the args and kwargs provided by the user.
        # TODO: we could integrate the callback mechanism within `qjit`,
        # so that we wouldn't need to recompile the qnode twice.
        mlir_module = qnode.mlir_module
        if mlir_module is None:
            new_qnode = qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(qnode.user_function)
            new_qnode.jit_compile(args, **kwargs)
            mlir_module = new_qnode.mlir_module

        Compiler.run(mlir_module, callback=_mlir_graph_callback)

    return wrapper
