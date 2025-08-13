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

import io
import subprocess
from functools import wraps

import graphviz
from catalyst import qjit
from catalyst.compiler import CompileError, _get_catalyst_cli_cmd
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from xdsl.printer import Printer

from pennylane.tape import QuantumScript
from pennylane.typing import Callable

from ..transforms.api.apply_transform_sequence import register_callback
from .collector import QMLCollector


def _catalyst(*args, stdin=None, stderr_return=False):
    """Raw interface to catalyst

    echo ${stdin} | catalyst *args -
    catalyst *args
    """
    cmd = _get_catalyst_cli_cmd(*args, stdin=stdin)
    try:
        result = subprocess.run(cmd, input=stdin, check=True, capture_output=True, text=True)
        if stderr_return:
            return result.stdout, result.stderr
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise CompileError(f"catalyst failed with error code {e.returncode}: {e.stderr}") from e


def _quantum_opt_stderr(*args, stdin=None, stderr_return=False):
    """Raw interface to quantum-opt

    echo ${stdin} | catalyst --tool=opt *args -
    catalyst --tool=opt *args
    """
    return _catalyst(("--tool", "opt"), *args, stdin=stdin, stderr_return=stderr_return)


def catalyst_qjit(qnode):
    """A method checking whether a qnode is compiled by catalyst.qjit"""
    return qnode.__class__.__name__ == "QJIT" and hasattr(qnode, "user_function")


# TODO: This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def draw(qnode, *, level: None | int = None):
    "Draw the quantum circuit at the specified level."

    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    if catalyst_qjit(qnode):
        qnode = qnode.original_function

    def draw_callback(pass_instance, module, pass_level):
        collector = QMLCollector(module)
        ops, meas = collector.collect()
        # This is just a quick way to visualize the circuit
        # using PennyLane's built-in drawing capabilities of QuantumScript
        tape = QuantumScript(ops, meas)
        cache[pass_level] = (tape.draw(), pass_instance.name if pass_level else "No transforms")

        buffer = io.StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(module)
        _, dot_graph = _quantum_opt_stderr(
            "--view-op-graph", stdin=buffer.getvalue(), stderr_return=True
        )
        graph = graphviz.Source(dot_graph)
        with open(f"level_{pass_level}.svg", "wb") as f:
            f.write(graph.pipe(format="svg"))

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        register_callback(draw_callback)
        qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(qnode)

        if level is not None:
            if level in cache:
                return cache[level][0]
            raise ValueError("No drawing available for the specified level.")

        for lvl in sorted(cache):
            print(f"\n[Level {lvl}]")
            print(f"Applying transform: {cache[lvl][1]}")
            print(cache[lvl][0])

    return wrapper
