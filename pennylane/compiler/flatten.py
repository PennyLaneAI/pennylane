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
"""A compile-time, one-way exit point from ``qjit``-compiled programs to tapes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pennylane.exceptions import CompileError

from .compiler import AvailableCompilers, _check_compiler_version, available

if TYPE_CHECKING:
    from pennylane.tape import QuantumScript


def flatten(fn, compiler="catalyst") -> Callable[..., QuantumScript]:
    """Return a function that produces the compiled tape of a ``qjit``-compiled program.

    The program is lowered to MLIR and compiled with its ``CompilePipeline`` (for example
    :func:`~.transforms.cancel_inverses` or :func:`~.transforms.merge_rotations`). The compiled
    program is then converted into a :class:`~.QuantumScript` holding the quantum instructions,
    the terminal measurements and the shots, similar to ``qnode._tape`` in the non-compiled
    pathway. Execution configuration, such as the MCM method or the differentiation method, is not
    part of the tape.

    All arguments of the returned function are treated as static (compile-time constants). Control
    flow is fully unrolled; branches conditioned on mid-circuit measurement outcomes
    (:func:`~.cond`) become :class:`~.ops.Conditional` operations tied to the corresponding
    :class:`~.ops.MidMeasure`. An error is raised if the program contains dynamic behaviour that
    cannot be resolved at compile time.

    The tape is a one-way exit: it can be transformed, executed and drawn with the tape-based
    PennyLane tooling, but it cannot be compiled with :func:`~.qjit` again.

    Args:
        fn (QJIT): a function decorated with :func:`~.qjit` that executes a single QNode.
        compiler (str): the name of the compiler package that compiled ``fn``.

    Returns:
        Callable[..., QuantumScript]: a function with the same signature as ``fn`` that returns
        the compiled tape.

    **Example**

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f(x):
            qp.H(0)
            qp.H(0)
            qp.RX(x, 1)
            qp.RX(0.2, 1)
            return qp.expval(qp.Z(1))

    >>> tape = qp.flatten(f)(0.1)
    >>> tape.operations
    [RX(0.30000000000000004, wires=[1])]
    >>> tape.measurements
    [expval(Z(1))]
    """
    if not available(compiler):
        raise CompileError(f"The {compiler} package is not installed.")  # pragma: no cover

    _check_compiler_version(compiler)

    ops_loader = AvailableCompilers.names_entrypoints[compiler]["ops"].load()
    if not hasattr(ops_loader, "flatten"):
        raise CompileError(f"The {compiler} compiler does not support flatten.")

    return ops_loader.flatten(fn)
