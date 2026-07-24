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
"""Index switch control flow."""

from collections.abc import Sequence

from pennylane.compiler.compiler import AvailableCompilers, active_compiler

from ._resource_hints import validate_estimated_probabilities


def switch(index_var, *, estimated_probabilities: Sequence[float] | None = None):
    """A :func:`~.qjit` compatible index-switch decorator for PennyLane programs.

    This decorator provides a functional version of an index switch, delegating to
    :func:`catalyst.switch` when used inside :func:`~.qjit`.

    Args:
        index_var: the case index used to select a branch at runtime
        estimated_probabilities (Sequence[float]): Optional hint for resource estimation.
            One probability per explicit ``case`` branch (in registration order). The default
            branch probability is computed automatically as the remaining mass. Only used with
            :func:`~.qjit` and Catalyst's resource analysis.

    Returns:
        Callable: A decorator wrapping the default branch of the switch.

    .. seealso:: :func:`catalyst.switch`, :func:`~.cond`, :func:`~.qjit`

    **Example**

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(i):
            @qp.switch(i, estimated_probabilities=[0.2, 0.3])
            def my_switch():
                qp.Z(0)

            @my_switch.branch(0)
            def branch0():
                qp.X(0)

            @my_switch.branch(1)
            def branch1():
                qp.H(0)

            my_switch()
            return qp.probs()
    """
    if estimated_probabilities is not None:
        estimated_probabilities = validate_estimated_probabilities(estimated_probabilities)

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        switch_kwargs = {}
        if estimated_probabilities is not None:
            switch_kwargs["estimated_probabilities"] = estimated_probabilities
        return ops_loader.switch(index_var, **switch_kwargs)

    raise RuntimeError(
        "qp.switch is only supported inside a qjit-compiled workflow. "
        "Please use catalyst.switch directly or compile your workflow with qp.qjit."
    )
