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

"""Common built-in decomposition rules."""

from __future__ import annotations

from pennylane.decomposition.decomposition_rule import register_resources


@register_resources({})
def null_decomp(*_, **__):
    """A decomposition rule that can be assigned to an operator so that the operator decomposes to nothing.

    **Example**

    .. code-block:: python

        import pennylane as qp
        from pennylane.decomposition import null_decomp

        qp.decomposition.enable_graph()

        @qp.transforms._tape_decompose(
            gate_set={qp.RZ},
            fixed_decomps={qp.GlobalPhase: null_decomp}
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.Z(0)

    >>> print(qp.draw(circuit)())
    0: ──RZ(3.14)─┤

    """
    return
