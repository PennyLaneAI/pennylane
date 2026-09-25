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
"""
This module contains :func:`apply`, which places a gadget into a program compiled with
:func:`~pennylane.qjit`.
"""

from __future__ import annotations

import functools

from pennylane import capture
from pennylane.wires import Wires

from .authoring import TracedGadget
from .ir import GadgetError


@functools.lru_cache(maxsize=1)
def get_gadget_call_prim():
    """Get the primitive that records a gadget call, creating it on the first call.

    The primitive takes the wires of the logical qubits as its arguments. Its parameters are
    the gadget's name and ``payload``, the text of the module emitted by
    :func:`~.lowering.emit`. Catalyst lowers it to a ``quantum.custom "GadgetCall"`` operation
    carrying the payload, which its ``convert-quantum-to-qecl`` pass replaces with the gadget's
    ``qecl`` operations.

    Returns:
        QpPrimitive: the primitive
    """
    from pennylane.capture.custom_primitives import QpPrimitive

    prim = QpPrimitive("gadget_call")
    prim.multiple_results = True

    @prim.def_abstract_eval
    def _(*_wires, **_params):
        return []

    return prim


def apply(traced: TracedGadget, wires) -> None:
    """Apply a gadget to logical qubits inside a program compiled with :func:`~pennylane.qjit`.

    The wires are logical qubits of a QNode compiled with Catalyst's QEC pipeline, which
    encodes each wire in a code block. The call is replaced by the gadget's operations when
    the pipeline converts the program to logical codeblocks, and the pipeline checks that the
    gadget was written for the code it compiles to.

    Only gadgets that the ``qecl`` dialect can express can be compiled: single-phase gadgets on
    one logical qubit, whose records are not used by later operations, such as
    :func:`~.library.steane_memory`. Other gadgets raise :class:`~.lowering.LoweringGap` when
    the program is compiled.

    Args:
        traced (~.TracedGadget): the gadget
        wires (Sequence[int] or int): the logical qubits it acts on, one per logical qubit of
            the gadget's code

    Raises:
        GadgetError: if the number of wires does not match the gadget's code, or program
            capture is not enabled

    **Example**

    .. code-block:: python

        import pennylane as qp
        from catalyst.ftqc import qec_pipeline
        from catalyst.python_interface.transforms.qecl import convert_quantum_to_qecl_pass
        from catalyst.python_interface.transforms.qecp import (
            convert_qecl_to_qecp_pass,
            convert_qecp_to_quantum_pass,
        )
        from pennylane.ftqc import gadget
        from pennylane.ftqc.gadget.library import steane_memory

        _, _, memory = steane_memory(rounds=3)

        @qp.qjit(capture=True, pipelines=qec_pipeline())
        @convert_qecp_to_quantum_pass
        @convert_qecl_to_qecp_pass(qec_code="Steane", number_errors=0)
        @convert_quantum_to_qecl_pass(k=1)
        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit", wires=1), mcm_method="one-shot")
        def circuit():
            qp.X(0)
            gadget.apply(memory, wires=0)
            return qp.sample(wires=[0])

    Each shot returns ``1``: the three rounds of the memory gadget preserve the logical state.
    """
    if not isinstance(traced, TracedGadget):
        raise GadgetError(f"apply: expected a TracedGadget, got {type(traced).__name__}")
    wires = Wires(wires)
    code = traced.program.code
    if len(wires) != code.k:
        raise GadgetError(
            f"apply: gadget {traced.name} acts on the {code.k} logical qubit(s) of code "
            f"{code.name}, but {len(wires)} wire(s) were given"
        )
    if not capture.enabled():
        raise GadgetError(
            "apply records a gadget call into a program captured for qjit; enable program "
            "capture, for example with qp.qjit(capture=True)"
        )

    from .lowering import emit

    payload = emit(traced.program).text()
    get_gadget_call_prim().bind(*wires, name=traced.name, payload=payload)


__all__ = ["apply", "get_gadget_call_prim"]
