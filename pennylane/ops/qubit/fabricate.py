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
Implements the fabricate operation for Pauli-based computation.
"""

from functools import lru_cache

from pennylane.capture import enabled as capture_enabled
from pennylane.compiler import compiler
from pennylane.core.operator import Operator
from pennylane.wires import Wires

has_jax = True
try:
    # pylint: disable=ungrouped-imports
    from pennylane.capture import QpPrimitive
    from pennylane.wires import AbstractQubit
except ImportError:
    has_jax = False

_VALID_INIT_STATES = frozenset({"plus_i", "minus_i", "magic", "magic_conj"})


class Fabricate(Operator):
    """Fabricate an auxiliary qubit from a qubit factory.

    This operation produces a new qubit in a logical state that is not available
    as a simple transversal preparation, such as a magic state or (depending on the
    error-correction scheme) a :math:`|Y\\rangle` state.

    Args:
        init_state (str): The logical state to fabricate. Must be one of
            ``"plus_i"``, ``"minus_i"``, ``"magic"``, or ``"magic_conj"``.
        wires (WiresLike): The dynamically allocated wire produced by this operation.

    .. note::

        Circuits comprising ``fabricate`` are currently not executable on most
        backends. This operation is intended for Pauli-based computation analysis
        and compilation with Catalyst.

    .. seealso::
        :func:`~.fabricate`, :func:`catalyst.passes.ppm_compilation`
    """

    def __init__(self, init_state: str, wires=None):
        if init_state not in _VALID_INIT_STATES:
            raise ValueError(
                f'The init_state "{init_state}" is not allowed. '
                f"Allowed values are {sorted(_VALID_INIT_STATES)}."
            )
        wires = Wires([]) if wires is None else wires
        super().__init__(wires=wires)
        self.hyperparameters["init_state"] = init_state

    @property
    def init_state(self) -> str:
        """The logical state to fabricate."""
        return self.hyperparameters["init_state"]

    def label(self, decimals=None, base_label=None, cache=None, wire=None) -> str:
        del decimals, base_label, cache, wire
        return f"Fabricate({self.init_state})"

    def __repr__(self) -> str:
        return f"Fabricate('{self.init_state}')"

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparameters_dict = dict(metadata[1])
        return cls(hyperparameters_dict["init_state"], wires=wires)


@lru_cache
def _create_fabricate_primitive():
    """Create a primitive corresponding to a fabricate operation."""
    if not has_jax:
        return None

    fabricate_prim = QpPrimitive("fabricate")
    fabricate_prim.multiple_results = True

    @fabricate_prim.def_impl
    def _fabricate_primitive_impl(*_, init_state=""):
        raise NotImplementedError("jaxpr containing fabricate cannot be executed.")

    @fabricate_prim.def_abstract_eval
    def _fabricate_primitive_abstract_eval(*_, **__):
        return [AbstractQubit()]

    return fabricate_prim


def fabricate(init_state: str):
    """Fabricate an auxiliary qubit from a qubit factory.

    A fabricate operation produces a new qubit in a logical state such as a magic
    state (:math:`|m\\rangle`, :math:`|\\overline{m}\\rangle`) or, depending on the
    scheme, :math:`|Y\\rangle` or :math:`|-Y\\rangle`.

    .. note::

        Circuits comprising ``fabricate`` are currently not executable on most
        backends. This operation is intended for Pauli-based computation analysis
        and compilation with Catalyst.

    Args:
        init_state (str): The logical state to fabricate. Must be one of
            ``"plus_i"``, ``"minus_i"``, ``"magic"``, or ``"magic_conj"``.

    Returns:
        AbstractQubit: A dynamically allocated qubit in the requested logical state.

    Raises:
        ValueError: if ``init_state`` is not one of the allowed values.

    **Example:**

    ``fabricate`` requires program capture. With Catalyst, enable capture via
    ``@qjit(capture=True)`` and inspect the lowered MLIR:

    .. code-block:: python

        import pennylane as qp
        from catalyst import qjit

        @qjit(
            pipelines=[
                (
                    "pipe",
                    [
                        "canonicalize",
                        "verify-no-quantum-use-after-free",
                        "convert-to-value-semantics",
                        "canonicalize",
                    ],
                )
            ],
            target="mlir",
            capture=True,
        )
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            magic = qp.fabricate("magic")
            qp.pauli_measure("ZZ", wires=[0, magic])
            qp.deallocate(magic)
            return qp.expval(qp.Z(0))

        print(circuit.mlir_opt)
    """
    if init_state not in _VALID_INIT_STATES:
        raise ValueError(
            f'The init_state "{init_state}" is not allowed. '
            f"Allowed values are {sorted(_VALID_INIT_STATES)}."
        )

    if capture_enabled():
        primitive = _create_fabricate_primitive()
        (qubit,) = primitive.bind(init_state=init_state)
        return qubit

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.fabricate(init_state=init_state)

    raise NotImplementedError("fabricate is only supported with program capture or Catalyst QJIT.")
