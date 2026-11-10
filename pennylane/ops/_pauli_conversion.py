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
"""Construct concrete operators from Pauli algebra."""

from functools import lru_cache

import pennylane as qp
from pennylane import capture, math, ops
from pennylane.core.queuing import QueuingManager
from pennylane.wires import WiresLike


def _make_factor(op, wire):
    """Construct a Pauli factor after the operator classes have loaded."""
    return {"I": ops.Identity, "X": ops.PauliX, "Y": ops.PauliY, "Z": ops.PauliZ}[op](wire)


_cached_factor = lru_cache(_make_factor)


@QueuingManager.stop_recording()
def to_operator(value, identity_wires: WiresLike = ()):
    """Convert a :class:`~pennylane.pauli.PauliWord` or :class:`~pennylane.pauli.PauliSentence` into a PennyLane :class:`~pennylane.core.operator.Operator`.

    Args:
        value (PauliWord or PauliSentence): The Pauli representation to convert.
        identity_wires (Iterable): Wires assigned to identity terms, including the
            identity used to represent zero. If empty, uses a nonempty sentence's
            wires; otherwise defaults to no wires. Does not change the wires of
            non-identity terms.

    Returns:
        Operator: The corresponding operator, without adding it to the queue.

    Raises:
        TypeError: If ``value`` is not a Pauli word or sentence.

    An empty word represents identity and an empty sentence represents zero.
    Zero coefficients are retained; no pruning is performed.

    **Example**

    >>> word = qp.pauli.PauliWord({0: "X", 1: "Z"})
    >>> word_op = qp.pauli.to_operator(word)
    >>> type(word).__name__, type(word_op).__name__
    ('PauliWord', 'Prod2')
    >>> word_op
    X(0) @ Z(1)
    >>> sentence = qp.pauli.PauliSentence({word: 2.0})
    >>> sentence_op = qp.pauli.to_operator(sentence)
    >>> type(sentence).__name__, type(sentence_op).__name__
    ('PauliSentence', 'SProd')
    >>> sentence_op
    2.0 * (X(0) @ Z(1))
    """

    if isinstance(value, qp.pauli.PauliWord):
        if len(value) == 0:
            return ops.Identity(wires=identity_wires)
        make_factor = _make_factor if capture.enabled() else _cached_factor
        factors = [make_factor(op, wire) for wire, op in value.items()]
        pauli_rep = qp.pauli.PauliSentence({value: 1})
        return factors[0] if len(factors) == 1 else ops.Prod2(factors, _init_pauli_rep=pauli_rep)

    if not isinstance(value, qp.pauli.PauliSentence):
        raise TypeError("Expected a PauliWord or PauliSentence.")
    if len(value) == 0:
        return ops.s_prod(0, ops.Identity(wires=identity_wires))
    summands = []
    identity_wires = identity_wires or value.wires
    for pw, coeff in value.items():
        pw_op = to_operator(pw, identity_wires=list(identity_wires))
        rep = qp.pauli.PauliSentence({pw: coeff})
        summands.append(
            pw_op
            if not math.is_tracer(coeff) and math.all(coeff == 1)
            else ops.SProd(coeff, pw_op, _pauli_rep=rep)
        )
    return summands[0] if len(summands) == 1 else ops.Sum(*summands, _pauli_rep=value)
