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
Implements the pauli measurement.
"""

import uuid
import warnings
from functools import lru_cache

from pennylane import math
from pennylane.capture import enabled as capture_enabled
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.operation import Operator
from pennylane.wires import Wires, WiresLike

from .measurement_value import MeasurementValue

_VALID_PAULI_CHARS = "XYZ"


class PauliMeasure(Operator):
    """A Pauli product measurement."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        pauli_word: str,
        wires: WiresLike,
        postselect: int | None = None,
        id: str | None = None,
        meas_uid: str | None = None,
    ):
        if id is not None:
            warnings.warn(
                "The 'id' argument has been renamed to 'meas_uid'. Access through 'id' will be removed in v0.46.",
                PennyLaneDeprecationWarning,
            )
            # Only override if meas_uid wasn't explicitly provided
            if meas_uid is None:
                meas_uid = id

        if not all(c in _VALID_PAULI_CHARS for c in pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that '
                "are not allowed. Allowed characters are X, Y and Z."
            )

        wires = Wires(wires)
        if len(pauli_word) != len(wires):
            raise ValueError(
                "The number of wires must be equal to the length of the Pauli "
                f"word. The Pauli word {pauli_word} has length {len(pauli_word)} "
                f"and {len(wires)} wires were given: {wires}."
            )
        super().__init__(wires=wires)
        self.hyperparameters["pauli_word"] = pauli_word
        self.hyperparameters["postselect"] = postselect
        self.hyperparameters["meas_uid"] = meas_uid

    @property
    def meas_uid(self) -> str | None:
        """The custom ID associated with the measurement instance."""
        return self.hyperparameters["meas_uid"]

    @property
    def pauli_word(self) -> str:
        """The Pauli word for the measurement."""
        return self.hyperparameters["pauli_word"]

    @property
    def postselect(self) -> int | None:
        """Which outcome to postselect after the measurement."""
        return self.hyperparameters["postselect"]

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)

    def __repr__(self) -> str:
        return f"PauliMeasure('{self.pauli_word}', wires={self.wires.tolist()})"

    def label(self, decimals=None, base_label=None, cache=None, wire=None) -> str:
        """How the pauli-product measurement is represented in diagrams and drawings."""
        postselect = "" if self.postselect is None else ("₁" if self.postselect == 1 else "₀")
        if wire is None:
            return f"┤↗{postselect}{self.pauli_word}├"
        return f"┤↗{postselect}{self.pauli_word[self.wires.index(wire)]}├"

    @property
    def hash(self) -> int:
        """int: An integer hash uniquely representing the measurement."""
        return hash(
            (self.__class__.__name__, self.pauli_word, tuple(self.wires.tolist()), self.meas_uid)
        )


def _pauli_measure_impl(wires: WiresLike, pauli_word: str, postselect: int | None = None):
    """Concrete implementation of the pauli_measure primitive."""
    measurement_id = str(uuid.uuid4())
    measurement = PauliMeasure(pauli_word, wires, postselect, meas_uid=measurement_id)
    return MeasurementValue([measurement])


@lru_cache
def _create_pauli_measure_primitive():
    """Create a primitive corresponding to a Pauli product measurement."""

    # pylint: disable=import-outside-toplevel
    import jax

    from pennylane.capture.custom_primitives import QmlPrimitive

    pauli_measure_p = QmlPrimitive("pauli_measure")

    @pauli_measure_p.def_impl
    def _pauli_measure_primitive_impl(*wires, pauli_word="", postselect=None):
        wires = [w if math.is_abstract(w) else int(w) for w in wires]
        return _pauli_measure_impl(wires, pauli_word=pauli_word, postselect=postselect)

    @pauli_measure_p.def_abstract_eval
    def _pauli_measure_primitive_abstract_eval(*_, **__):
        dtype = jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        return jax.core.ShapedArray((), dtype)

    return pauli_measure_p


def pauli_measure(pauli_word: str, wires: WiresLike, postselect: int | None = None):
    """Perform a Pauli product measurement.

    A Pauli product measurement (PPM) is the measurement of a tensor product of Pauli observables (``X``, ``Y``, ``Z``, and ``I``).

    The eigenvalue of this tensor product is one of 1 or -1, which is mapped to the 0 or 1 outcome of
    the PPM, respectively. After the measurement, the state collapses to the superpositions of all
    degenerate eigenstates corresponding to the measured eigenvalue.

    .. note::

        Circuits comprising ``pauli_measure`` are currently not executable on any backend.
        This function is only for analysis using the ``null.qubit`` device and potential future execution when a suitable backend is
        available.

    .. seealso::
        For more information on Pauli product measurements, check out the `Quantum Compilation hub <https://pennylane.ai/compilation/pauli-based-computation>`_ and
        :func:`catalyst.passes.ppm_compilation` for compiling these circuits with Catalyst.

    Args:
        pauli_word (str): The Pauli word to measure.
        wires (Wires): The wires that the Pauli word acts on.
        postselect (Optional[int]): The postselection value, one of ``0`` or ``1``. It determines which subspace of
            degenerate eigenstates to postselect after a Pauli product measurement. ``None`` by default.

    Returns:
        MeasurementValue: A reference to the future result of the Pauli product measurement

    Raises:
        ValueError: if the Pauli word has characters other than X, Y and Z.
        ValueError: if the number of wires does not match the length of the Pauli word.

    **Example:**

    The following example illustrates how to include a Pauli product measurement (PPM) in a circuit by specifiying
    the Pauli word and the wires it acts on.

    .. code-block:: python

        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(2)

            ppm = qml.pauli_measure(pauli_word="XY", wires=[0, 2])
            qml.cond(ppm, qml.X)(wires=1)

            return qml.expval(qml.Z(0))

    The ``X`` operation on wire ``1`` will be applied conditionally on the value of the PPM outcome:

    >>> print(qml.draw(circuit)())
    0: ──H─╭┤↗X├────┤  <Z>
    1: ────│──────X─┤
    2: ──H─╰┤↗Y├──║─┤
             ╚════╝

    Additionally, the number of PPM operations in a circuit can be easily inspected with :func:`~.specs`
    where they are denoted as a :class:`~.ops.mid_measure.pauli_measure.PauliMeasure` gate type:

    >>> print(qml.specs(circuit)()['resources'])
    Total wire allocations: 3
    Total gates: 4
    Circuit depth: 3
    Gate types:
      Hadamard: 2
      PauliMeasure: 1
      Conditional(PauliX): 1
    Measurements:
      expval(PauliZ): 1
    """

    if capture_enabled():
        primitive = _create_pauli_measure_primitive()
        wires = (wires,) if math.shape(wires) == () else tuple(wires)
        return primitive.bind(*wires, pauli_word=pauli_word, postselect=postselect)

    return _pauli_measure_impl(wires, pauli_word, postselect)
