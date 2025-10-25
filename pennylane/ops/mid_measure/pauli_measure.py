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
from functools import lru_cache

from pennylane import math
from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.wires import Wires, WiresLike

from .measurement_value import MeasurementValue

_VALID_PAULI_CHARS = "XYZ"


class PauliMeasure(Operator):
    """A Pauli product measurement."""

    resource_keys = {"pauli_word"}

    def __init__(
        self,
        pauli_word: str,
        wires: WiresLike,
        postselect: int | None = None,
        id: str | None = None,
    ):
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
        super().__init__(wires=wires, id=id)
        self.hyperparameters["pauli_word"] = pauli_word
        self.hyperparameters["postselect"] = postselect

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

    @property
    def resource_params(self) -> dict:
        return {"pauli_word": self.hyperparameters["pauli_word"]}

    @property
    def hash(self) -> int:
        """int: An integer hash uniquely representing the measurement."""
        return hash((self.__class__.__name__, self.pauli_word, tuple(self.wires.tolist()), self.id))


def _pauli_measure_impl(wires: WiresLike, pauli_word: str, postselect: int | None = None):
    """Concrete implementation of the pauli_measure primitive."""
    measurement_id = str(uuid.uuid4())
    measurement = PauliMeasure(pauli_word, wires, postselect, measurement_id)
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
        return _pauli_measure_impl(wires, pauli_word=pauli_word, postselect=postselect)

    @pauli_measure_p.def_abstract_eval
    def _pauli_measure_primitive_abstract_eval(*_, **__):
        dtype = jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        return jax.core.ShapedArray((), dtype)

    return pauli_measure_p


def pauli_measure(pauli_word: str, wires: WiresLike, postselect: int | None = None):
    """Perform a Pauli product measurement."""

    if capture_enabled():
        primitive = _create_pauli_measure_primitive()
        wires = (wires,) if math.shape(wires) == () else tuple(wires)
        return primitive.bind(*wires, pauli_word=pauli_word, postselect=postselect)

    return _pauli_measure_impl(wires, pauli_word, postselect)
