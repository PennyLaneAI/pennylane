# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate a quantum script for a qubit mixed state device."""
# pylint: skip-file
# black: skip-file
import pennylane as qml
from pennylane.typing import Result


def simulate(  # pylint: disable=too-many-arguments
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    readout_errors=None,
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by ``default.qubit.mixed``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
            to simulate readout errors.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.
    """
    raise NotImplementedError
