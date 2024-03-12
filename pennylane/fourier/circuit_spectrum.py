# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a transform that computes the simple frequency spectrum
of a quantum circuit, that is the frequencies without considering
preprocessing in the QNode."""
from typing import Sequence, Callable
from functools import partial
from pennylane import transform
from pennylane.tape import QuantumTape
from .utils import get_spectrum, join_spectra


@partial(transform, is_informative=True)
def circuit_spectrum(
    tape: QuantumTape, encoding_gates=None, decimals=8
) -> (Sequence[QuantumTape], Callable):
    r"""Compute the frequency spectrum of the Fourier representation of
    simple quantum circuits ignoring classical preprocessing.

    The circuit must only use simple single-parameter gates of the form :math:`e^{-i x_j G}` as
    input-encoding gates, which allows the computation of the spectrum by inspecting the gates'
    generators :math:`G`. The most important example of such gates are Pauli rotations.

    .. note::

        More precisely, the ``circuit_spectrum`` function relies on the gate to
        define a ``generator``, and will fail if gates marked as inputs do not
        have this attribute.

    Gates are marked as input-encoding gates in the quantum function by giving them an ``id``.
    If two gates have the same ``id``, they are considered
    to be used to encode the same input :math:`x_j`. The ``encoding_gates`` argument can be used
    to indicate that only gates with a specific ``id`` should be interpreted as input-encoding gates.
    Otherwise, all gates with an explicit ``id`` are considered to be input-encoding gates.

    .. note::
        If no input-encoding gates are found, an empty dictionary is returned.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit in which
            input-encoding gates are marked by their ``id`` attribute
        encoding_gates (list[str]): list of input-encoding gate ``id`` strings
            for which to compute the frequency spectra
        decimals (int): number of decimals to which to round frequencies.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will return a dictionary with the input-encoding gate ``id`` as keys and their frequency spectra as values.


    **Details**

    A circuit that returns an expectation value which depends on
    :math:`N` scalar inputs :math:`x_j` can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. This function can always be
    expressed by a Fourier-type sum

    .. math::

        \sum \limits_{\omega_1\in \Omega_1} \dots \sum \limits_{\omega_N \in \Omega_N}
        c_{\omega_1,\dots, \omega_N} e^{-i x_1 \omega_1} \dots e^{-i x_N \omega_N}

    over the *frequency spectra* :math:`\Omega_j \subseteq \mathbb{R},`
    :math:`j=1,\dots,N`. Each spectrum has the property that
    :math:`0 \in \Omega_j`, and the spectrum is
    symmetric (for every :math:`\omega \in \Omega_j` we have that :math:`-\omega \in
    \Omega_j`). If all frequencies are integer-valued, the Fourier sum becomes a
    *Fourier series*.

    As shown in `Vidal and Theis (2019)
    <https://arxiv.org/abs/1901.11434>`_ and `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`_, if an input :math:`x_j, j = 1 \dots N`,
    only enters into single-parameter gates of the form :math:`e^{-i x_j G}` (where :math:`G` is a Hermitian generator),
    the frequency spectrum :math:`\Omega_j` is fully determined by the eigenvalues
    of :math:`G`. In many situations, the spectra are limited
    to a few frequencies only, which in turn limits the function class that the circuit
    can express.

    The ``circuit_spectrum`` function computes all frequencies that will potentially appear in the
    sets :math:`\Omega_1` to :math:`\Omega_N`.

    **Example**

    Consider the following example, which uses non-trainable inputs ``x`` and
    trainable parameters ``w`` as arguments to the qnode.

    .. code-block:: python

        import pennylane as qml
        import numpy as np

        n_layers = 2
        n_qubits = 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, w):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(x[i], wires=i, id="x"+str(i))
                    qml.Rot(w[l,i,0], w[l,i,1], w[l,i,2], wires=i)
            qml.RZ(x[0], wires=0, id="x0")
            return qml.expval(qml.Z(0))

        x = np.array([1, 2, 3])
        w = np.random.random((n_layers, n_qubits, 3))
        res = qml.fourier.circuit_spectrum(circuit)(x, w)

    >>> print(qml.draw(circuit)(x, w))
    0: ──RX(1.00)──Rot(0.53,0.70,0.90)──RX(1.00)──Rot(0.81,0.38,0.43)──RZ(1.00)─┤  <Z>
    1: ──RX(2.00)──Rot(0.56,0.61,0.96)──RX(2.00)──Rot(0.32,0.49,0.77)───────────┤
    2: ──RX(3.00)──Rot(0.11,0.63,0.31)──RX(3.00)──Rot(0.52,0.46,0.83)───────────┤

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    'x0': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    'x1': [-2.0, -1.0, 0.0, 1.0, 2.0]
    'x2': [-2.0, -1.0, 0.0, 1.0, 2.0]

    .. note::
        While the Fourier spectrum usually does not depend
        on trainable circuit parameters or the actual values of the inputs,
        it may still change based on inputs to the QNode that alter the architecture
        of the circuit.

    The input-encoding gates to consider can also be explicitly selected by using the
    ``encoding_gates`` keyword argument:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0, id="x0")
            qml.PhaseShift(x[0], wires=0, id="x0")
            qml.RX(x[1], wires=0, id="x1")
            return qml.expval(qml.Z(0))

        x = np.array([1, 2])
        res = qml.fourier.circuit_spectrum(circuit, encoding_gates=["x0"])(x)

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    'x0': [-2.0, -1.0, 0.0, 1.0, 2.0]

    .. note::
        The ``circuit_spectrum`` function does not check if the result of the
        circuit is an expectation, or if gates with the same ``id``
        take the same value in a given call of the function.

    The ``circuit_spectrum`` function works in all interfaces:

    .. code-block:: python

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface='tf')
        def circuit(x):
            qml.RX(x[0], wires=0, id="x0")
            qml.PhaseShift(x[1], wires=0, id="x1")
            return qml.expval(qml.Z(0))

        x = tf.constant([1, 2])
        res = qml.fourier.circuit_spectrum(circuit)(x)

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    'x0': [-1.0, 0.0, 1.0]
    'x1': [-1.0, 0.0, 1.0]

    """

    def processing_fn(tapes):
        """Process the tapes extract the spectrum of the circuit."""
        tape = tapes[0]
        freqs = {}
        for op in tape.operations:
            id = op.id

            # if the operator has no specific ID, move to the next
            if id is None:
                continue

            # if user has not specified encoding_gate id's,
            # consider any id
            is_encoding_gate = encoding_gates is None or id in encoding_gates

            if is_encoding_gate:
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as "
                        f"data-encoding gates; got {op.name}."
                    )

                spec = get_spectrum(op, decimals=decimals)

                # if id has been seen before, join this spectrum to another one
                if id in freqs:
                    spec = join_spectra(freqs[id], spec)

                freqs[id] = spec

        # Turn spectra into sorted lists and include negative frequencies
        for id, spec in freqs.items():
            spec = sorted(spec)
            freqs[id] = [-f for f in spec[:0:-1]] + spec

        # Add trivial spectrum for requested gate ids that are not in the circuit
        if encoding_gates is not None:
            for id in set(encoding_gates).difference(freqs):
                freqs[id] = []

        return freqs

    return [tape], processing_fn
