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
"""Contains a function that computes the fourier spectrum of a tape."""
import numpy as np
from itertools import product


def _get_spectrum(op):
    r"""Get the spectrum of the input x of a single-parameter operation :math:`\exp(-i x G)`.

    The spectrum is the set of sums of :math:`G`'s eigenvalues.
    """

    g, coeff = op.generator

    # some generators are operations
    if not isinstance(g, np.ndarray) and g is not None:
        g = g.matrix

    # the matrix or generator could be undefined
    if g is None:
        raise ValueError("cannot get spectrum for data-encoding gate {}".format(op.name))

    g = coeff * g
    evals = np.linalg.eigvals(g)
    # eigenvalues of hermitian ops are guaranteed to be real
    evals = np.real(evals)

    # append negative to cater for the
    # complex conjugate part of the expectation,
    # which subtracts eigenvalues
    evals = [evals, -evals]

    frequencies = [np.round(sum(comb), decimals=8) for comb in product(*evals)]
    unique_frequencies = list(set(frequencies))
    return sorted(unique_frequencies)


def _join_spectra(spec1, spec2):
    r"""Join two spectra that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, spectra are
    joined by taking the set of sums of their elements.

    Args:
        spec1 (list[float]): first spectrum
        spec2 (list[float]): second spectrum
    Returns:
        list[float]: joined spectrum
    """
    sums = [s1 + s2 for s1 in spec1 for s2 in spec2]
    return sorted(list(set(sums)))


def spectrum(tape, encoding_gates=None):
    r"""Compute the frequency spectrum of the quantum circuit represented by a quantum tape.

    EXPLAIN MARKING

    EXPLAIN QNODE FORM

    Args:
        tape (pennylane.tapes.QuantumTape): a quantum tape in which data-encoding
            gates are marked

    Returns:
        (Dict[str, list[float]]): Dictionary with the input scalars' gate IDs as keys and
            their frequency spectra as values.

    **Details**

    If the circuit represented by the qnode returns the expectation value of an
    observable, it can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. This function can always be
    expressed by a Fourier series

    .. math::

        \sum \limits_{n_1\in \Omega_1} \dots \sum \limits_{n_N \in \Omega_N}
        c_{n_1,\dots, n_N} e^{-i x_1 n_1} \dots e^{-i x_N n_N}

    summing over the *frequency spectra* :math:`\Omega_i \subseteq \mathbb{Z},`
    :math:`i=1,\dots,N`, where :math:`\mathbb{Z}` are the integers. Each
    spectrum has the property that :math:`0 \in \Omega_i`, and the spectrum is
    symmetric (for every :math:`n \in \Omega_i` we have that :math:`-n \in
    \Omega_i`).

    As shown in `Vidal and Theis (2019)
    <https://arxiv.org/abs/1901.11434>`_ and `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`_, if an input :math:`x_j, j = 1 \dots N`
    only enters into single-parameter gates of the form :math:`e^{-i x_j G}`, the
    frequency spectrum :math:`\Omega_j` is fully determined by the eigenvalues
    of the generators :math:`G`. In many situations, the spectra are limited
    to a few frequencies only, which in turn limits the function class that the circuit
    can express.


    **Example**

    .. code-block:: python

        import pennylane as qml
        import numpy as np
        from pennylane.fourier import spectrum

        n_layers = 2
        n_qubits = 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, w):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(x[i], wires=0, id="x"+str(i))
                    qml.Rot(w[l,i,0], w[l,i,1], w[l,i,3], wires=0)
            qml.RZ(x[0], wires=0, id="x0")
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([1, 2, 3])
        w = np.random.random((n_layers, n_qubits, 3))

        circuit(x, w)

        res = spectrum(circuit.qtape)

        for inp, freqs in res.items():
            print(f"{inp}: {freqs}")

        >>> 'x1': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> 'x2': [-2.0, -1.0, 0.0, 1.0, 2.0]
        >>> 'x3': [-2.0, -1.0, 0.0, 1.0, 2.0]
    """

    freqs = {}
    for op in tape.operations:
        id = op.id

        # if the operator has no specific ID,
        # move to the next
        if id is None:
            continue

        # if user has not specified encoding gate id's,
        # consider any id
        is_encoding_gate = encoding_gates is None or id in encoding_gates
        if is_encoding_gate:

            if len(op.parameters) != 1:
                raise ValueError("Spectrum function can only consider one-parameter gates as data-encoding gates; "
                                 "got {}.".format(op.name))

            spec = _get_spectrum(op)

            # if id has been seen before,
            # join this spectrum to another one
            if id in freqs:
                spec = _join_spectra(freqs[id], spec)

            freqs[id] = spec

    return freqs


