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
"""Contains a QNode transform that computes the Fourier spectrum of a quantum
circuit."""
from itertools import product
from functools import wraps
import numpy as np
import pennylane as qml


def _get_spectrum(op):
    r"""Extract the frequencies contributed by a data-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the data-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.

    Args:
        op (~pennylane.operation.Operation): an instance of the `Operation` class

    Returns:
        list: frequencies contributed by this data-encoding gate
    """
    evals = None
    g, coeff = op.generator

    # the generator is undefined
    if g is None:
        raise ValueError(
            "no generator defined for data-encoding gate {}; "
            "cannot extract Fourier spectrum".format(op.name)
        )

    # if g is an Operator instance ("PauliX(wires=0)") or class ("PauliX"),
    # extract its eigenvalues or matrix representation
    if hasattr(g, "eigvals"):
        # first try if we can directly find eigenvalues
        evals = g.eigvals

        if evals is not None:
            # scale evals correctly
            evals = evals * coeff
    elif hasattr(g, "matrix"):
        # try to extract the matrix
        g = g.matrix

        # if this also fails, we need to abort
        if g.matrix is None:
            raise ValueError(
                "no matrix or eigenvalues defined for generator {} of data-encoding gate {}; "
                "cannot extract Fourier spectrum".format(g, op.name)
            )

    # if we have to use the matrix representation,
    # extract the eigenvalues from the matrix
    if evals is None:
        g = coeff * g
        # eigenvalues of hermitian ops are guaranteed to be real
        evals = qml.math.real(np.linalg.eigvals(g))

    # compute all differences of eigenvalues
    frequencies = [np.round(e1 - e2, decimals=8) for e1, e2 in product(evals, evals)]
    unique_frequencies = list(set(frequencies))
    return sorted(unique_frequencies)


def _join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input scalar.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, frequency sets of two gates
    encoding the same :math:`x` are joined by computing the set of sums of their elements.

    Args:
        spec1 (list[float]): first spectrum
        spec2 (list[float]): second spectrum
    Returns:
        list[float]: joined spectrum
    """
    sums = [s1 + s2 for s1 in spec1 for s2 in spec2]
    return sorted(list(set(sums)))


def spectrum(qnode, encoding_gates=None):
    r"""Compute the frequency spectrum of the Fourier representation of simple quantum circuits.

    The circuit must only use single-parameter gates of the form :math:`e^{-ix_j G}` as
    data-encoding gates, which allows the computation of the spectrum by inspecting the gates'
    generators :math:`G`.

    Gates are marked as data-encoding gates on the tape by giving them an id which is
    found in encoding_gates. If two gates have the same id, they are considered
    to be used to encode the same input :math:`x_j`. If encoding_gates is None,
    all gates on the tape which have a custom id attribute are considered to be input-encoding gates.

    Args:
        qnode (pennylane.QNode): a quantum node representing a circuit in which
            input-encoding gates are marked by IDs
        encoding_gates (list[str]): list of input-encoding gate IDs
            for which to compute the frequency spectra

    Returns:
        (Dict[str, list[float]]): Dictionary with the input scalars' gate IDs as keys and
            their frequency spectra as values.

    **Details**

    A circuit that returns an expectation value which depends on
    :math:`N` scalar inputs :math:`x_i` can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. This function can always be
    expressed by a Fourier-type sum

    .. math::

        \sum \limits_{\omega_1\in \Omega_1} \dots \sum \limits_{\omega_N \in \Omega_N}
        c_{\omega_1,\dots, \omega_N} e^{-i x_1 \omega_1} \dots e^{-i x_N \omega_N}

    over the *frequency spectra* :math:`\Omega_i \subseteq \mathbb{R},`
    :math:`i=1,\dots,N`. Each
    spectrum has the property that :math:`0 \in \Omega_i`, and the spectrum is
    symmetric (for every :math:`\omega \in \Omega_i` we have that :math:`-\omega \in
    \Omega_i`). If all frequencies are integer-valued, the Fourier sum becomes a
    *Fourier series*.

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

        res = spectrum(circuit, encoding_gates)(x, w)

        for inp, freqs in res.items():
            print(f"{inp}: {freqs}")

        >>> 'x1': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> 'x2': [-2.0, -1.0, 0.0, 1.0, 2.0]
        >>> 'x3': [-2.0, -1.0, 0.0, 1.0, 2.0]

    .. note::
        While the Fourier spectrum usually does not depend
        on the actual circuit parameters and inputs, it may still change with
        arguments of the QNode, for example if the arguments change the architecture
        of the circuit.

    .. note::
        The `spectrum` function does not check if the result of the
        circuit is an expectation value.
    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        tape = qnode.qtape

        freqs = {}
        for op in tape.operations:
            id = op.id

            # if the operator has no specific ID,
            # move to the next
            if id is None:
                continue

            # if user has not specified encoding_gate id's,
            # consider any id
            is_encoding_gate = encoding_gates is None or id in encoding_gates
            if is_encoding_gate:

                if len(op.parameters) != 1:
                    raise ValueError(
                        "can only consider one-parameter gates as data-encoding gates; "
                        "got {}.".format(op.name)
                    )

                spec = _get_spectrum(op)

                # if id has been seen before,
                # join this spectrum to another one
                if id in freqs:
                    spec = _join_spectra(freqs[id], spec)

                freqs[id] = spec

        return freqs

    return wrapper
