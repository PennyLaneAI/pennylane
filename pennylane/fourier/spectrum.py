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
"""Contains a transform that computes the frequency spectrum of a quantum
circuit."""
from itertools import chain, combinations
from functools import wraps
import numpy as np
import pennylane as qml


def _get_spectrum(op):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.

    Args:
        op (~pennylane.operation.Operation): an instance of the `Operation` class

    Returns:
        list: frequencies contributed by this input-encoding gate
    """
    no_generator = False
    if hasattr(op, "generator"):
        g, coeff = op.generator

        if isinstance(g, np.ndarray):
            matrix = g
        elif hasattr(g, "matrix"):
            matrix = g.matrix
        else:
            no_generator = True
    else:
        no_generator = True

    if no_generator:
        raise ValueError(f"Generator of operation {op} is not defined.")

    matrix = coeff * matrix
    # eigenvalues of hermitian ops are guaranteed to be real
    # todo: use qml.math.linalg once it is tested properly
    evals = qml.math.real(np.linalg.eigvals(matrix))

    # compute all differences of eigenvalues
    unique_frequencies = set(
        chain.from_iterable(
            np.round((x[1] - x[0], x[0] - x[1]), decimals=8) for x in combinations(evals, 2)
        )
    )
    unique_frequencies = unique_frequencies.union({0})
    return sorted(unique_frequencies)


def _join_spectra(spec1, spec2):
    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, frequency sets of two gates
    encoding the same :math:`x` are joined by computing the set of sums of their elements.

    Args:
        spec1 (list[float]): first spectrum
        spec2 (list[float]): second spectrum
    Returns:
        list[float]: joined spectrum
    """
    if spec1 == []:
        return sorted(set(spec2))
    if spec2 == []:
        return sorted(set(spec1))

    sums = [s1 + s2 for s1 in spec1 for s2 in spec2]
    return sorted(set(sums))


def spectrum(qnode, encoding_gates=None):
    r"""Compute the frequency spectrum of the Fourier representation of simple quantum circuits.

    The circuit must only use simple single-parameter gates of the form :math:`e^{-i x_j G}` as
    input-encoding gates, which allows the computation of the spectrum by inspecting the gates'
    generators :math:`G`. The most important example of such gates are Pauli rotations.

    .. note::

        More precisely, the spectrum function relies on the gate to define a ``generator``, and will
        fail if gates marked as inputs do not have this attribute.

    Gates are marked as input-encoding gates in the quantum function by giving them an ``id``.
    If two gates have the same ``id``, they are considered
    to be used to encode the same input :math:`x_j`. The ``encoding_gates`` argument can be used
    to indicate that only gates with a specific ``id`` should be interpreted as input-encoding gates.
    Otherwise, all gates with an explicit ``id`` are considered to be input-encoding gates.

    .. note::
        If no input-encoding gates are found, an empty dictionary is returned.

    Args:
        qnode (pennylane.QNode): a quantum node representing a circuit in which
            input-encoding gates are marked by their ``id`` attribute
        encoding_gates (list[str]): list of input-encoding gate ``id`` strings
            for which to compute the frequency spectra

    Returns:
        (dict[str, list[float]]): Dictionary with the input-encoding gate ``id`` as keys and
            their frequency spectra as values.


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

    The ``spectrum`` function computes all frequencies that will potentially appear in the
    sets :math:`\Omega_1` to :math:`\Omega_N`.

    **Example**

    Consider the following example, which uses non-trainable inputs ``x`` and
    trainable parameters ``w`` as arguments to the qnode.

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
                    qml.Rot(w[l,i,0], w[l,i,1], w[l,i,2], wires=0)
            qml.RZ(x[0], wires=0, id="x0")
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([1, 2, 3])
        w = np.random.random((n_layers, n_qubits, 3))
        res = spectrum(circuit)(x, w)

    >>> print(qml.draw(circuit)(x, w))
    0: ──RX(1)──Rot(0.863, 0.611, 0.281)───RX(1)──Rot(0.47, 0.158, 0.648)───RZ(1)──┤ ⟨Z⟩
    1: ──RX(2)──Rot(0.0781, 0.971, 0.457)──RX(2)──Rot(0.896, 0.224, 0.731)─────────┤
    2: ──RX(3)──Rot(0.462, 0.286, 0.929)───RX(3)──Rot(0.879, 0.399, 0.215)─────────┤

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
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([1, 2])
        res = spectrum(circuit, encoding_gates=["x0"])(x)

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    'x0': [-2.0, -1.0, 0.0, 1.0, 2.0]

    .. note::
        The ``spectrum`` function does not check if the result of the
        circuit is an expectation, or if gates with the same ``id``
        take the same value in a given call of the function.

    The ``spectrum`` function works in all interfaces:

    .. code-block:: python

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface='tf')
        def circuit(x):
            qml.RX(x[0], wires=0, id="x0")
            qml.PhaseShift(x[1], wires=0, id="x1")
            return qml.expval(qml.PauliZ(wires=0))

        x = tf.constant([1, 2])
        res = spectrum(circuit)(x)

    >>> for inp, freqs in res.items():
    >>>     print(f"{inp}: {freqs}")
    'x0': [-1.0, 0.0, 1.0]
    'x1': [-1.0, 0.0, 1.0]

    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        tape = qnode.qtape

        if encoding_gates is None:
            freqs = {}
        else:
            freqs = {input_id: [] for input_id in encoding_gates}

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
                        "Can only consider one-parameter gates as data-encoding gates; "
                        f"got {op.name}."
                    )

                spec = _get_spectrum(op)

                # if id has been seen before,
                # join this spectrum to another one
                if id in freqs:
                    spec = _join_spectra(freqs[id], spec)

                freqs[id] = spec

        return freqs

    return wrapper
