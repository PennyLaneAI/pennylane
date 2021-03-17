# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pennylane as qml
from pennylane import numpy as np
from itertools import product


def fourier_coefficients(f, n_inputs, degree, apply_rfftn=False):
    """Computes the first 2*degree+1 Fourier coefficients of a 2*pi periodic function.

    Args:
        f (callable): function that takes an array of N scalar inputs
        N (int): dimension of the input
        degree (int): degree up to which Fourier coeffs are to be computed
        apply_rfftn (bool): If True, call rfftn instead of fftn.

    Returns:
        (np.ndarray): The Fourier coefficients of the function f.
    """
    # number of integer values for the indices n_i = -degree,...,0,...,degree
    k = 2 * degree + 1

    # create generator for indices nvec = (n1, ..., nN), ranging from (-d,...,-d) to (d,...,d).
    n_range = np.array(range(-degree, degree + 1))
    n_ranges = [n_range] * n_inputs
    nvecs = product(*n_ranges)

    # here we will collect the discretized values of function f
    shp = tuple([k] * n_inputs)
    f_discrete = np.zeros(shape=shp)

    for nvec in nvecs:
        nvec = np.array(nvec)

        # compute the evaluation points for frequencies nvec
        sample_points = 2 * np.pi / k * nvec

        # fill discretized function array with value of f at inpts
        f_discrete[tuple(nvec)] = f(x=sample_points)

    # Now we have a discretized verison of f we can use
    # the discrete fourier transform.
    # The normalization factor is the number of discrete points (??)
    if apply_rfftn:
        coeffs = np.fft.rfftn(f_discrete) / f_discrete.size
    else:
        coeffs = np.fft.fftn(f_discrete) / f_discrete.size

    return coeffs


def frequency_spectra(tape):
    """Return the frequency spectrum of a tape that returns the expectation value of
a single quantum observable.

    .. note::

        This function currently only works with the autograd interface.

    .. note::

        This function currently only works with the branch 'fourier_spectrum' of PennyLane.

    The circuit represented by the tape can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`. This function can always be
    expressed by a Fourier series

    .. math::

        \sum \limits_{n_1\in \Omega_1} \dots \sum \limits_{n_N \in \Omega_N}
        c_{n_1,\dots, n_N} e^{-i x_1 n_1} \dots e^{-i x_N n_N}

    summing over the *frequency spectra* :math:`\Omega_i \subseteq \mathbb{Z}`
    :math:`i=1,\dots,N`, where :math:`\mathbb{Z}` are the integers. Each
    spectrum has the property that :math:`0 \in \Omega_i` and are symmetric (for
    every :math:`n \in \Omega_i` we have that :math:`-n \in \Omega_i`).

    As shown in `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`_ and XXX, the frequency spectra only
    depend on the eigenvalues of the generator :math:`G` of the gates
    :math:`e^{-ix_i G}` that the inputs enter into.  In many situations, the
    spectra are limited to a few frequencies only, which in turn limits the
    function class that the circuit can express.

    This function extracts the frequency spectra for all inputs and a circuit
    represented by a :class:`QuantumTape`.  To mark quantum circuit arguments as
    inputs, create them via:

    .. code-block:: python

        from pennylane import numpy as np

        x = np.array(0.1, is_input=True)

    The marking is independent of whether an argument is trainable or not; both
    data and trainable parameters can be interpreted as inputs to the overall
    function :math:`f`.

    Let :math:`x_i` enter :math:`J` gates with generator eigenvalue spectra
    :math:`ev_1 = \{\lambda^1_1 \dots \lambda_{T_1}^1 \}, \dots, ev_J =
    \{\lambda^1_J \dots \lambda_{T_J}^J\}`.

    The frequency spectrum :math:`\Omega_i` of input :math:`x_i` consists of all
    unique element in the set

    .. math::

        \Omega_i = \{\sum_{j=1}^J \lambda_{j} - \sum_{j'=1}^J \lambda_{j'}\}, \;
        \lambda_j, \lambda_{j'} \in ev_j

    Args:
        tape (pennylane.tape.QuantumTape): tape representing the circuit

    Returns:
        list[pennylane.numpy.tensor], list[list] : inputs and list of frequency spectra

    **Example**

    .. code-block:: python

        import pennylane as qml

        inpt = anp.array([0.1, 0.3], requires_grad=False, is_input=True)
        weights = anp.array([0.5, 0.2], requires_grad=True, is_input=False)

        with qml.tape.QuantumTape() as tape:

            qml.RX(inpt[0], wires='a')
            qml.Rot(0.1, 0.2, 0.3, wires='a')

            qml.RY(inpt[1], wires='a')
            qml.Rot(-4.1, 3.2, 1.3, wires='a')

            qml.RX(inpt[0], wires='a')
            qml.Hadamard(wires='a')

            qml.RX(weights[0], wires='a')
            qml.T(wires='a')

            qml.expval(qml.PauliZ(wires='a'))

        inpts, spectra = frequency_spectra(tape)
        for inp, spectrum in zip(inpts, spectra):
            print(inp, spectrum)
    ```

    """

    all_params = tape.get_parameters(trainable_only=False)

    # get all unique tensors marked as input
    inputs = list(
        set([p for p in all_params if (isinstance(p, np.tensor) and p.is_input)])
    )

    if not inputs:
        return [], []

    generator_evals = [[] for _ in range(len(inputs))]

    # go through operations to find generators of inputs
    for obj in tape.operations:
        if isinstance(obj, qml.operation.Operation):

            # do nothing if no gate parameter is marked as input
            if not np.any([p.is_input for p in obj.data if isinstance(p, np.tensor)]):
                continue

            if len(obj.data) > 1:
                raise ValueError(
                    "Function does not support inputs fed into multi-parameter gates."
                )
            inp = obj.data[0]

            idx = inputs.index(inp)

            # get eigenvalues of gate generator
            gen, coeff = obj.generator
            evals = np.linalg.eigvals(coeff * gen.matrix)
            # eigenvalues of hermitian ops are guaranteed to be real
            evals = np.real(evals)
            generator_evals[idx].append(evals)
            # append negative to cater for the complex conjugate part which subtracts eigenvalues
            generator_evals[idx].append(-evals)

    # for each spectrum, compute sums of all possible combinations of eigenvalues
    frequencies = [[sum(comb) for comb in product(*e)] for e in generator_evals]
    frequencies = [np.round(fs, decimals=8) for fs in frequencies]

    unique_frequencies = [sorted(set(fs)) for fs in frequencies]

    return inputs, unique_frequencies
