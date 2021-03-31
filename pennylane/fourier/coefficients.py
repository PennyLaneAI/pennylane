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
"""Contains methods for computing Fourier coefficients and frequency spectra of quantum functions ."""
from itertools import product

import pennylane as qml
from pennylane import numpy as np

from .utils import extract_evals
from .custom_decompositions import *

custom_decomps_required = {"CRot": custom_CRot_decomposition}


def frequency_spectra(tape):
    r"""Return the frequency spectrum of a tape that returns the expectation value of
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

    summing over the *frequency spectra* :math:`\Omega_i \subseteq \mathbb{Z},`
    :math:`i=1,\dots,N`, where :math:`\mathbb{Z}` are the integers. Each
    spectrum has the property that :math:`0 \in \Omega_i`, and the spectrum is
    symmetric (for every :math:`n \in \Omega_i` we have that :math:`-n \in
    \Omega_i`).

    As shown in `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`_ and XXX, the frequency spectra only
    depend on the eigenvalues of the generator :math:`G` of the gates
    :math:`e^{-ix_i G}` that the inputs enter into.  In many situations, the
    spectra are limited to a few frequencies only, which in turn limits the
    function class that the circuit can express.

    This function extracts the frequency spectra for all inputs and a circuit
    represented by a :class:`~.pennylane.QNode`.  To mark quantum circuit arguments as
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

        \Omega_i = \left\{\sum_{j=1}^J \lambda_{j} - \sum_{j'=1}^J \lambda_{j'}\right\}, \;
        \lambda_j, \lambda_{j'} \in ev_j

    Args:
        tape (pennylane.tape.QuantumTape): a quantum tape representing the circuit

    Returns:
        (Dict[pennylane.numpy.tensor, list[float]]): Dictionary of input keys
        with a list of their frequency spectra.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as anp

        inpt = anp.array([0.1, 0.3], requires_grad=False, is_input=True)
        weights = anp.array([0.5, 0.2], requires_grad=True, is_input=False)

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(weights, inpt):
            qml.RX(inpt[0], wires='a')
            qml.Rot(0.1, 0.2, 0.3, wires='a')

            qml.RY(inpt[1], wires='a')
            qml.Rot(-4.1, 3.2, 1.3, wires='a')

            qml.RX(inpt[0], wires='a')
            qml.Hadamard(wires='a')

            qml.RX(weights[0], wires='a')
            qml.T(wires='a')

            return qml.expval(qml.PauliZ(wires='a'))

        circuit(weights, inpt)

        frequencies = frequency_spectra(circuit.qtape)

        for inp, freqs in frequencies.items():
            print(f"{inp}: {freqs}")

    """

    all_params = tape.get_parameters(trainable_only=False)

    # get all unique tensors marked as input
    inputs = list(set([p for p in all_params if (isinstance(p, np.tensor) and p.is_input)]))

    if not inputs:
        return {}

    # We will now go through the circuit, and collect the set of gates relevant
    # to each input.
    generator_evals = [[] for _ in inputs]

    for obj in tape.operations:
        if isinstance(obj, qml.operation.Operation):
            # Ignore operations with no parameters, or with no parameters marked as inputs
            if obj.num_params == 0:
                continue

            tensor_parameters = [p for p in obj.data if isinstance(p, np.tensor)]

            if not np.any([p.is_input for p in tensor_parameters]):
                continue

            # Collect the set of inputs on this gate
            relevant_inputs = [inputs.index(p) for p in tensor_parameters if p.is_input]

            # It could be the case that more than one parameter of a given gate is
            # an input; ignore this case for now.
            if len(relevant_inputs) > 1:
                raise ValueError("Function does not support inputs fed into multi-parameter gates.")

            # Get the index of this particular input
            input_idx = relevant_inputs[0]

            # Check if this is a gate should be unrolled into Pauli operators
            try:
                if obj.name in custom_decomps_required.keys():
                    decomp = custom_decomps_required[obj.name](*obj.parameters, wires=obj.wires)
                else:
                    decomp = obj.decomposition(*obj.parameters, wires=obj.wires)
            except NotImplementedError:
                # If no decomposition required, simply get the eigenvalues of gate generator
                evals = extract_evals(obj)
                generator_evals[input_idx].extend(evals)
                continue

            # If decomposition is required, need to get eigenvalues for each gate in
            # the decomposition that uses this parameter
            for decomp_op in decomp:
                if np.any([p.is_input for p in decomp_op.data if isinstance(p, np.tensor)]):
                    evals = extract_evals(decomp_op)
                    generator_evals[input_idx].extend(evals)

    # for each spectrum, compute sums of all possible combinations of eigenvalues
    frequencies = [[sum(comb) for comb in product(*e)] for e in generator_evals]
    frequencies = [np.round(fs, decimals=8) for fs in frequencies]

    unique_frequencies = [sorted(set(fs)) for fs in frequencies]

    # Convert to a dictionary of parameter values and integers
    frequency_dict = {
        inp.base.take(0): [f.base.take(0) for f in freqs]
        for inp, freqs in zip(inputs, unique_frequencies)
    }

    return frequency_dict


def fourier_coefficients(f, n_inputs, degree, lowpass_filter=True, filter_threshold=None):
    """Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi`
    periodic function, where :math:`d` is the highest desired frequency in the
    Fourier spectrum.

    By default, a low-pass filter is applied prior to computing the coefficients
    in order to mitigate the effects of aliasing. Coefficients up to a threshold
    value are computed, and then frequencies higher than the degree are simply removed. This
    ensures that the coefficients returned will have the correct values, though they
    may not be the full set of coefficients. If no threshold value is provided, the
    threshold will be set to ``2 * degree``.

    Args:
        f (callable): function that takes an array of :math:`N` scalar inputs
        n_inputs (int): number of function inputs
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d ` will be computed.
        lowpass_filter (bool): If True (default), a simple low-pass filter is applied prior to
            computing the set of coefficients in order to filter out frequencies above the
            given degree.
        filter_threshold (None or int): The integer frequency at which to filter. If no value is
            specified, ``2 * degree`` is used.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as anp

        # Expected Fourier series over 2 parameters with frequencies 0 and 1
        num_inputs = 2
        degree = 1

        weights = anp.array([0.5, 0.2], requires_grad=True, is_input=False)

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(weights, inpt):
            qml.RX(inpt[0], wires='a')
            qml.Rot(0.1, 0.2, 0.3, wires='a')

            qml.RY(inpt[1], wires='a')
            qml.Rot(-4.1, 3.2, 1.3, wires='a')

            return qml.expval(qml.PauliZ(wires='a'))

        # Coefficients of the "inpt" variable will be computed
        coeffs = fourier_coefficients(partial(circuit, weights), num_inputs, degree)

    """
    if not lowpass_filter:
        return _fourier_coefficients_no_filter(f, n_inputs, degree)

    if filter_threshold is None:
        filter_threshold = 2 * degree

    # Compute the fft of the function at 2x the specified degree
    unfiltered_coeffs = _fourier_coefficients_no_filter(f, n_inputs, filter_threshold)

    # Shift the frequencies so that the 0s are at the centre
    shifted_unfiltered_coeffs = np.fft.fftshift(unfiltered_coeffs)

    # Next, slice up the array so that we get only the coefficients we care about,
    # those between -degree and degree
    range_slices = list(
        range(
            filter_threshold - degree,
            shifted_unfiltered_coeffs.shape[0] - (filter_threshold - degree),
        )
    )

    shifted_filtered_coeffs = shifted_unfiltered_coeffs.copy()

    # Go axis by axis and take only the central components
    for axis in range(n_inputs - 1, -1, -1):
        shifted_filtered_coeffs = np.take(shifted_filtered_coeffs, range_slices, axis=axis)

    # Shift everything back into "normal" fft ordering
    filtered_coeffs = np.fft.ifftshift(shifted_filtered_coeffs)

    # Compute the inverse FFT
    f_discrete_filtered = np.fft.ifftn(filtered_coeffs)

    # Now compute the FFT again on the filtered data
    coeffs = np.fft.fftn(f_discrete_filtered)

    return coeffs


def _fourier_coefficients_no_filter(f, n_inputs, degree):
    """Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``fourier_coefficients`` function.

    Args:
        f (callable): function that takes an array of :math:`N` scalar inputs
        n_inputs (int): number of function inputs
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d ` will be computed.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.
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
        f_discrete[tuple(nvec)] = f(sample_points)

    coeffs = np.fft.fftn(f_discrete) / f_discrete.size

    return coeffs
