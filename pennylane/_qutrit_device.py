# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`QutritDevice` abstract base class.
"""

# For now, arguments may be different from the signatures provided in QubitDevice to minimize size of pull request
# e.g. instead of expval(self, observable, wires, par) have expval(self, observable)
# pylint: disable=arguments-differ, abstract-method, no-value-for-parameter,too-many-instance-attributes,too-many-branches, no-member, bad-option-value, arguments-renamed
import itertools

import numpy as np

import pennylane as qml
from pennylane import QubitDevice
from pennylane.measurements import MeasurementProcess
from pennylane.wires import Wires


class QutritDevice(QubitDevice):  # pylint: disable=too-many-public-methods
    """Abstract base class for PennyLane qutrit devices.

    The following abstract method **must** be defined:

    * :meth:`~.apply`: append circuit operations, compile the circuit (if applicable),
      and perform the quantum computation.

    Devices that generate their own samples (such as hardware) may optionally
    overwrite :meth:`~.probability`. This method otherwise automatically
    computes the probabilities from the generated samples, and **must**
    overwrite the following method:

    * :meth:`~.generate_samples`: Generate samples from the device from the
      exact or approximate probability distribution.

    Analytic devices **must** overwrite the following method:

    * :meth:`~.analytic_probability`: returns the probability or marginal probability from the
      device after circuit execution. :meth:`~.marginal_prob` may be used here.

    This device contains common utility methods for qutrit-based devices. These
    do not need to be overwritten. Utility methods include:

    * :meth:`~.expval`, :meth:`~.var`, :meth:`~.sample`: return expectation values,
      variances, and samples of observables after the circuit has been rotated
      into the observable eigenbasis.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        r_dtype: Real floating point precision type.
        c_dtype: Complex floating point precision type.
    """

    # TODO: Update set of supported observables as new observables are added
    observables = {
        "Identity",
        "THermitian",
    }

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(model="qutrit")
        return capabilities

    def generate_samples(self):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant trit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        number_of_states = 3**self.num_wires

        rotated_prob = self.analytic_probability()

        samples = self.sample_basis_states(number_of_states, rotated_prob)

        return self.states_to_ternary(samples, self.num_wires)

    def generate_basis_states(self, num_wires, dtype=np.uint32):
        """Generates basis states in ternary representation according to the number
        of wires specified.

        Args:
            num_wires (int): the number of wires
            dtype=np.uint32 (type): the data type of the arrays to use

        Returns:
            array[int]: the sampled basis states
        """
        basis_states_generator = itertools.product((0, 1, 2), repeat=num_wires)
        return np.fromiter(itertools.chain(*basis_states_generator), dtype=dtype).reshape(
            -1, num_wires
        )

    @staticmethod
    def states_to_ternary(samples, num_wires, dtype=np.int64):
        """Convert basis states from base 10 to ternary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (array[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qutrits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            array[int]: basis states in ternary representation
        """
        ternary_arr = []
        for sample in samples:
            num = []
            for _ in range(num_wires):
                sample, r = divmod(sample, 3)
                num.append(r)

            ternary_arr.append(num[::-1])

        return np.array(ternary_arr, dtype=dtype)

    def density_matrix(self, wires):
        """Returns the reduced density matrix prior to measurement.

        Args:
            wires (Wires): wires of the reduced system

        Raises:
            QuantumFunctionError: density matrix is currently unsupported on :class:`~.QutritDevice`
        """
        # TODO: Add support for DensityMatrix return type. Currently, qml.math is hard coded to calculate this for qubit
        # states (see `qml.math.reduced_dm()`), so it needs to be updated before DensityMatrix can be supported for qutrits.
        # For now, if a user tries to request this return type, an error will be raised.
        raise qml.QuantumFunctionError(
            "Unsupported return type specified for observable density matrix"
        )

    def vn_entropy(self, wires, log_base):
        r"""Returns the Von Neumann entropy prior to measurement.

        .. math::
            S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

        Args:
            wires (Wires): Wires of the considered subsystem.
            log_base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

        Raises:
            QuantumFunctionError: Von Neumann entropy is currently unsupported on :class:`~.QutritDevice`
        """
        # TODO: Add support for VnEntropy return type. Currently, qml.math is hard coded to calculate this for qubit
        # states (see `qml.math.vn_entropy()`), so it needs to be updated before VnEntropy can be supported for qutrits.
        # For now, if a user tries to request this return type, an error will be raised.
        raise qml.QuantumFunctionError(
            "Unsupported return type specified for observable Von Neumann entropy"
        )

    def mutual_info(self, wires0, wires1, log_base):
        r"""Returns the mutual information prior to measurement:

        .. math::

            I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

        where :math:`S` is the von Neumann entropy.

        Args:
            wires0 (Wires): wires of the first subsystem
            wires1 (Wires): wires of the second subsystem
            log_base (float): base to use in the logarithm

        Raises:
            QuantumFunctionError: Mutual information is currently unsupported on :class:`~.QutritDevice`
        """
        # TODO: Add support for MutualInfo return type. Currently, qml.math is hard coded to calculate this for qubit
        # states (see `qml.math.mutual_info()`), so it needs to be updated before MutualInfo can be supported for qutrits.
        # For now, if a user tries to request this return type, an error will be raised.
        raise qml.QuantumFunctionError(
            "Unsupported return type specified for observable mutual information"
        )

    def classical_shadow(self, obs, circuit):
        """
        Returns the measured trits and recipes in the classical shadow protocol.

        Please refer to :func:`~.pennylane.classical_shadow` for detailed documentation.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow measurement process
            circuit (~.tapes.QuantumTape): The quantum tape that is being executed

        Raises:
            QuantumFunctionError: Classical shadow is currently unsupported on :class:`~.QutritDevice`
        """
        # TODO: Add support for ClassicalShadowMP
        raise qml.QuantumFunctionError(
            "Qutrit devices don't support classical shadow measurements."
        )

    def shadow_expval(self, obs, circuit):
        r"""Compute expectation values using classical shadows in a differentiable manner.

        Please refer to :func:`~.pennylane.shadow_expval` for detailed documentation.

        .. seealso:: :func:`~pennylane.shadow_expval`

        Args:
            obs (~.pennylane.measurements.ShadowExpvalMP): The classical shadow expectation
                value measurement process
            circuit (~.tapes.QuantumTape): The quantum tape that is being executed

        Raises:
            QuantumFunctionError: Shadow Expectation values are currently unsupported on :class:`~.QutritDevice`
        """
        # TODO: Add support for ShadowExpvalMP
        raise qml.QuantumFunctionError(
            "Qutrit devices don't support shadow expectation value measurements."
        )

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """

        wires = wires or self.wires
        # convert to a wires object
        wires = Wires(wires)
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)
        samples = self._samples[sample_slice, device_wires]

        # convert samples from a list of 0, 1, 2 integers, to base 10 representation
        powers_of_three = 3 ** np.arange(len(device_wires))[::-1]
        indices = samples @ powers_of_three

        # count the basis state occurrences, and construct the probability vector
        if bin_size is not None:
            bins = len(samples) // bin_size

            indices = indices.reshape((bins, -1))
            prob = np.zeros([3 ** len(device_wires), bins], dtype=np.float64)

            for b, idx in enumerate(indices):
                basis_states, counts = np.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size

        else:
            basis_states, counts = np.unique(indices, return_counts=True)
            prob = np.zeros([3 ** len(device_wires)], dtype=np.float64)
            prob[basis_states] = counts / len(samples)

        return self._asarray(prob, dtype=self.R_DTYPE)

    def marginal_prob(self, prob, wires=None):
        r"""Return the marginal probability of the computational basis
        states by summing the probabiliites on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            If the provided wires are not in the order as they appear on the device,
            the returned marginal probabilities take this permutation into account.

            For example, if the addressable wires on this device are ``Wires([0, 1, 2])`` and
            this function gets passed ``wires=[2, 0]``, then the returned marginal
            probability vector will take this 'reversal' of the two wires
            into account:

            .. math::

                \mathbb{P}^{(2, 0)}
                            = \left[
                               |00\rangle, |10\rangle, |20\rangle, |01\rangle, |11\rangle,
                               |21\rangle, |02\rangle, |12\rangle, |22\rangle
                              \right]

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array of the resulting marginal probabilities.
        """

        if wires is None:
            # no need to marginalize
            return prob

        wires = Wires(wires)
        # determine which subsystems are to be summed over
        inactive_wires = Wires.unique_wires([self.wires, wires])

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        inactive_device_wires = self.map_wires(inactive_wires)

        # reshape the probability so that each axis corresponds to a wire
        prob = self._reshape(prob, [3] * self.num_wires)

        # sum over all inactive wires
        # hotfix to catch when default.qutrit uses this method
        # since then device_wires is a list
        if isinstance(inactive_device_wires, Wires):
            wires = inactive_device_wires.labels
        else:
            wires = inactive_device_wires

        prob = self._reduce_sum(prob, wires)
        prob = self._transpose(prob, np.argsort(np.argsort(device_wires)))
        return self._flatten(prob)

    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        def _samples_to_counts(samples, no_observable_provided):
            """Group the obtained samples into a dictionary.

            **Example**

                >>> samples
                tensor([[0, 0, 1],
                        [0, 0, 1],
                        [1, 1, 1]], requires_grad=True)
                >>> self._samples_to_counts(samples)
                {'111':1, '001':2}
            """
            if no_observable_provided:
                # If we describe a state vector, we need to convert its list representation
                # into string (it's hashable and good-looking).
                # Before converting to str, we need to extract elements from arrays
                # to satisfy the case of jax interface, as jax arrays do not support str.
                samples = ["".join([str(s.item()) for s in sample]) for sample in samples]
            states, counts = np.unique(samples, return_counts=True)
            return dict(zip(states, counts))

        # TODO: Add special cases for any observables that require them once list of
        # observables is updated.

        # translate to wire labels used by device
        device_wires = self.map_wires(observable.wires)
        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)
        no_observable_provided = isinstance(observable, MeasurementProcess)

        if no_observable_provided:  # if no observable was provided then return the raw samples
            if (
                len(observable.wires) != 0
            ):  # if wires are provided, then we only return samples from those wires
                samples = self._samples[sample_slice, np.array(device_wires)]
            else:
                samples = self._samples[sample_slice]

        else:
            # Replace the basis state in the computational basis with the correct eigenvalue.
            # Extract only the columns of the basis samples required based on ``wires``.
            samples = self._samples[
                sample_slice, np.array(device_wires)
            ]  # Add np.array here for Jax support.
            powers_of_three = 3 ** np.arange(samples.shape[-1])[::-1]
            indices = samples @ powers_of_three
            indices = np.array(indices)  # Add np.array here for Jax support.
            try:
                samples = observable.eigvals()[indices]
            except qml.operation.EigvalsUndefinedError as e:
                # if observable has no info on eigenvalues, we cannot return this measurement
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute samples of {observable.name}."
                ) from e

        if bin_size is None:
            if counts:
                return _samples_to_counts(samples, no_observable_provided)
            return samples

        num_wires = len(device_wires) if len(device_wires) > 0 else self.num_wires
        if counts:
            shape = (-1, bin_size, num_wires) if no_observable_provided else (-1, bin_size)
            return [
                _samples_to_counts(bin_sample, no_observable_provided)
                for bin_sample in samples.reshape(shape)
            ]
        return (
            samples.reshape((num_wires, bin_size, -1))
            if no_observable_provided
            else samples.reshape((bin_size, -1))
        )

    # TODO: Implement function. Currently unimplemented due to lack of decompositions available
    # for existing operations and lack of non-parametrized observables.
    def adjoint_jacobian(
        self, tape, starting_state=None, use_device_state=False
    ):  # pylint: disable=missing-function-docstring
        raise NotImplementedError
