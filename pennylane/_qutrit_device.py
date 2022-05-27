import abc
import itertools
import warnings

import numpy as np

import pennylane as qml
from pennylane import DeviceError
from pennylane.operation import operation_derivative
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
from pennylane import Device
from pennylane.math import sum as qmlsum
from pennylane.math import multiply as qmlmul
from pennylane.wires import Wires

from pennylane.measurements import MeasurementProcess


class QutritDevice(Device):
    """Abstract base class for Pennylane qutrit devices.

    The following abstract method **must** be defined:

    * :meth:`~.apply`: append circuit operations, compile the circuit (if applicable),
      and perform the quantum computation.

    Devices that generate their own samples (such as hardware) may optionally
    overwrite :meth:`~.probabilty`. This method otherwise automatically
    computes the probabilities from the generated samples, and **must**
    overwrite the following method:

    * :meth:`~.generate_samples`: Generate samples from the device from the
      exact or approximate probability distribution.

    Analytic devices **must** overwrite the following method:

    * :meth:`~.analytic_probability`: returns the probability or marginal probability from the
      device after circuit execution. :meth:`~.marginal_prob` may be used here.

    This device contains common utility methods for qubit-based devices. These
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

    _asarray = staticmethod(np.asarray)
    _dot = staticmethod(np.dot)
    _abs = staticmethod(np.abs)
    _reduce_sum = staticmethod(lambda array, axes: np.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(np.reshape)
    _flatten = staticmethod(lambda array: array.flatten())
    _gather = staticmethod(lambda array, indices: array[indices])
    _einsum = staticmethod(np.einsum)
    _cast = staticmethod(np.asarray)
    _transpose = staticmethod(np.transpose)
    _tensordot = staticmethod(np.tensordot)
    _conj = staticmethod(np.conj)
    _imag = staticmethod(np.imag)
    _roll = staticmethod(np.roll)
    _stack = staticmethod(np.stack)
    _outer = staticmethod(np.outer)
    _diag = staticmethod(np.diag)
    _real = staticmethod(np.real)

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = np.zeros(new_dimensions, dtype=array.dtype.type)
        new_array[indices] = array
        return new_array

    @staticmethod
    def _const_mul(constant, array):
        """Data type preserving multiply operation"""
        return qmlmul(constant, array, dtype=array.dtype)

    def _permute_wires(self, observable):
        r"""Given an observable which acts on multiple wires, permute the wires to
        be consistent with the device wire order.

        This function uses the observable wires and the global device wire ordering in order to determine the
        permutation of the wires in the observable required such that if our quantum state vector is
        permuted accordingly then the amplitudes of the state will match the matrix representation of the observable.

        Args:
            observable (Observable): the observable whose wires are to be permuted.

        Returns:
            permuted_wires (Wires): permuted wires object
        """
        ordered_obs_wire_lst = self.order_wires(
            observable.wires
        ).tolist()  # order according to device wire order

        mapped_wires = self.map_wires(observable.wires)
        if isinstance(mapped_wires, Wires):
            # by default this should be a Wires obj, but it is overwritten to list object in default.qubit
            mapped_wires = mapped_wires.tolist()

        permutation = np.argsort(mapped_wires)  # extract permutation via argsort

        permuted_wires = Wires([ordered_obs_wire_lst[index] for index in permutation])
        return permuted_wires

    #########################################################################
    #########################################################################
    #################### TODO: Add list of observables ######################
    #########################################################################
    #########################################################################
    observables = {}

    def __init__(
        self, wires=1, shots=None, *, r_dtype=np.float64, c_dtype=np.complex128, analytic=None
    ):
        super().__init__(wires=wires, shots=shots, analytic=analytic)

        if "float" not in str(r_dtype):
            raise DeviceError("Real datatype must be a floating point type.")
        if "complex" not in str(c_dtype):
            raise DeviceError("Complex datatype must be a complex floating point type.")

        self.C_DTYPE = c_dtype
        self.R_DTYPE = r_dtype

        self._samples = None
        """None or array[int]: stores the samples generated by the device
        *after* rotation to diagonalize the observables."""

    @classmethod
    def capabilities(cls):

        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qutrit",
            supports_finite_shots=True,
            supports_tensor_observables=True,
            returns_probs=True,
        )
        return capabilities

    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        self._samples = None

    #########################################################################
    #########################################################################
    ##### TODO: Implement all functions that are currently being passed #####
    #########################################################################
    #########################################################################
    def execute(self, circuit, **kwargs):
        pass

    def batch_execute(self, circuits):
        pass

    @abc.abstractmethod
    def apply(self, operations, **kwargs):
        pass

    @staticmethod
    def active_wires(operators):
        pass

    def statistics(self, observables, shot_range=None, bin_size=None):
        pass

    def access_state(self, wires=None):
        pass

    def generate_samples(self):
        pass

    def sample_basis_states(self, number_of_states, state_probability):
        pass

    @staticmethod
    def generate_basis_states(num_wires, dtype=np.uint32):
        pass

    @staticmethod
    def states_to_ternary(samples, num_wires, dtype=np.int64):
        pass

    @property
    def circuit_hash(self):
        """The hash of the circuit upon the last execution.

        This can be used by devices in :meth:`~.apply` for parametric compilation.
        """
        raise NotImplementedError

    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def density_matrix(self, wires):
        """Returns the reduced density matrix prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def analytic_probability(self, wires=None):
        r"""Return the (marginal) probability of each computational basis
        state from the last run of the device.

        PennyLane uses the convention
        :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where :math:`q_0` is the most
        significant trit.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            :meth:`marginal_prob` may be used as a utility method
            to calculate the marginal probability distribution.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        raise NotImplementedError

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
        pass

    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Return either the analytic probability or estimated probability of
        each computational basis state.

        Devices that require a finite number of shots always return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """

        if self.shots is None:
            return self.analytic_probability(wires=wires)

        return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

    def marginal_prob(self, prob, wires=None):
        pass

    def expval(self, observable, shot_range=None, bin_size=None):
        pass

    def var(self, observable, shot_range=None, bin_size=None):
        pass

    def sample(self, observable, shot_range=None, bin_size=None):
        pass

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        pass