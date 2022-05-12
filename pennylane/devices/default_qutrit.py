r"""
The default.qutrit device is Pennylane's standard qutrit-based device.

It implements the :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qutrit operations <pennylane.ops.qutrit>`, and provides a simple pure state
simulation of qutrit-based quantum circuit architecture
"""
import functools
import numpy as np

import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.wires import Wires, WireError
from .._version import __version__
from default_qubit import _get_slice

from pennylane.measurements import MeasurementProcess

# tolerance for numerical errors
tolerance = 1e-10


class DefaultQutrit(Device):
    """Default qutrit device for PennyLane

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """

    name = "Default qutrit PennyLane plugin"
    short_name = "default.qutrit"
    pennylane_requires = __version__
    version = __version__
    # TODO: Add author

    # TODO: Add list of operations and observables
    operations = {}

    observables = {}

    def __init__(
        self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None,
    ):
        super().__init__(wires, shots, analytic)
        self._debugger = None

        # Create the initial state. Internally, we store the
        # state as an array of dimension [3]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # TODO: Add operations


        self._apply_ops = {}

    @functools.lru_cache()
    def map_wires(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        try:
            mapped_wires = [self.wire_map[w] for w in wires]
        except KeyError as e:
            raise WireError(
                f"Did not find some of the wires {wires.labels} on device with wires {self.wires.labels}."
            ) from e

        return mapped_wires

    def define_wire_map(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        consecutive_wires = range(self.num_wires)
        wire_map = zip(wires, consecutive_wires)
        return dict(wire_map)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

    def _apply_operation(self, state, operation):
        pass

    def expval(self, observable, shot_range=None, bin_size=None):
        pass

    def _get_unitary_matrix(self, unitary):
        pass

    @classmethod
    def capabilities(cls):
        pass

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[3]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(3**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [3] * self.num_wires)

    @property
    def state(self):
        return self._flatten(self._pre_rotated_state)

    def density_matrix(self, wires):
        pass

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        flat_state = self._flatten(self._state)
        real_state = self._real(flat_state)
        imag_state = self._imag(flat_state)
        # prob = self.marginal_prob(real_state**2 + imag_state**2, wires)
        # return prob

