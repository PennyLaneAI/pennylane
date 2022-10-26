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
# pylint: disable=protected-access
"""
This module contains the functions for computing different types of measurement
outcomes from quantum observables - expectation values, variances of expectations,
and measurement samples using AnnotatedQueues.
"""
# pylint: disable=too-many-instance-attributes
import contextlib
import copy
import functools
from enum import Enum
from typing import Generic, TypeVar

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .expval import Expectation
from .mid_measure import MidMeasure
from .mutual_info import MutualInfo
from .probs import Probability
from .sample import Sample
from .state import State
from .var import Variance
from .vn_entropy import VnEntropy

# =============================================================================
# ObservableReturnTypes types
# =============================================================================


class ObservableReturnTypes(Enum):
    """Enumeration class to represent the return types of an observable."""

    Sample = "sample"
    Counts = "counts"
    AllCounts = "allcounts"
    Variance = "var"
    Expectation = "expval"
    Probability = "probs"
    State = "state"
    MidMeasure = "measure"
    VnEntropy = "vnentropy"
    MutualInfo = "mutualinfo"
    Shadow = "shadow"
    ShadowExpval = "shadowexpval"

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)


class MeasurementShapeError(ValueError):
    """An error raised when an unsupported operation is attempted with a
    quantum tape."""


class MeasurementProcess:
    """Represents a measurement process occurring at the end of a
    quantum variational circuit.

    Args:
        return_type (.ObservableReturnTypes): The type of measurement process.
            This includes ``Expectation``, ``Variance``, ``Sample``, ``State``, or ``Probability``.
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
    """

    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-arguments

    def __init__(
        self, return_type, obs: Operator = None, wires=None, eigvals=None, id=None, log_base=None
    ):
        self.return_type = return_type
        self.obs = obs
        self.id = id
        self.log_base = log_base

        if wires is not None and obs is not None:
            raise ValueError("Cannot set the wires if an observable is provided.")

        self._wires = wires or Wires([])
        self._eigvals = None

        if eigvals is not None:
            if obs is not None:
                raise ValueError("Cannot set the eigenvalues if an observable is provided.")

            self._eigvals = np.array(eigvals)

        # TODO: remove the following lines once devices
        # have been refactored to accept and understand recieving
        # measurement processes rather than specific observables.

        # The following lines are only applicable for measurement processes
        # that do not have corresponding observables (e.g., Probability). We use
        # them to 'trick' the device into thinking it has recieved an observable.

        # Below, we imitate an identity observable, so that the
        # device undertakes no action upon recieving this observable.
        self.name = "Identity"
        self.data = []

        # Queue the measurement process
        self.queue()

    @property
    @functools.lru_cache()
    def numeric_type(self):
        """The Python numeric type of the measurement result.

        Returns:
            type: The output numeric type; ``int``, ``float`` or ``complex``.

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        if self.return_type in (Expectation, MutualInfo, Probability, Variance, VnEntropy):
            return float

        if self.return_type is State:
            return complex

        if self.return_type is Sample:

            # Note: we only assume an integer numeric type if the observable is a
            # built-in observable with integer eigenvalues or a tensor product thereof
            if self.obs is None:

                # Computational basis samples
                return int
            int_eigval_obs = {qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity}
            tensor_terms = self.obs.obs if hasattr(self.obs, "obs") else [self.obs]
            every_term_standard = all(o.__class__ in int_eigval_obs for o in tensor_terms)
            return int if every_term_standard else float

        raise qml.QuantumFunctionError(
            "Cannot deduce the numeric type of the measurement process with unrecognized "
            + f"return_type {self.return_type}."
        )

    @functools.lru_cache()
    def shape(self, device=None):
        """The expected output shape of the MeasurementProcess.

        Note that the output shape is dependent on the device when:

        * The ``return_type`` is either ``Probability``, ``State`` (from :func:`.state`) or
          ``Sample``;
        * The shot vector was defined in the device.

        For example, assuming a device with ``shots=None``, expectation values
        and variances define ``shape=(1,)``, whereas probabilities in the qubit
        model define ``shape=(1, 2**num_wires)`` where ``num_wires`` is the
        number of wires the measurement acts on.

        Note that the shapes for vector-valued return types such as
        ``Probability`` and ``State`` are adjusted to the output of
        ``qml.execute`` and may have an extra first element that is squeezed
        when using QNodes.

        Args:
            device (.Device): a PennyLane device to use for determining the
                shape

        Returns:
            tuple: the output shape

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        if qml.active_return():
            return self._shape_new(device=device)

        shape = None

        # First: prepare the shape for return types that do not require a
        # device
        if self.return_type in (Expectation, MutualInfo, Variance, VnEntropy):
            shape = (1,)

        if self.return_type == State and self.wires:
            dim = 2 ** len(self.wires)
            shape = (1, dim, dim)

        # Determine shape if device with shot vector
        if device is not None and device._shot_vector is not None:
            shape = self._shot_vector_shape(device, main_shape=shape)

        # If we have a shape, return it here
        if shape is not None:
            return shape

        # Then: handle return types that require a device; no shot vector
        if device is None and self.return_type in (Probability, State, Sample):
            raise MeasurementShapeError(
                "The device argument is required to obtain the shape of the measurement process; "
                + f"got return type {self.return_type}."
            )

        if self.return_type == Probability:
            len_wires = len(self.wires)
            dim = self._get_num_basis_states(len_wires, device)
            return (1, dim)

        if self.return_type == State:

            # Note: qml.density_matrix has its shape defined, so we're handling
            # the qml.state case; acts on all device wires
            dim = 2 ** len(device.wires)
            return (1, dim)

        if self.return_type == Sample:
            len_wires = len(device.wires)
            # qml.sample(some_observable) case  // qml.sample() case
            return (1, device.shots) if self.obs is not None else (1, device.shots, len_wires)

        raise qml.QuantumFunctionError(
            "Cannot deduce the shape of the measurement process with unrecognized return_type "
            + f"{self.return_type}."
        )

    @functools.lru_cache()
    def _shape_new(self, device=None):
        """The expected output shape of the MeasurementProcess.

        Note that the output shape is dependent on the device when:

        * The ``return_type`` is either ``Probability``, ``State`` (from :func:`.state`) or
          ``Sample``;
        * The shot vector was defined in the device.

        For example, assuming a device with ``shots=None``, expectation values
        and variances define ``shape=(,)``, whereas probabilities in the qubit
        model define ``shape=(2**num_wires)`` where ``num_wires`` is the
        number of wires the measurement acts on.

        Args:
            device (.Device): a PennyLane device to use for determining the
                shape

        Returns:
            tuple: the output shape

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        shape = None

        # First: prepare the shape for return types that do not require a
        # device
        if self.return_type in (Expectation, MutualInfo, Variance, VnEntropy):
            shape = ()

        if self.return_type == State and self.wires:
            dim = 2 ** len(self.wires)
            shape = (dim, dim)

        # Determine shape if device with shot vector
        if device is not None and device._shot_vector is not None:
            shape = self._shot_vector_shape(device, main_shape=shape)

        # If we have a shape, return it here
        if shape is not None:
            return shape

        # Then: handle return types that require a device; no shot vector
        if device is None and self.return_type in (Probability, State, Sample):
            raise MeasurementShapeError(
                "The device argument is required to obtain the shape of the measurement process; "
                + f"got return type {self.return_type}."
            )

        if self.return_type == Probability:
            len_wires = len(self.wires)
            dim = self._get_num_basis_states(len_wires, device)
            return (dim,)

        if self.return_type == State:
            # Note: qml.density_matrix has its shape defined, so we're handling
            # the qml.state case; acts on all device wires
            dim = 2 ** len(device.wires)
            return (dim,)

        if self.return_type == Sample:
            if self.obs is not None:
                # qml.sample(some_observable) case
                return () if device.shots == 1 else (device.shots,)

            # qml.sample() case
            len_wires = len(device.wires)
            return (len_wires,) if device.shots == 1 else (device.shots, len_wires)

        raise qml.QuantumFunctionError(
            "Cannot deduce the shape of the measurement process with unrecognized return_type "
            + f"{self.return_type}."
        )

    @functools.lru_cache()
    def _shot_vector_shape(self, device, main_shape=None):
        """Auxiliary function for getting the output shape when the device has
        the shot vector defined.

        The shape is device dependent even if the return type has a main shape
        pre-defined (e.g., expectation values, states, etc.).
        """
        if qml.active_return():
            return self._shot_vector_shape_new(device, main_shape=main_shape)

        shot_vector = device._shot_vector
        # pylint: disable=consider-using-generator
        num_shot_elements = sum(s.copies for s in shot_vector)
        shape = ()

        if main_shape is not None:

            # Expval, var and density_matrix case
            shape = list(main_shape)
            shape[0] *= num_shot_elements
            shape = tuple(shape)

        elif self.return_type == qml.measurements.Probability:

            len_wires = len(self.wires)
            dim = self._get_num_basis_states(len_wires, device)
            shape = (num_shot_elements, dim)

        elif self.return_type == qml.measurements.Sample:
            if self.obs is not None:
                shape = tuple(
                    (shot_val,) if shot_val != 1 else tuple()
                    for shot_val in device._raw_shot_sequence
                )
            else:
                # TODO: revisit when qml.sample without an observable fully
                # supports shot vectors
                raise MeasurementShapeError(
                    "Getting the output shape of a measurement returning samples along with "
                    "a device with a shot vector is not supported."
                )

        elif self.return_type == qml.measurements.State:

            # Note: qml.density_matrix has its shape defined, so we're handling
            # the qml.state case; acts on all device wires
            dim = 2 ** len(device.wires)
            shape = (num_shot_elements, dim)

        return shape

    @functools.lru_cache()
    def _shot_vector_shape_new(self, device, main_shape=None):
        """Auxiliary function for getting the output shape when the device has
        the shot vector defined.

        The shape is device dependent even if the return type has a main shape
        pre-defined (e.g., expectation values, states, etc.).
        """
        shot_vector = device._shot_vector
        # pylint: disable=consider-using-generator
        num_shot_elements = sum(s.copies for s in shot_vector)
        shape = ()

        if main_shape is not None:
            shape = tuple(main_shape for _ in range(num_shot_elements))

        elif self.return_type == qml.measurements.Probability:
            dim = self._get_num_basis_states(len(self.wires), device)
            shape = tuple((dim,) for _ in range(num_shot_elements))

        elif self.return_type == qml.measurements.Sample:
            if self.obs is not None:
                shape = tuple(
                    (shot_val,) if shot_val != 1 else tuple()
                    for shot_val in device._raw_shot_sequence
                )
            else:
                shape = tuple(
                    (shot_val, len(device.wires)) if shot_val != 1 else (len(device.wires),)
                    for shot_val in device._raw_shot_sequence
                )

        elif self.return_type == qml.measurements.State:

            # Note: qml.density_matrix has its shape defined, so we're handling
            # the qml.state case; acts on all device wires
            dim = 2 ** len(device.wires)
            shape = tuple((dim,) for _ in range(num_shot_elements))

        return shape

    @staticmethod
    @functools.lru_cache()
    def _get_num_basis_states(num_wires, device):
        """Auxiliary function to determine the number of basis states given the
        number of systems and a quantum device.

        This function is meant to be used with the Probability measurement to
        determine how many outcomes there will be. With qubit based devices
        we'll have two outcomes for each subsystem. With continuous variable
        devices that impose a Fock cutoff the number of basis states per
        subsystem equals the cutoff value.

        Args:
            num_wires (int): the number of qubits/qumodes
            device (.Device): a PennyLane device

        Returns:
            int: the number of basis states
        """
        cutoff = getattr(device, "cutoff", None)
        base = 2 if cutoff is None else cutoff
        return base**num_wires

    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[.Operation]: the operations that diagonalize the observables
        """
        try:
            # pylint: disable=no-member
            return self.expand().operations
        except qml.operation.DecompositionUndefinedError:
            return []

    def __repr__(self):
        """Representation of this class."""
        if self.obs is None:
            return f"{self.return_type.value}(wires={self.wires.tolist()})"

        # Todo: when tape is core the return type will always be taken from the MeasurementProcess
        if self.obs.return_type is None:
            return f"{self.return_type.value}({self.obs})"

        return f"{self.obs}"

    def __copy__(self):
        cls = self.__class__

        if self.obs is not None:
            return cls(self.return_type, obs=copy.copy(self.obs))

        if self.log_base is not None:
            return cls(self.return_type, wires=self._wires, log_base=self.log_base)

        return cls(self.return_type, eigvals=self._eigvals, wires=self._wires)

    @property
    def wires(self):
        r"""The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if self.obs is not None:
            return self.obs.wires

        return Wires.all_wires(self._wires) if isinstance(self._wires, list) else self._wires

    @property
    def raw_wires(self):
        r"""The wires the measurement process acts on.

        For measurements involving more than one set of wires (such as
        mutual information), this is a list of the Wires objects. Otherwise,
        this is the same as :func:`~.MeasurementProcess.wires`
        """
        return self._wires

    def eigvals(self):
        r"""Eigenvalues associated with the measurement process.

        If the measurement process has an associated observable,
        the eigenvalues will correspond to this observable. Otherwise,
        they will be the eigenvalues provided when the measurement
        process was instantiated.

        Note that the eigenvalues are not guaranteed to be in any
        particular order.

        **Example:**

        >>> m = MeasurementProcess(Expectation, obs=qml.PauliX(wires=1))
        >>> m.eigvals()
        array([1, -1])

        Returns:
            array: eigvals representation
        """
        if self.obs is not None:
            with contextlib.suppress(qml.operation.EigvalsUndefinedError):
                return self.obs.eigvals()
        return self._eigvals

    @property
    def has_decomposition(self):
        r"""Bool: Whether or not the MeasurementProcess returns a defined decomposition
        when calling ``expand``.
        """
        # If self.obs is not None, `expand` queues the diagonalizing gates of self.obs,
        # which we have to check to be defined. The subsequent creation of the new
        # `MeasurementProcess` within `expand` should never fail with the given parameters.
        return False if self.obs is None else self.obs.has_diagonalizing_gates

    def expand(self):
        """Expand the measurement of an observable to a unitary
        rotation and a measurement in the computational basis.

        Returns:
            .QuantumTape: a quantum tape containing the operations
            required to diagonalize the observable

        **Example**

        Consider a measurement process consisting of the expectation
        value of an Hermitian observable:

        >>> H = np.array([[1, 2], [2, 4]])
        >>> obs = qml.Hermitian(H, wires=['a'])
        >>> m = MeasurementProcess(Expectation, obs=obs)

        Expanding this out:

        >>> tape = m.expand()

        We can see that the resulting tape has the qubit unitary applied,
        and a measurement process with no observable, but the eigenvalues
        specified:

        >>> print(tape.operations)
        [QubitUnitary(array([[-0.89442719,  0.4472136 ],
              [ 0.4472136 ,  0.89442719]]), wires=['a'])]
        >>> print(tape.measurements[0].eigvals())
        [0. 5.]
        >>> print(tape.measurements[0].obs)
        None
        """
        if self.obs is None:
            raise qml.operation.DecompositionUndefinedError

        with qml.tape.QuantumTape() as tape:
            self.obs.diagonalizing_gates()
            MeasurementProcess(self.return_type, wires=self.obs.wires, eigvals=self.obs.eigvals())

        return tape

    def queue(self, context=qml.QueuingManager):
        """Append the measurement process to an annotated queue."""
        if self.obs is not None:
            context.update_info(self.obs, owner=self)
            context.append(self, owns=self.obs)
        else:
            context.append(self)

        return self

    @property
    def _queue_category(self):
        """Denotes that `MeasurementProcess` objects should be processed into the `_measurements` list
        in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.
        """
        return "_ops" if self.return_type is MidMeasure else "_measurements"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""

        if self.obs is None:
            fingerprint = (
                str(self.name),
                tuple(self.wires.tolist()),
                str(self.data),
                self.return_type,
            )
        else:
            fingerprint = (
                str(self.obs.name),
                tuple(self.wires.tolist()),
                str(self.obs.data),
                self.return_type,
            )

        return hash(fingerprint)

    def simplify(self):
        """Reduce the depth of the observable to the minimum.

        Returns:
            .MeasurementProcess: A measurement process with a simplified observable.
        """
        if self.obs is None:
            return self

        return MeasurementProcess(return_type=self.return_type, obs=self.obs.simplify())


T = TypeVar("T")


class MeasurementValueError(ValueError):
    """Error raised when an unknown measurement value is being used."""


class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurement_id (str): The id of the measurement that this object depends on.
        zero_case (float): the first measurement outcome value
        one_case (float): the second measurement outcome value
    """

    __slots__ = ("_depends_on", "_zero_case", "_one_case", "_control_value")

    def __init__(
        self,
        measurement_id: str,
        zero_case: float = 0,
        one_case: float = 1,
    ):
        self._depends_on = measurement_id
        self._zero_case = zero_case
        self._one_case = one_case
        self._control_value = one_case  # By default, control on the one case

    @property
    def branches(self):
        """A dictionary representing all the possible outcomes of the MeasurementValue."""
        branch_dict = {}
        branch_dict[(0,)] = self._zero_case
        branch_dict[(1,)] = self._one_case
        return branch_dict

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        inverted_self = copy.copy(self)
        zero = self._zero_case
        one = self._one_case

        inverted_self._control_value = one if self._control_value == zero else zero

        return inverted_self

    def __eq__(self, control_value):
        """Allow asserting measurement values."""
        measurement_outcomes = {self._zero_case, self._one_case}

        if not isinstance(control_value, tuple(type(val) for val in measurement_outcomes)):
            raise MeasurementValueError(
                "The equality operator is used to assert measurement outcomes, but got a value "
                + f"with type {type(control_value)}."
            )

        if control_value not in measurement_outcomes:
            raise MeasurementValueError(
                "Unknown measurement value asserted; the set of possible measurement outcomes is: "
                + f"{measurement_outcomes}."
            )

        self._control_value = control_value
        return self

    @property
    def control_value(self):
        """The control value to consider for the measurement outcome."""
        return self._control_value

    @property
    def measurements(self):
        """List of all measurements this MeasurementValue depends on."""
        return [self._depends_on]
