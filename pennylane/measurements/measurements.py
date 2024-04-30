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
"""
This module contains the functions for computing different types of measurement
outcomes from quantum observables - expectation values, variances of expectations,
and measurement samples using AnnotatedQueues.
"""
import copy
import functools

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional, Union

import pennylane as qml
from pennylane.operation import Operator, DecompositionUndefinedError, EigvalsUndefinedError
from pennylane.pytrees import register_pytree
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .shots import Shots

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
    Purity = "purity"

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)


Sample = ObservableReturnTypes.Sample
"""Enum: An enumeration which represents sampling an observable."""

Counts = ObservableReturnTypes.Counts
"""Enum: An enumeration which represents returning the number of times
 each of the observed outcomes occurred in sampling."""

AllCounts = ObservableReturnTypes.AllCounts
"""Enum: An enumeration which represents returning the number of times
 each of the possible outcomes occurred in sampling, including 0 counts
 for unobserved outcomes."""

Variance = ObservableReturnTypes.Variance
"""Enum: An enumeration which represents returning the variance of
an observable on specified wires."""

Expectation = ObservableReturnTypes.Expectation
"""Enum: An enumeration which represents returning the expectation
value of an observable on specified wires."""

Probability = ObservableReturnTypes.Probability
"""Enum: An enumeration which represents returning probabilities
of all computational basis states."""

State = ObservableReturnTypes.State
"""Enum: An enumeration which represents returning the state in the computational basis."""

MidMeasure = ObservableReturnTypes.MidMeasure
"""Enum: An enumeration which represents returning sampling the computational
basis in the middle of the circuit."""

VnEntropy = ObservableReturnTypes.VnEntropy
"""Enum: An enumeration which represents returning Von Neumann entropy before measurements."""

MutualInfo = ObservableReturnTypes.MutualInfo
"""Enum: An enumeration which represents returning the mutual information before measurements."""

Shadow = ObservableReturnTypes.Shadow
"""Enum: An enumeration which represents returning the bitstrings and recipes from
the classical shadow protocol"""

ShadowExpval = ObservableReturnTypes.ShadowExpval
"""Enum: An enumeration which represents returning the estimated expectation value
from a classical shadow measurement"""

Purity = ObservableReturnTypes.Purity
"""Enum: An enumeration which represents returning the purity of the system prior ot measurement."""


class MeasurementShapeError(ValueError):
    """An error raised when an unsupported operation is attempted with a
    quantum tape."""


class MeasurementProcess(ABC):
    """Represents a measurement process occurring at the end of a
    quantum variational circuit.

    Args:
        obs (Union[.Operator, .MeasurementValue, Sequence[.MeasurementValue]]): The observable that
            is to be measured as part of the measurement process. Not all measurement processes
            require observables (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    # pylint:disable=too-many-instance-attributes

    def __init_subclass__(cls, **_):
        register_pytree(cls, cls._flatten, cls._unflatten)

    def _flatten(self):
        metadata = (("wires", self.raw_wires),)
        return (self.obs or self.mv, self._eigvals), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        if data[0] is not None:
            return cls(obs=data[0], **dict(metadata))
        if data[1] is not None:
            return cls(eigvals=data[1], **dict(metadata))
        return cls(**dict(metadata))

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        obs: Optional[
            Union[
                Operator,
                "qml.measurements.MeasurementValue",
                Sequence["qml.measurements.MeasurementValue"],
            ]
        ] = None,
        wires: Optional[Wires] = None,
        eigvals: Optional[TensorLike] = None,
        id: Optional[str] = None,
    ):
        if getattr(obs, "name", None) == "MeasurementValue" or isinstance(obs, Sequence):
            # Cast sequence of measurement values to list
            self.mv = obs if getattr(obs, "name", None) == "MeasurementValue" else list(obs)
            self.obs = None
        else:
            self.obs = obs
            self.mv = None

        self.id = id

        if wires is not None:
            if len(wires) == 0:
                raise ValueError("Cannot set an empty list of wires.")
            if obs is not None:
                raise ValueError("Cannot set the wires if an observable is provided.")

        # _wires = None indicates broadcasting across all available wires.
        # It translates to the public property wires = Wires([])
        self._wires = wires
        self._eigvals = None

        if eigvals is not None:
            if obs is not None:
                raise ValueError("Cannot set the eigenvalues if an observable is provided.")

            self._eigvals = qml.math.asarray(eigvals)

        # Queue the measurement process
        self.queue()

    @property
    def return_type(self) -> Optional[ObservableReturnTypes]:
        """Measurement return type."""
        return None

    @property
    def numeric_type(self) -> type:
        """The Python numeric type of the measurement result.

        Returns:
            type: The output numeric type; ``int``, ``float`` or ``complex``.

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        raise qml.QuantumFunctionError(
            f"The numeric type of the measurement {self.__class__.__name__} is not defined."
        )

    def shape(self, device, shots: Shots) -> Tuple:
        """The expected output shape of the MeasurementProcess.

        Note that the output shape is dependent on the shots or device when:

        * The measurement type is either ``_Probability``, ``_State`` (from :func:`.state`) or
          ``_Sample``;
        * The shot vector was defined.

        For example, assuming a device with ``shots=None``, expectation values
        and variances define ``shape=(,)``, whereas probabilities in the qubit
        model define ``shape=(2**num_wires)`` where ``num_wires`` is the
        number of wires the measurement acts on.

        Args:
            device (pennylane.Device): a PennyLane device to use for determining the shape
            shots (~.Shots): object defining the number and batches of shots

        Returns:
            tuple: the output shape

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        raise qml.QuantumFunctionError(
            f"The shape of the measurement {self.__class__.__name__} is not defined"
        )

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
            device (pennylane.Device): a PennyLane device

        Returns:
            int: the number of basis states
        """
        cutoff = getattr(device, "cutoff", None)
        base = 2 if cutoff is None else cutoff
        return base**num_wires

    @qml.QueuingManager.stop_recording()
    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[.Operation]: the operations that diagonalize the observables
        """
        return self.obs.diagonalizing_gates() if self.obs else []

    def __eq__(self, other):
        return qml.equal(self, other)

    def __hash__(self):
        return self.hash

    def __repr__(self):
        """Representation of this class."""
        if self.mv:
            return f"{self.return_type.value}({repr(self.mv)})"
        if self.obs:
            return f"{self.return_type.value}({self.obs})"
        if self._eigvals is not None:
            return f"{self.return_type.value}(eigvals={self._eigvals}, wires={self.wires.tolist()})"

        # Todo: when tape is core the return type will always be taken from the MeasurementProcess
        return f"{getattr(self.return_type, 'value', 'None')}(wires={self.wires.tolist()})"

    def __copy__(self):
        cls = self.__class__
        copied_m = cls.__new__(cls)

        for attr, value in vars(self).items():
            setattr(copied_m, attr, value)

        if self.obs is not None:
            copied_m.obs = copy.copy(self.obs)

        return copied_m

    @property
    def wires(self):
        r"""The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if self.mv is not None:
            if isinstance(self.mv, list):
                return qml.wires.Wires.all_wires([m.wires for m in self.mv])
            return self.mv.wires

        if self.obs is not None:
            return self.obs.wires

        return (
            Wires.all_wires(self._wires)
            if isinstance(self._wires, (tuple, list))
            else self._wires or Wires([])
        )

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

        >>> m = MeasurementProcess(Expectation, obs=qml.X(1))
        >>> m.eigvals()
        array([1, -1])

        Returns:
            array: eigvals representation
        """
        if self.mv is not None:
            if getattr(self.mv, "name", None) == "MeasurementValue":
                # Indexing a MeasurementValue gives the output of the processing function
                # for the binary number corresponding to the index.
                return qml.math.asarray([self.mv[i] for i in range(2 ** len(self.wires))])
            return qml.math.arange(0, 2 ** len(self.wires), 1)

        if self.obs is not None:
            try:
                return qml.eigvals(self.obs)
            except DecompositionUndefinedError as e:
                raise EigvalsUndefinedError from e
        return self._eigvals

    @property
    def has_decomposition(self):
        r"""Bool: Whether or not the MeasurementProcess returns a defined decomposition
        when calling ``expand``.
        """
        # If self.obs is not None, `expand` queues the diagonalizing gates of self.obs,
        # which we have to check to be defined. The subsequent creation of the new
        # `MeasurementProcess` within `expand` should never fail with the given parameters.
        return self.obs.has_diagonalizing_gates if self.obs is not None else False

    @property
    def samples_computational_basis(self):
        r"""Bool: Whether or not the MeasurementProcess measures in the computational basis."""
        return self.obs is None

    def expand(self):
        """Expand the measurement of an observable to a unitary
        rotation and a measurement in the computational basis.

        Returns:
            .QuantumTape: a quantum tape containing the operations
            required to diagonalize the observable

        **Example:**

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

        with qml.queuing.AnnotatedQueue() as q:
            self.obs.diagonalizing_gates()
            self.__class__(wires=self.obs.wires, eigvals=self.obs.eigvals())

        return qml.tape.QuantumScript.from_queue(q)

    def queue(self, context=qml.QueuingManager):
        """Append the measurement process to an annotated queue."""
        if self.obs is not None:
            context.remove(self.obs)
        context.append(self)

        return self

    @property
    def _queue_category(self):
        """Denotes that `MeasurementProcess` objects should be processed into the `_measurements` list
        in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.
        """
        return "_measurements"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            getattr(self.obs, "hash", "None"),
            getattr(self.mv, "hash", "None"),
            str(self._eigvals),  # eigvals() could be expensive to compute for large observables
            tuple(self.wires.tolist()),
        )

        return hash(fingerprint)

    def simplify(self):
        """Reduce the depth of the observable to the minimum.

        Returns:
            .MeasurementProcess: A measurement process with a simplified observable.
        """
        return self if self.obs is None else self.__class__(obs=self.obs.simplify())

    # pylint: disable=protected-access
    def map_wires(self, wire_map: dict):
        """Returns a copy of the current measurement process with its wires changed according to
        the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .MeasurementProcess: new measurement process
        """
        new_measurement = copy.copy(self)
        if self.mv is not None:
            new_measurement.mv = (
                self.mv.map_wires(wire_map=wire_map)
                if getattr(self.mv, "name", None) == "MeasurementValue"
                else [m.map_wires(wire_map=wire_map) for m in self.mv]
            )
        elif self.obs is not None:
            new_measurement.obs = self.obs.map_wires(wire_map=wire_map)
        elif self._wires is not None:
            new_measurement._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        return new_measurement


class SampleMeasurement(MeasurementProcess):
    """Sample-based measurement process.

    Any class inheriting from ``SampleMeasurement`` should define its own ``process_samples`` method,
    which should have the following arguments:

    * samples (Sequence[complex]): computational basis samples generated for all wires
    * wire_order (Wires): wires determining the subspace that ``samples`` acts on
    * shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
        to use. If not specified, all samples are used.
    * bin_size (int): Divides the shot range into bins of size ``bin_size``, and
        returns the measurement statistic separately over each bin. If not
        provided, the entire shot range is treated as a single bin.

    **Example:**

    Let's create a measurement that returns the sum of all samples of the given wires.

    >>> class MyMeasurement(SampleMeasurement):
    ...     def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
    ...         return qml.math.sum(samples[..., self.wires])

    We can now execute it in a QNode:

    >>> dev = qml.device("default.qubit", wires=2, shots=1000)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.X(0)
    ...     return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])
    >>> circuit()
    (tensor(1000, requires_grad=True), tensor(0, requires_grad=True))
    """

    @abstractmethod
    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        """Process the given samples.

        Args:
            samples (Sequence[complex]): computational basis samples generated for all wires
            wire_order (Wires): wires determining the subspace that ``samples`` acts on
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.
        """

    @abstractmethod
    def process_counts(self, counts: dict, wire_order: Wires):
        """Calculate the measurement given a counts histogram dictionary.

        Args:
            counts (dict): a dictionary matching the format returned by :class:`~.CountsMP`
            wire_order (Wires): the wire order used in producing the counts

        Note that the input dictionary may only contain states with non-zero entries (``all_outcomes=False``).
        """


class StateMeasurement(MeasurementProcess):
    """State-based measurement process.

    Any class inheriting from ``StateMeasurement`` should define its own ``process_state`` method,
    which should have the following arguments:

    * state (Sequence[complex]): quantum state with a flat shape. It may also have an
        optional batch dimension
    * wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
        dimension :math:`2^n` acts on a subspace of :math:`n` wires

    **Example:**

    Let's create a measurement that returns the diagonal of the reduced density matrix.

    >>> class MyMeasurement(StateMeasurement):
    ...     def process_state(self, state, wire_order):
    ...         # use the already defined `qml.density_matrix` measurement to compute the
    ...         # reduced density matrix from the given state
    ...         density_matrix = qml.density_matrix(wires=self.wires).process_state(state, wire_order)
    ...         return qml.math.diagonal(qml.math.real(density_matrix))

    We can now execute it in a QNode:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.Hadamard(0)
    ...     qml.CNOT([0, 1])
    ...     return MyMeasurement(wires=[0])
    >>> circuit()
    tensor([0.5, 0.5], requires_grad=True)
    """

    @abstractmethod
    def process_state(self, state: Sequence[complex], wire_order: Wires):
        """Process the given quantum state.

        Args:
            state (Sequence[complex]): quantum state with a flat shape. It may also have an
                optional batch dimension
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
        """


class MeasurementTransform(MeasurementProcess):
    """Measurement process that applies a transform into the given quantum tape. This transform
    is carried out inside the gradient black box, thus is not tracked by the gradient transform.

    Any class inheriting from ``MeasurementTransform`` should define its own ``process`` method,
    which should have the following arguments:

    * tape (QuantumTape): quantum tape to transform
    * device (pennylane.Device): device used to transform the quantum tape
    """

    @abstractmethod
    def process(self, tape, device):
        """Process the given quantum tape.

        Args:
            tape (QuantumTape): quantum tape to transform
            device (pennylane.Device): device used to transform the quantum tape
        """
