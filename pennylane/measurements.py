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
import copy
import functools
import uuid
import warnings
from enum import Enum
from typing import Generic, TypeVar

import numpy as np

import pennylane as qml
from pennylane.wires import Wires

# =============================================================================
# ObservableReturnTypes types
# =============================================================================


class ObservableReturnTypes(Enum):
    """Enumeration class to represent the return types of an observable."""

    Sample = "sample"
    Counts = "counts"
    Variance = "var"
    Expectation = "expval"
    Probability = "probs"
    State = "state"
    MidMeasure = "measure"
    VnEntropy = "vnentropy"
    MutualInfo = "mutualinfo"

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)


Sample = ObservableReturnTypes.Sample
"""Enum: An enumeration which represents sampling an observable."""

Counts = ObservableReturnTypes.Counts
"""Enum: An enumeration which represents returning the number of times
 each sample was obtained."""

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
        self,
        return_type,
        obs=None,
        wires=None,
        eigvals=None,
        id=None,
        log_base=None,
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
                numeric_type = int
            else:
                int_eigval_obs = {qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity}
                tensor_terms = [self.obs] if not hasattr(self.obs, "obs") else self.obs.obs
                every_term_standard = all(o.__class__ in int_eigval_obs for o in tensor_terms)
                numeric_type = int if every_term_standard else float
            return numeric_type

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
        shape = None

        # First: prepare the shape for return types that do not require a
        # device
        if self.return_type in (Expectation, MutualInfo, Variance, VnEntropy):
            shape = (1,)

        density_matrix_return = self.return_type == State and self.wires

        if density_matrix_return:
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

            if self.obs is not None:
                # qml.sample(some_observable) case
                return (1, device.shots)

            # qml.sample() case
            return (1, device.shots, len_wires)

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
        shot_vector = device._shot_vector
        # pylint: disable=consider-using-generator
        num_shot_elements = sum([s.copies for s in shot_vector])
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

        if not isinstance(self._wires, list):
            return self._wires

        return Wires.all_wires(self._wires)

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
            try:
                return self.obs.eigvals()
            except qml.operation.EigvalsUndefinedError:
                pass

        return self._eigvals

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

    def queue(self, context=qml.QueuingContext):
        """Append the measurement process to an annotated queue."""
        if self.obs is not None:
            context.safe_update_info(self.obs, owner=self)
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


def expval(op):
    r"""Expectation value of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    -0.4794255386042029

    Args:
        op (Observable): a quantum observable object

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not op.is_hermitian:
        warnings.warn(f"{op.name} might not be an observable.")

    return MeasurementProcess(Expectation, obs=op)


def var(op):
    r"""Variance of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    0.7701511529340698

    Args:
        op (Observable): a quantum observable object

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not op.is_hermitian:
        warnings.warn(f"{op.name} might not be an observable.")
    return MeasurementProcess(Variance, obs=op)


def sample(op=None, wires=None):
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning raw samples. If no observable is provided then basis state samples are returned
    directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or None): a quantum observable object
        wires (Sequence[int] or int or None): the wires we wish to sample from, ONLY set wires if
        op is None

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    array([ 1.,  1.,  1., -1.])

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

    Executing this QNode:

    >>> circuit(0.5)
    array([[0, 1],
           [0, 0],
           [1, 1],
           [0, 0]])

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. The one exception
        is if the QNode uses the parameter-shift method (``diff_method="parameter-shift"``), in
        which case ``qml.sample(obs)`` is interpreted as a single-shot expectation value of the
        observable ``obs``.
    """
    if op is not None and not op.is_hermitian:  # None type is also allowed for op
        warnings.warn(f"{op.name} might not be an observable.")

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is "
                "provided. The wires to sample will be determined directly from the observable."
            )
        wires = qml.wires.Wires(wires)

    return MeasurementProcess(Sample, obs=op, wires=wires)


def counts(op=None, wires=None):
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or None): a quantum observable object
        wires (Sequence[int] or int or None): the wires we wish to sample from, ONLY set wires if
        op is None

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    {-1: 2, 1: 2}

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

    Executing this QNode:

    >>> circuit(0.5)
    {'00': 3, '01': 1}

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. The one exception
        is if the QNode uses the parameter-shift method (``diff_method="parameter-shift"``), in
        which case ``qml.sample(obs)`` is interpreted as a single-shot expectation value of the
        observable ``obs``.
    """
    if op is not None and not op.is_hermitian:  # None type is also allowed for op
        warnings.warn(f"{op.name} might not be an observable.")

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is "
                "provided. The wires to sample will be determined directly from the observable."
            )
        wires = qml.wires.Wires(wires)

    return MeasurementProcess(Counts, obs=op, wires=wires)


def probs(wires=None, op=None):
    r"""Probability of each computational basis state.

    This measurement function accepts either a wire specification or
    an observable. Passing wires to the function
    instructs the QNode to return a flat array containing the
    probabilities :math:`|\langle i | \psi \rangle |^2` of measuring
    the computational basis state :math:`| i \rangle` given the current
    state :math:`| \psi \rangle`.

    Marginal probabilities may also be requested by restricting
    the wires to a subset of the full system; the size of the
    returned array will be ``[2**len(wires)]``.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> circuit()
    array([0.5, 0.5, 0. , 0. ])

    The returned array is in lexicographic order, so corresponds
    to a :math:`50\%` chance of measuring either :math:`|00\rangle`
    or :math:`|01\rangle`.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            return qml.probs(op=qml.Hermitian(H, wires=0))

    >>> circuit()

    array([0.14644661 0.85355339])

    The returned array is in lexicographic order, so corresponds
    to a :math:`14.6\%` chance of measuring the rotated :math:`|0\rangle` state
    and :math:`85.4\%` of measuring the rotated :math:`|1\rangle` state.

    Note that the output shape of this measurement process depends on whether
    the device simulates qubit or continuous variable quantum systems.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        op (Observable): Observable (with a diagonalzing_gates attribute) that rotates
         the computational basis
    """
    # pylint: disable=protected-access

    if wires is None and op is None:
        raise qml.QuantumFunctionError(
            "qml.probs requires either the wires or the observable to be passed."
        )

    if isinstance(op, qml.Hamiltonian):
        raise qml.QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if isinstance(op, (qml.ops.Sum, qml.ops.SProd, qml.ops.Prod)):  # pylint: disable=no-member
        raise qml.QuantumFunctionError(
            "Symbolic Operations are not supported for rotating probabilities yet."
        )

    if op is not None and not qml.operation.defines_diagonalizing_gates(op):
        raise qml.QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise qml.QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        return MeasurementProcess(Probability, wires=qml.wires.Wires(wires))
    return MeasurementProcess(Probability, obs=op)


def state():
    r"""Quantum state in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its state. A
    ``wires`` argument should *not* be provided since ``state()`` always returns a pure state
    describing all wires in the device.

    Note that the output shape of this measurement process depends on the
    number of wires defined for the device.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.state()

    Executing this QNode:

    >>> circuit()
    array([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

    The returned array is in lexicographic order. Hence, we have a :math:`1/\sqrt{2}` amplitude
    in both :math:`|00\rangle` and :math:`|01\rangle`.

    .. note::

        Differentiating :func:`~.state` is currently only supported when using the
        classical backpropagation differentiation method (``diff_method="backprop"``) with a
        compatible device.

    .. details::
        :title: Usage Details

        A QNode with the ``qml.state`` output can be used in a cost function with
        is then differentiated:

        >>> dev = qml.device('default.qubit', wires=2)
        >>> qml.qnode(dev, diff_method="backprop")
        ... def test(x):
        ...     qml.RY(x, wires=[0])
        ...     return qml.state()
        >>> def cost(x):
        ...     return np.abs(test(x)[0])
        >>> cost(x)
        tensor(0.98877108, requires_grad=True)
        >>> qml.grad(cost)(x)
        -0.07471906623679961
    """
    # pylint: disable=protected-access
    return MeasurementProcess(State)


def density_matrix(wires):
    r"""Quantum density matrix in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its density
    matrix or reduced density matrix. The ``wires`` argument gives the possibility
    to trace out a part of the system. It can result in obtaining a mixed state, which can be
    only represented by the reduced density matrix.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliY(wires=0)
            qml.Hadamard(wires=1)
            return qml.density_matrix([0])

    Executing this QNode:

    >>> circuit()
    array([[0.+0.j 0.+0.j]
        [0.+0.j 1.+0.j]])

    The returned matrix is the reduced density matrix, where system 1 is traced out.

    Args:
        wires (Sequence[int] or int): the wires of the subsystem

    .. note::

        Calculating the derivative of :func:`~.density_matrix` is currently only supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device.
    """
    # pylint: disable=protected-access
    wires = qml.wires.Wires(wires)
    return MeasurementProcess(State, wires=wires)


def vn_entropy(wires, log_base=None):
    r"""Von Neumann entropy of the system prior to measurement.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        wires (Sequence[int] or int): The wires of the subsystem
        log_base (float): Base for the logarithm. If None, the natural logarithm is used.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=[0])

    Executing this QNode:

    >>> circuit_entropy(np.pi/2)
    0.6931472

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit_entropy)(param)
    0.6232252401402305

    .. note::

        Calculating the derivative of :func:`~.vn_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`pennylane.qinfo.transforms.vn_entropy` and :func:`pennylane.math.vn_entropy`
    """
    wires = qml.wires.Wires(wires)
    return MeasurementProcess(VnEntropy, wires=wires, log_base=log_base)


def mutual_info(wires0, wires1, log_base=None):
    r"""Mutual information between the subsystems prior to measurement:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        wires0 (Sequence[int] or int): the wires of the first subsystem
        wires1 (Sequence[int] or int): the wires of the second subsystem
        log_base (float): Base for the logarithm. If None, the natural logarithm is used.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_mutual(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

    Executing this QNode:

    >>> circuit_mutual(np.pi/2)
    1.3862943611198906

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit_mutual)(param)
    1.2464504802804612

    .. note::

        Calculating the derivative of :func:`~.mutual_info` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`~.vn_entropy`, :func:`pennylane.qinfo.transforms.mutual_info` and :func:`pennylane.math.mutual_info`
    """
    # the subsystems cannot overlap
    if [wire for wire in wires0 if wire in wires1]:
        raise qml.QuantumFunctionError(
            "Subsystems for computing mutual information must not overlap."
        )

    wires0 = qml.wires.Wires(wires0)
    wires1 = qml.wires.Wires(wires1)
    return MeasurementProcess(MutualInfo, wires=[wires0, wires1], log_base=log_base)


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


def measure(wires):
    """Perform a mid-circuit measurement in the computational basis on the
    supplied qubit.

    Measurement outcomes can be obtained and used to conditionally apply
    operations.

    If a device doesn't support mid-circuit measurements natively, then the
    QNode will apply the :func:`defer_measurements` transform.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1)

            qml.cond(m_0, qml.RY)(y, wires=0)
            return qml.probs(wires=[0])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.90165331, 0.09834669], requires_grad=True)

    Args:
        wires (Wires): The wire of the qubit the measurement process applies to.

    Raises:
        QuantumFunctionError: if multiple wires were specified
    """
    wire = qml.wires.Wires(wires)
    if len(wire) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    MeasurementProcess(MidMeasure, wires=wire, id=measurement_id)
    return MeasurementValue(measurement_id)
