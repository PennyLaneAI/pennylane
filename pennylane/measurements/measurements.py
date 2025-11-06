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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pennylane import math
from pennylane.capture import ABCCaptureMeta
from pennylane.capture import enabled as capture_enabled
from pennylane.exceptions import (
    DecompositionUndefinedError,
    EigvalsUndefinedError,
    PennyLaneDeprecationWarning,
    QuantumFunctionError,
)
from pennylane.math.utils import is_abstract
from pennylane.operation import Operator, _get_abstract_operator
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .capture_measurements import (
    create_measurement_mcm_primitive,
    create_measurement_obs_primitive,
    create_measurement_wires_primitive,
)
from .measurement_value import MeasurementValue


class MeasurementProcess(ABC, metaclass=ABCCaptureMeta):
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

    _shortname = None

    _obs_primitive: Optional["jax.extend.core.Primitive"] = None
    _wires_primitive: Optional["jax.extend.core.Primitive"] = None
    _mcm_primitive: Optional["jax.extend.core.Primitive"] = None

    def __init_subclass__(cls, **_):
        register_pytree(cls, cls._flatten, cls._unflatten)
        name = cls._shortname or cls.__name__
        cls._wires_primitive = create_measurement_wires_primitive(cls, name=name)
        cls._obs_primitive = create_measurement_obs_primitive(cls, name=name)
        cls._mcm_primitive = create_measurement_mcm_primitive(cls, name=name)

    @classmethod
    def _primitive_bind_call(cls, obs=None, wires=None, eigvals=None, id=None, **kwargs):
        """Called instead of ``type.__call__`` if ``qml.capture.enabled()``.

        Measurements have three "modes":

        1) Wires or wires + eigvals
        2) Observable
        3) Mid circuit measurements

        Not all measurements support all three modes. For example, ``VNEntropyMP`` does not
        allow being specified via an observable. But we handle the generic case here.

        """
        if cls._obs_primitive is None:
            # safety check if primitives aren't set correctly.
            return type.__call__(cls, obs=obs, wires=wires, eigvals=eigvals, id=id, **kwargs)
        if obs is None:
            wires = () if wires is None else wires
            if eigvals is None:
                out = cls._wires_primitive.bind(*wires, **kwargs)  # wires
                return tuple(out) if isinstance(out, list) else out
            return cls._wires_primitive.bind(
                *wires, eigvals, has_eigvals=True, **kwargs
            )  # wires + eigvals

        if isinstance(obs, Operator) or isinstance(
            getattr(obs, "aval", None), _get_abstract_operator()
        ):
            return cls._obs_primitive.bind(obs, **kwargs)
        if isinstance(obs, (list, tuple)):
            return cls._mcm_primitive.bind(*obs, single_mcm=False, **kwargs)  # iterable of mcms
        return cls._mcm_primitive.bind(obs, single_mcm=True, **kwargs)  # single mcm

    # pylint: disable=unused-argument
    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ) -> tuple[tuple, type]:
        """Calculate the shape and dtype that will be returned when a measurement is performed.

        This information is similar to ``numeric_type`` and ``shape``, but is provided through
        a class method and does not require the creation of an instance.

        Note that ``shots`` should strictly be ``None`` or ``int``. Shot vectors are handled higher
        in the stack.

        If ``n_wires is None``, then the measurement process contains an observable. An integer
        ``n_wires`` can correspond either to the number of wires or to the number of mid circuit
        measurements. ``n_wires = 0`` indicates a measurement that is broadcasted across all device wires.

        >>> ProbabilityMP._abstract_eval(n_wires=2)
        ((4,), float)
        >>> ProbabilityMP._abstract_eval(n_wires=0, num_device_wires=2)
        ((4,), float)
        >>> SampleMP._abstract_eval(n_wires=0, shots=50, num_device_wires=2)
        ((50, 2), int)
        >>> SampleMP._abstract_eval(n_wires=4, has_eigvals=True, shots=50)
        ((50,), float)
        >>> SampleMP._abstract_eval(n_wires=None, shots=50)
        ((50,), float)

        """
        return (), float

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

    def __init__(
        self,
        obs: None | (Operator | MeasurementValue | Sequence[MeasurementValue]) = None,
        wires: Wires | None = None,
        eigvals: TensorLike | None = None,
        id: str | None = None,
    ):
        if getattr(obs, "name", None) == "MeasurementValue" or isinstance(obs, Sequence):
            # Cast sequence of measurement values to list
            self.mv = obs if getattr(obs, "name", None) == "MeasurementValue" else list(obs)
            self.obs = None
        elif is_abstract(obs):  # Catalyst program with qml.sample(m, wires=i)
            self.mv = obs
            self.obs = None
        else:
            self.obs = obs
            self.mv = None

        self.id = id

        if wires is not None:
            if not capture_enabled() and len(wires) == 0:
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

            self._eigvals = math.asarray(eigvals)

        # Queue the measurement process
        self.queue()

    @property
    def numeric_type(self) -> type:
        """The Python numeric type of the measurement result.

        Returns:
            type: The output numeric type; ``int``, ``float`` or ``complex``.

        Raises:
            QuantumFunctionError: the return type of the measurement process is
                unrecognized and cannot deduce the numeric type
        """
        raise QuantumFunctionError(
            f"The numeric type of the measurement {self.__class__.__name__} is not defined."
        )

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple[int, ...]:
        """Calculate the shape of the result object tensor.

        Args:
            shots (Optional[int]) = None: the number of shots used execute the circuit. ``None``
               indicates an analytic simulation.  Shot vectors are handled by calling this method
               multiple times.
            num_device_wires (int)=0 : The number of wires that will be used if the measurement is
               broadcasted across all available wires (``len(mp.wires) == 0``). If the device
               itself doesn't provide a number of wires, the number of tape wires will be provided
               here instead:

        Returns:
            tuple[int,...]: An arbitrary length tuple of ints.  May be an empty tuple.

        >>> qml.probs(wires=(0,1)).shape()
        (4,)
        >>> qml.sample(wires=(0,1)).shape(shots=50)
        (50, 2)
        >>> qml.state().shape(num_device_wires=4)
        (16,)
        >>> qml.expval(qml.Z(0)).shape()
        ()

        """
        raise QuantumFunctionError(
            f"The shape of the measurement {self.__class__.__name__} is not defined"
        )

    @QueuingManager.stop_recording()
    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[.Operation]: the operations that diagonalize the observables
        """
        return self.obs.diagonalizing_gates() if self.obs else []

    def __eq__(self, other):
        # tach-ignore
        from pennylane import equal  # pylint: disable=import-outside-toplevel

        return equal(self, other)

    def __hash__(self):
        return self.hash

    def __repr__(self):
        """Representation of this class."""
        name_str = self._shortname or type(self).__name__

        if self.mv is not None:
            return f"{name_str}({repr(self.mv)})"
        if self.obs:
            return f"{name_str}({self.obs})"
        if self._eigvals is not None:
            return f"{name_str}(eigvals={self._eigvals}, wires={self.wires.tolist()})"

        # Todo: when tape is core the return type will always be taken from the MeasurementProcess
        name_str = self._shortname or "None"
        return f"{name_str}(wires={self.wires.tolist()})"

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
        if self.mv is not None and not is_abstract(self.mv):
            if isinstance(self.mv, list):
                return Wires.all_wires([m.wires for m in self.mv])
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
                # "Eigvals" should be the processed values for all branches of a MeasurementValue
                _, processed_values = tuple(zip(*self.mv.items()))
                interface = math.get_deep_interface(processed_values)
                return math.asarray(processed_values, like=interface)
            return math.arange(0, 2 ** len(self.wires), 1)

        if self.obs is not None:
            try:
                return self.obs.eigvals()
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

        .. warning::

            This method is deprecated due to circular dependency issues and lack of use.

            The relevant code can be reproduced by:

            .. code-block:: python

                diagonalized_mp = type(mp)(eigvals=mp.eigvals(), wires=mp.wires)
                qml.tape.QuantumScript(mp.diagonalizing_gates(), [diagonalized_mp])

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
        warnings.warn(
            "MeasurementProcess.expand is deprecated. Use diagonalizing_gates and eigvals manually instead.",
            PennyLaneDeprecationWarning,
        )
        if self.obs is None:
            raise DecompositionUndefinedError

        with AnnotatedQueue() as q:
            self.obs.diagonalizing_gates()
            self.__class__(wires=self.obs.wires, eigvals=self.obs.eigvals())

        # tach-ignore
        from pennylane.tape import QuantumScript  # pylint: disable=import-outside-toplevel

        return QuantumScript.from_queue(q)

    def queue(self, context=QueuingManager):
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

    >>> from functools import partial
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @partial(qml.set_shots, shots=1000)
    ... @qml.qnode(dev)
    ... def circuit():
    ...     qml.X(0)
    ...     return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])
    ...
    >>> circuit()
    (tensor(1000, requires_grad=True), tensor(0, requires_grad=True))
    """

    _shortname = "sample"

    @abstractmethod
    def process_samples(
        self,
        samples: TensorLike,
        wire_order: Wires,
        shot_range: None | tuple[int] = None,
        bin_size: None | int = None,
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
    def process_state(self, state: TensorLike, wire_order: Wires):
        """Process the given quantum state.

        Args:
            state (TensorLike): quantum state with a flat shape. It may also have an
                optional batch dimension
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
        """

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        """
        Process the given density matrix.

        Args:
            density_matrix (TensorLike): The density matrix representing the (mixed) quantum state,
                which may be single or batched. For a single matrix, the shape should be ``(2^n, 2^n)``
                where `n` is the number of wires the matrix acts upon. For batched matrices, the shape
                should be ``(batch_size, 2^n, 2^n)``.
            wire_order (Wires): The wires determining the subspace that the ``density_matrix`` acts on.
                A matrix of dimension :math:`2^n` acts on a subspace of :math:`n` wires. This parameter specifies
                the mapping of matrix dimensions to physical qubits, allowing the function to correctly
                trace out the subsystems not involved in the measurement or operation.
        """
        raise NotImplementedError


class MeasurementTransform(MeasurementProcess):
    """Measurement process that applies a transform into the given quantum tape. This transform
    is carried out inside the gradient black box, thus is not tracked by the gradient transform.

    Any class inheriting from ``MeasurementTransform`` should define its own ``process`` method,
    which should have the following arguments:

    * tape (QuantumTape): quantum tape to transform
    * device (pennylane.devices.LegacyDevice): device used to transform the quantum tape
    """

    @abstractmethod
    def process(self, tape, device):
        """Process the given quantum tape.

        Args:
            tape (QuantumTape): quantum tape to transform
            device (pennylane.devices.LegacyDevice): device used to transform the quantum tape
        """
