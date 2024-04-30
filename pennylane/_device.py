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
This module contains the :class:`Device` abstract base class.
"""
# pylint: disable=too-many-format-args, use-maxsplit-arg, protected-access
import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np

import pennylane as qml
from pennylane.measurements import (
    MeasurementProcess,
    CountsMP,
    Expectation,
    ExpectationMP,
    MidMeasureMP,
    Probability,
    ProbabilityMP,
    Sample,
    SampleMP,
    ShadowExpvalMP,
    State,
    Variance,
)

from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum, LinearCombination, Prod, SProd
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager


def _local_tape_expand(tape, depth, stop_at):
    """Expand all objects in a tape to a specific depth excluding measurements.
    see `pennylane.tape.tape.expand_tape` for examples.

    Args:
        tape (QuantumTape): The tape to expand
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.

    Returns:
        QuantumTape: The expanded version of ``tape``.
    """
    # This function mimics `pennylane.tape.tape.expand_tape()`, but does not expand measurements and
    # does not perform validation checks for non-commuting measurements on the same wires.
    if depth == 0:
        return tape

    new_ops = []
    new_measurements = []

    for queue, new_queue in [
        (tape.operations, new_ops),
        (tape.measurements, new_measurements),
    ]:
        for obj in queue:
            if isinstance(obj, MeasurementProcess) or stop_at(obj):
                new_queue.append(obj)
                continue

            if isinstance(obj, Operator):
                if obj.has_decomposition:
                    with QueuingManager.stop_recording():
                        obj = QuantumScript(obj.decomposition(), _update=False)
                else:
                    new_queue.append(obj)
                    continue

            # recursively expand out the newly created tape
            expanded_tape = _local_tape_expand(obj, stop_at=stop_at, depth=depth - 1)

            new_ops.extend(expanded_tape.operations)
            new_measurements.extend(expanded_tape.measurements)

    # preserves inheritance structure
    # if tape is a QuantumTape, returned object will be a quantum tape
    new_tape = tape.__class__(new_ops, new_measurements, shots=tape.shots, _update=False)

    # Update circuit info
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape._batch_size = tape._batch_size
    new_tape._output_dim = tape._output_dim
    return new_tape


class DeviceError(Exception):
    """Exception raised by a :class:`~.pennylane._device.Device` when it encounters an illegal
    operation in the quantum circuit.
    """


class Device(abc.ABC):
    """Abstract base class for PennyLane devices.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1000 if not specified.
    """

    # pylint: disable=too-many-public-methods,too-many-instance-attributes
    _capabilities = {"model": None, "supports_broadcasting": False}
    """The capabilities dictionary stores the properties of a device. Devices can add their
    own custom properties and overwrite existing ones by overriding the ``capabilities()`` method."""

    _circuits = {}  #: dict[str->Circuit]: circuit templates associated with this API class
    _asarray = staticmethod(np.asarray)

    def __init__(self, wires=1, shots=1000, *, analytic=None):
        self.shots = shots

        if analytic is not None:
            msg = (
                "The analytic argument has been replaced by shots=None. "
                "Please use shots=None instead of analytic=True."
            )
            raise DeviceError(msg)

        if not isinstance(wires, Iterable):
            # interpret wires as the number of consecutive wires
            wires = range(wires)

        self._wires = Wires(wires)
        self.num_wires = len(self._wires)
        self._wire_map = self.define_wire_map(self._wires)
        self._num_executions = 0
        self._op_queue = None
        self._obs_queue = None
        self._parameters = None

        self.tracker = qml.Tracker()
        self.custom_expand_fn = None

    def __repr__(self):
        """String representation."""
        return f"<{self.__class__.__name__} device (wires={self.num_wires}, shots={self.shots}) at {hex(id(self))}>"

    def __str__(self):
        """Verbose string representation."""
        package = self.__module__.split(".")[0]
        return (
            f"{self.name}\nShort name: {self.short_name}\n"
            f"Package: {package}\n"
            f"Plugin version: {self.version}\n"
            f"Author: {self.author}\n"
            f"Wires: {self.num_wires}\n"
            f"Shots: {self.shots}"
        )

    @property
    @abc.abstractmethod
    def name(self):
        """The full name of the device."""

    @property
    @abc.abstractmethod
    def short_name(self):
        """Returns the string used to load the device."""

    @property
    @abc.abstractmethod
    def pennylane_requires(self):
        """The current API version that the device plugin was made for."""

    @property
    @abc.abstractmethod
    def version(self):
        """The current version of the plugin."""

    @property
    @abc.abstractmethod
    def author(self):
        """The author(s) of the plugin."""

    @property
    @abc.abstractmethod
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """

    @property
    @abc.abstractmethod
    def observables(self):
        """Get the supported set of observables.

        Returns:
            set[str]: the set of PennyLane observable names the device supports
        """

    @property
    def shots(self):
        """Number of circuit evaluations/random samples used to estimate
        expectation values of observables"""
        return self._shots

    @property
    def analytic(self):
        """Whether shots is None or not. Kept for backwards compatability."""
        return self._shots is None

    @property
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

    @property
    def wire_map(self):
        """Ordered dictionary that defines the map from user-provided wire labels to
        the wire labels used on this device"""
        return self._wire_map

    @property
    def num_executions(self):
        """Number of times this device is executed by the evaluation of QNodes
        running on this device

        Returns:
            int: number of executions
        """
        return self._num_executions

    @shots.setter
    def shots(self, shots):
        """Changes the number of shots.

        Args:
            shots (int): number of circuit evaluations/random samples used to estimate
                expectation values of observables

        Raises:
            DeviceError: if number of shots is less than 1
        """
        if shots is None:
            # device is in analytic mode
            self._shots = shots
            self._shot_vector = None
            self._raw_shot_sequence = None

        elif isinstance(shots, int):
            # device is in sampling mode (unbatched)
            if shots < 1:
                raise DeviceError(
                    f"The specified number of shots needs to be at least 1. Got {shots}."
                )

            self._shots = shots
            self._shot_vector = None

        elif isinstance(shots, Sequence) and not isinstance(shots, str):
            # device is in batched sampling mode
            shot_obj = qml.measurements.Shots(shots)
            self._shots, self._shot_vector = shot_obj.total_shots, list(shot_obj.shot_vector)
            self._raw_shot_sequence = shots

        else:
            raise DeviceError(
                "Shots must be a single non-negative integer or a sequence of non-negative integers."
            )

    @property
    def shot_vector(self):
        """list[~pennylane.measurements.ShotCopies]: Returns the shot vector, a sparse
        representation of the shot sequence used by the device
        when evaluating QNodes.

        **Example**

        >>> dev = qml.device("default.qubit.legacy", wires=2, shots=[3, 1, 2, 2, 2, 2, 6, 1, 1, 5, 12, 10, 10])
        >>> dev.shots
        57
        >>> dev.shot_vector
        [ShotCopies(3 shots x 1),
         ShotCopies(1 shots x 1),
         ShotCopies(2 shots x 4),
         ShotCopies(6 shots x 1),
         ShotCopies(1 shots x 2),
         ShotCopies(5 shots x 1),
         ShotCopies(12 shots x 1),
         ShotCopies(10 shots x 2)]

        The sparse representation of the shot
        sequence is returned, where tuples indicate the number of times a shot
        integer is repeated.
        """
        return self._shot_vector

    def _has_partitioned_shots(self):
        """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
        return self._shot_vector is not None and (
            len(self._shot_vector) > 1 or self._shot_vector[0].copies > 1
        )

    def define_wire_map(self, wires):
        """Create the map from user-provided wire labels to the wire labels used by the device.

        The default wire map maps the user wire labels to wire labels that are consecutive integers.

        However, by overwriting this function, devices can specify their preferred, non-consecutive and/or non-integer
        wire labels.

        Args:
            wires (Wires): user-provided wires for this device

        Returns:
            OrderedDict: dictionary specifying the wire map

        **Example**

        >>> dev = device('my.device', wires=['b', 'a'])
        >>> dev.wire_map()
        OrderedDict( [(<Wires = ['a']>, <Wires = [0]>), (<Wires = ['b']>, <Wires = [1]>)])
        """
        consecutive_wires = Wires(range(self.num_wires))

        wire_map = zip(wires, consecutive_wires)
        return OrderedDict(wire_map)

    def order_wires(self, subset_wires):
        """Given some subset of device wires return a Wires object with the same wires;
        sorted according to the device wire map.

        Args:
            subset_wires (Wires): The subset of device wires (in any order).

        Raise:
            ValueError: Could not find some or all subset wires subset_wires in device wires device_wires.

        Return:
            ordered_wires (Wires): a new Wires object containing the re-ordered wires set
        """
        subset_lst = subset_wires.tolist()

        try:
            ordered_subset_lst = sorted(subset_lst, key=lambda label: self.wire_map[label])
        except KeyError as e:
            raise ValueError(
                f"Could not find some or all subset wires {subset_wires} in device wires {self.wires}"
            ) from e

        return Wires(ordered_subset_lst)

    @lru_cache()
    def map_wires(self, wires):
        """Map the wire labels of wires using this device's wire map.

        Args:
            wires (Wires): wires whose labels we want to map to the device's internal labelling scheme

        Returns:
            Wires: wires with new labels
        """
        try:
            mapped_wires = wires.map(self.wire_map)
        except WireError as e:
            raise WireError(
                f"Did not find some of the wires {wires} on device with wires {self.wires}."
            ) from e

        return mapped_wires

    @classmethod
    def capabilities(cls):
        """Get the capabilities of this device class.

        Inheriting classes that change or add capabilities must override this method, for example via

        .. code-block:: python

            @classmethod
            def capabilities(cls):
                capabilities = super().capabilities().copy()
                capabilities.update(
                    supports_a_new_capability=True,
                )
                return capabilities

        Returns:
            dict[str->*]: results
        """
        return cls._capabilities

    # pylint: disable=too-many-branches,unused-argument
    def execute(self, queue, observables, parameters=None, **kwargs):
        """Execute a queue of quantum operations on the device and then measure the given observables.

        For plugin developers: Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_measure`,
        :meth:`expval`, :meth:`var`, :meth:`sample`, :meth:`post_measure`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): operations to execute on the device
            observables (Iterable[~.operation.Observable]): observables to measure and return
            parameters (dict[int, list[ParameterDependency]]): Mapping from free parameter index to the list of
                :class:`Operations <pennylane.operation.Operation>` (in the queue) that depend on it.

        Keyword Args:
            return_native_type (bool): If True, return the result in whatever type the device uses
                internally, otherwise convert it into array[float]. Default: False.

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        self.check_validity(queue, observables)
        self._op_queue = queue
        self._obs_queue = observables
        self._parameters = {}
        if parameters is not None:
            self._parameters.update(parameters)

        results = []
        if self._shot_vector is not None:
            # The following warning assumes that QubitDevice.execute is stand-alone
            warnings.warn(
                "Specifying a list of shots is only supported for "
                "QubitDevice based devices. Falling back to executions using all shots in the shot list."
            )

        with self.execution_context():
            self.pre_apply()

            for operation in queue:
                self.apply(operation.name, operation.wires, operation.parameters)

            self.post_apply()

            self.pre_measure()

            for mp in observables:
                obs = mp.obs if isinstance(mp, MeasurementProcess) and mp.obs is not None else mp
                if isinstance(obs, Tensor):
                    wires = [ob.wires for ob in obs.obs]
                else:
                    wires = obs.wires

                if mp.return_type is Expectation:
                    results.append(self.expval(obs.name, wires, obs.parameters))

                elif mp.return_type is Variance:
                    results.append(self.var(obs.name, wires, obs.parameters))

                elif mp.return_type is Sample:
                    results.append(np.array(self.sample(obs.name, wires, obs.parameters)))

                elif mp.return_type is Probability:
                    results.append(list(self.probability(wires=wires).values()))

                elif mp.return_type is State:
                    raise qml.QuantumFunctionError("Returning the state is not supported")

                elif mp.return_type is not None:
                    raise qml.QuantumFunctionError(
                        f"Unsupported return type specified for observable {obs.name}"
                    )

            self.post_measure()

            self._op_queue = None
            self._obs_queue = None
            self._parameters = None

            # increment counter for number of executions of device
            self._num_executions += 1

            if self.tracker.active:
                self.tracker.update(executions=1, shots=self._shots, results=self._asarray(results))
                self.tracker.record()

            # Ensures that a combination with sample does not put
            # expvals and vars in superfluous arrays
            if all(mp.return_type is Sample for mp in observables):
                return self._asarray(results)
            if any(mp.return_type is Sample for mp in observables):
                return self._asarray(results, dtype="object")

            return self._asarray(results)

    def batch_execute(self, circuits):
        """Execute a batch of quantum circuits on the device.

        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.

        For plugin developers: This function should be overwritten if the device can efficiently run multiple
        circuits on a backend, for example using parallel and/or asynchronous executions.

        Args:
            circuits (list[.tape.QuantumTape]): circuits to execute on the device

        Returns:
            list[array[float]]: list of measured value(s)
        """
        results = []
        for circuit in circuits:
            # we need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()

            res = self.execute(circuit.operations, circuit.measurements)
            results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results

    def execute_and_gradients(self, circuits, method="jacobian", **kwargs):
        """Execute a batch of quantum circuits on the device, and return both the
        results and the gradients.

        The circuits are represented by tapes, and they are executed
        one-by-one using the device's ``execute`` method. The results and the
        corresponding Jacobians are collected in a list.

        For plugin developers: This method should be overwritten if the device
        can efficiently run multiple circuits on a backend, for example using
        parallel and/or asynchronous executions, and return both the results and the
        Jacobians.

        Args:
            circuits (list[.tape.QuantumTape]): circuits to execute on the device
            method (str): the device method to call to compute the Jacobian of a single circuit
            **kwargs: keyword argument to pass when calling ``method``

        Returns:
            tuple[list[array[float]], list[array[float]]]: Tuple containing list of measured value(s)
            and list of Jacobians. Returned Jacobians should be of shape ``(output_shape, num_params)``.
        """
        if self.tracker.active:
            self.tracker.update(execute_and_derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()
        gradient_method = getattr(self, method)

        res = []
        jacs = []

        for circuit in circuits:
            # Evaluations and gradients are paired, so that
            # devices can re-use the device state for the
            # gradient computation (if applicable).
            res.append(self.batch_execute([circuit])[0])
            jacs.append(gradient_method(circuit, **kwargs))

        return res, jacs

    def gradients(self, circuits, method="jacobian", **kwargs):
        """Return the gradients of a batch of quantum circuits on the device.

        The gradient method ``method`` is called sequentially for each
        circuit, and the corresponding Jacobians are collected in a list.

        For plugin developers: This method should be overwritten if the device
        can efficiently compute the gradient of multiple circuits on a
        backend, for example using parallel and/or asynchronous executions.

        Args:
            circuits (list[.tape.QuantumTape]): circuits to execute on the device
            method (str): the device method to call to compute the Jacobian of a single circuit
            **kwargs: keyword argument to pass when calling ``method``

        Returns:
            list[array[float]]: List of Jacobians. Returned Jacobians should be of
            shape ``(output_shape, num_params)``.
        """
        if self.tracker.active:
            self.tracker.update(derivatives=len(circuits))
            self.tracker.record()
        gradient_method = getattr(self, method)
        return [gradient_method(circuit, **kwargs) for circuit in circuits]

    @property
    def stopping_condition(self):
        """.BooleanFn: Returns the stopping condition for the device. The returned
        function accepts a queuable object (including a PennyLane operation
        and observable) and returns ``True`` if supported by the device."""
        return qml.BooleanFn(
            lambda obj: not isinstance(obj, QuantumScript)
            and (isinstance(obj, MeasurementProcess) or self.supports_operation(obj.name))
        )

    def custom_expand(self, fn):
        """Register a custom expansion function for the device.

        **Example**

        .. code-block:: python

            dev = qml.device("default.qubit.legacy", wires=2)

            @dev.custom_expand
            def my_expansion_function(self, tape, max_expansion=10):
                ...
                # can optionally call the default device expansion
                tape = self.default_expand_fn(tape, max_expansion=max_expansion)
                return tape

        The custom device expansion function must have arguments
        ``self`` (the device object), ``tape`` (the input circuit
        to transform and execute), and ``max_expansion`` (the number of
        times the circuit should be expanded).

        The default :meth:`~.default_expand_fn` method of the original
        device may be called. It is highly recommended to call this
        before returning, to ensure that the expanded circuit is supported
        on the device.
        """
        self.custom_expand_fn = types.MethodType(fn, self)

    def default_expand_fn(self, circuit, max_expansion=10):
        """Method for expanding or decomposing an input circuit.
        This method should be overwritten if custom expansion logic is
        required.

        By default, this method expands the tape if:

        - state preparation operations are called mid-circuit,
        - nested tapes are present,
        - any operations are not supported on the device, or
        - multiple observables are measured on the same wire.

        Args:
            circuit (.QuantumTape): the circuit to expand.
            max_expansion (int): The number of times the circuit should be
                expanded. Expansion occurs when an operation or measurement is not
                supported, and results in a gate decomposition. If any operations
                in the decomposition remain unsupported by the device, another
                expansion occurs.

        Returns:
            .QuantumTape: The expanded/decomposed circuit, such that the device
            will natively support all operations.
        """
        # pylint: disable=protected-access
        if max_expansion == 0:
            return circuit

        expand_state_prep = any(isinstance(op, StatePrepBase) for op in circuit.operations[1:])

        if expand_state_prep:  # expand mid-circuit StatePrepBase operations
            circuit = expand_tape_state_prep(circuit)

        comp_basis_sampled_multi_measure = (
            len(circuit.measurements) > 1 and circuit.samples_computational_basis
        )
        obs_on_same_wire = len(circuit._obs_sharing_wires) > 0 or comp_basis_sampled_multi_measure
        obs_on_same_wire &= not any(
            isinstance(o, (Hamiltonian, LinearCombination)) for o in circuit._obs_sharing_wires
        )
        ops_not_supported = not all(self.stopping_condition(op) for op in circuit.operations)

        if obs_on_same_wire:
            circuit = circuit.expand(depth=max_expansion, stop_at=self.stopping_condition)

        elif ops_not_supported:
            circuit = _local_tape_expand(
                circuit, depth=max_expansion, stop_at=self.stopping_condition
            )
            circuit._update()

        return circuit

    def expand_fn(self, circuit, max_expansion=10):
        """Method for expanding or decomposing an input circuit.
        Can be the default or a custom expansion method, see
        :meth:`.Device.default_expand_fn` and :meth:`.Device.custom_expand` for more
        details.

        Args:
            circuit (.QuantumTape): the circuit to expand.
            max_expansion (int): The number of times the circuit should be
                expanded. Expansion occurs when an operation or measurement is not
                supported, and results in a gate decomposition. If any operations
                in the decomposition remain unsupported by the device, another
                expansion occurs.

        Returns:
            .QuantumTape: The expanded/decomposed circuit, such that the device
            will natively support all operations.
        """
        if self.custom_expand_fn is not None:
            # pylint:disable=not-callable
            return self.custom_expand_fn(circuit, max_expansion=max_expansion)

        return self.default_expand_fn(circuit, max_expansion=max_expansion)

    def batch_transform(self, circuit: QuantumTape):
        """Apply a differentiable batch transform for preprocessing a circuit
        prior to execution. This method is called directly by the QNode, and
        should be overwritten if the device requires a transform that
        generates multiple circuits prior to execution.

        By default, this method contains logic for generating multiple
        circuits, one per term, of a circuit that terminates in ``expval(H)``,
        if the underlying device does not support Hamiltonian expectation values,
        or if the device requires finite shots.

        .. warning::

            This method will be tracked by autodifferentiation libraries,
            such as Autograd, JAX, TensorFlow, and Torch. Please make sure
            to use ``qml.math`` for autodiff-agnostic tensor processing
            if required.

        Args:
            circuit (.QuantumTape): the circuit to preprocess

        Returns:
            tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
            the sequence of circuits to be executed, and a post-processing function
            to be applied to the list of evaluated circuit results.
        """
        supports_hamiltonian = self.supports_observable("Hamiltonian")
        supports_sum = self.supports_observable("Sum")
        finite_shots = self.shots is not None
        grouping_known = all(
            obs.grouping_indices is not None
            for obs in circuit.observables
            if isinstance(obs, (Hamiltonian, LinearCombination))
        )
        # device property present in braket plugin
        use_grouping = getattr(self, "use_grouping", True)

        hamiltonian_in_obs = any(
            isinstance(obs, (Hamiltonian, LinearCombination)) for obs in circuit.observables
        )

        expval_sum_or_prod_in_obs = any(
            isinstance(m.obs, (Sum, Prod, SProd)) and isinstance(m, ExpectationMP)
            for m in circuit.measurements
        )

        is_shadow = any(isinstance(m, ShadowExpvalMP) for m in circuit.measurements)

        hamiltonian_unusable = not supports_hamiltonian or (finite_shots and not is_shadow)

        if hamiltonian_in_obs and (hamiltonian_unusable or (use_grouping and grouping_known)):
            # If the observable contains a Hamiltonian and the device does not
            # support Hamiltonians, or if the simulation uses finite shots, or
            # if the Hamiltonian explicitly specifies an observable grouping,
            # split tape into multiple tapes of diagonalizable known observables.
            try:
                circuits, hamiltonian_fn = qml.transforms.hamiltonian_expand(circuit, group=False)
            except ValueError:
                circuits, hamiltonian_fn = qml.transforms.sum_expand(circuit)

        elif expval_sum_or_prod_in_obs and not is_shadow and not supports_sum:
            circuits, hamiltonian_fn = qml.transforms.sum_expand(circuit)

        elif (
            len(circuit._obs_sharing_wires) > 0
            and not hamiltonian_in_obs
            and all(
                not isinstance(m, (SampleMP, ProbabilityMP, CountsMP)) for m in circuit.measurements
            )
        ):
            # Check for case of non-commuting terms and that there are no Hamiltonians
            # TODO: allow for Hamiltonians in list of observables as well.
            circuits, hamiltonian_fn = qml.transforms.split_non_commuting(circuit)

        else:
            # otherwise, return the output of an identity transform
            circuits = [circuit]

            def hamiltonian_fn(res):
                return res[0]

        # Check whether the circuit was broadcasted (then the Hamiltonian-expanded
        # ones will be as well) and whether broadcasting is supported
        if circuit.batch_size is None or self.capabilities().get("supports_broadcasting"):
            # If the circuit wasn't broadcasted or broadcasting is supported, no action required
            return circuits, hamiltonian_fn

        # Expand each of the broadcasted Hamiltonian-expanded circuits
        expanded_tapes, expanded_fn = qml.transforms.broadcast_expand(circuits)

        # Chain the postprocessing functions of the broadcasted-tape expansions and the Hamiltonian
        # expansion. Note that the application order is reversed compared to the expansion order,
        # i.e. while we first applied `hamiltonian_expand` to the tape, we need to process the
        # results from the broadcast expansion first.
        def total_processing(results):
            return hamiltonian_fn(expanded_fn(results))

        return expanded_tapes, total_processing

    @property
    def op_queue(self):
        """The operation queue to be applied.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Raises:
            ValueError: if outside of the execution context

        Returns:
            list[~.operation.Operation]
        """
        if self._op_queue is None:
            raise ValueError("Cannot access the operation queue outside of the execution context!")

        return self._op_queue

    @property
    def obs_queue(self):
        """The observables to be measured and returned.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Raises:
            ValueError: if outside of the execution context

        Returns:
            list[~.operation.Observable]
        """
        if self._obs_queue is None:
            raise ValueError(
                "Cannot access the observable value queue outside of the execution context!"
            )

        return self._obs_queue

    @property
    def parameters(self):
        """Mapping from free parameter index to the list of
        :class:`Operations <~.Operation>` in the device queue that depend on it.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Raises:
            ValueError: if outside of the execution context

        Returns:
            dict[int->list[ParameterDependency]]: the mapping
        """
        if self._parameters is None:
            raise ValueError(
                "Cannot access the free parameter mapping outside of the execution context!"
            )

        return self._parameters

    def pre_apply(self):
        """Called during :meth:`execute` before the individual operations are executed."""

    def post_apply(self):
        """Called during :meth:`execute` after the individual operations have been executed."""

    def pre_measure(self):
        """Called during :meth:`execute` before the individual observables are measured."""

    def post_measure(self):
        """Called during :meth:`execute` after the individual observables have been measured."""

    def execution_context(self):
        """The device execution context used during calls to :meth:`execute`.

        You can overwrite this function to return a context manager in case your
        quantum library requires that;
        all operations and method calls (including :meth:`apply` and :meth:`expval`)
        are then evaluated within the context of this context manager (see the
        source of :meth:`execute` for more details).
        """

        # pylint: disable=no-self-use
        class MockContext:  # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute()."""

            def __enter__(self):
                pass

            def __exit__(self, type, value, traceback):
                pass

        return MockContext()

    def supports_operation(self, operation):
        """Checks if an operation is supported by this device.

        Args:
            operation (type or str): operation to be checked

        Raises:
            ValueError: if `operation` is not a :class:`~.Operation` class or string

        Returns:
            bool: ``True`` if supplied operation is supported
        """
        if isinstance(operation, type) and issubclass(operation, Operation):
            return operation.__name__ in self.operations
        if isinstance(operation, str):
            return operation in self.operations

        raise ValueError(
            "The given operation must either be a pennylane.Operation class or a string."
        )

    def supports_observable(self, observable):
        """Checks if an observable is supported by this device. Raises a ValueError,
         if not a subclass or string of an Observable was passed.

        Args:
            observable (type or str): observable to be checked

        Raises:
            ValueError: if `observable` is not a :class:`~.Observable` class or string

        Returns:
            bool: ``True`` iff supplied observable is supported
        """
        if isinstance(observable, type) and issubclass(observable, Observable):
            return observable.__name__ in self.observables
        if isinstance(observable, str):
            return observable in self.observables

        raise ValueError(
            "The given observable must either be a pennylane.Observable class or a string."
        )

    def check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device

        Raises:
            DeviceError: if there are operations in the queue or observables that the device does
                not support
        """

        for o in queue:
            operation_name = o.name

            if isinstance(o, MidMeasureMP) and not self.capabilities().get(
                "supports_mid_measure", False
            ):
                raise DeviceError(
                    f"Mid-circuit measurements are not natively supported on device {self.short_name}. "
                    "Apply the @qml.defer_measurements decorator to your quantum function to "
                    "simulate the application of mid-circuit measurements on this device."
                )

            if isinstance(o, qml.Projector):
                raise ValueError(f"Postselection is not supported on the {self.name} device.")

            if not self.stopping_condition(o):
                raise DeviceError(
                    f"Gate {operation_name} not supported on device {self.short_name}"
                )

        for o in observables:
            if isinstance(o, MeasurementProcess):
                o = o.obs
                if o is None:
                    continue

            if isinstance(o, Tensor):
                # TODO: update when all capabilities keys changed to "supports_tensor_observables"
                supports_tensor = self.capabilities().get(
                    "supports_tensor_observables", False
                ) or self.capabilities().get("tensor_observables", False)
                if not supports_tensor:
                    raise DeviceError(
                        f"Tensor observables not supported on device {self.short_name}"
                    )

                for i in o.obs:
                    if not self.supports_observable(i.name):
                        raise DeviceError(
                            f"Observable {i.name} not supported on device {self.short_name}"
                        )

            elif isinstance(o, qml.ops.Prod):

                supports_prod = self.supports_observable(o.name)
                if not supports_prod:
                    raise DeviceError(f"Observable Prod not supported on device {self.short_name}")

                simplified_op = o.simplify()
                if isinstance(simplified_op, qml.ops.Prod):
                    for i in o.simplify().operands:
                        if not self.supports_observable(i.name):
                            raise DeviceError(
                                f"Observable {i.name} not supported on device {self.short_name}"
                            )

            else:
                observable_name = o.name

                if not self.supports_observable(observable_name):
                    raise DeviceError(
                        f"Observable {observable_name} not supported on device {self.short_name}"
                    )

    @abc.abstractmethod
    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        For plugin developers: this function should apply the operation on the device.

        Args:
            operation (str): name of the operation
            wires (Wires): wires that the operation is applied to
            par (tuple): parameters for the operation
        """

    @abc.abstractmethod
    def expval(self, observable, wires, par):
        r"""Returns the expectation value of observable on specified wires.

        Note: all arguments accept _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) are to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """

    def var(self, observable, wires, par):
        r"""Returns the variance of observable on specified wires.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support variance computation

        Returns:
            float: variance :math:`\mathrm{var}(A) = \bra{\psi}A^2\ket{\psi} - \bra{\psi}A\ket{\psi}^2`
        """
        raise NotImplementedError(
            f"Returning variances from QNodes not currently supported by {self.short_name}"
        )

    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        The number of samples is determined by the value of ``Device.shots``,
        which can be directly modified.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support sampling

        Returns:
            array[float]: samples in an array of dimension ``(shots,)``
        """
        raise NotImplementedError(
            f"Returning samples from QNodes not currently supported by {self.short_name}"
        )

    def probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        raise NotImplementedError(
            f"Returning probability not currently supported by {self.short_name}"
        )

    @abc.abstractmethod
    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
