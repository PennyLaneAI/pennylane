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
This module defines the QuantumScript object responsible for storing quantum operations and measurements to be
executed by a device.
"""
# pylint: disable=too-many-instance-attributes, protected-access, too-many-public-methods

import contextlib
import copy
import warnings
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from functools import cached_property
from typing import Any, ParamSpec, TypeVar, Union

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.measurements import MeasurementProcess
from pennylane.measurements.shots import Shots, ShotsLike
from pennylane.operation import _UNSET_BATCH_SIZE, Operation, Operator
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

QS = TypeVar("QS", bound="QuantumScript")


class QuantumScript:
    r"""The operations and measurements that represent instructions for
    execution on a quantum device.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed

    Keyword Args:
        shots (None, int, Sequence[int], ~.Shots): Number and/or batches of shots for execution.
            Note that this property is still experimental and under development.
        trainable_params (None, Sequence[int]): the indices for which parameters are trainable

    .. seealso:: :class:`pennylane.tape.QuantumTape`

    **Example:**

    .. code-block:: python

        from pennylane.tape import QuantumScript

        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a")),
               qml.RX(0.432, 0),
               qml.RY(0.543, 0),
               qml.CNOT((0,"a")),
               qml.RX(0.133, "a")]

        qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

    >>> list(qscript)
    [BasisState(array([1, 1]), wires=[0, 'a']), RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a']), expval(Z(0))]
    >>> qscript.operations
    [BasisState(array([1, 1]), wires=[0, 'a']), RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
    >>> qscript.measurements
    [expval(Z(0))]

    Iterating over the quantum script can be done by:

    >>> for op in qscript:
    ...     print(op)
    BasisState(array([1, 1]), wires=[0, 'a'])
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(Z(0))

    Quantum scripts also support indexing and length determination:

    >>> qscript[0]
    BasisState(array([1, 1]), wires=[0, 'a'])
    >>> len(qscript)
    6

    Once constructed, the script can be executed directly on a quantum device
    using the :func:`~.pennylane.execute` function:

    >>> dev = qml.device('default.qubit', wires=(0,'a'))
    >>> qml.execute([qscript], dev, diff_method=None)
    (np.float64(-0.7775069381227451),)

    Quantum scripts can also store information about the number and batches of
    executions by setting the ``shots`` keyword argument. This information is internally
    stored in a :class:`pennylane.measurements.Shots` object:

    >>> s_vec = [1, 1, 2, 2, 2]
    >>> qscript = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.Z(0))], shots=s_vec)
    >>> qscript.shots.shot_vector
    (ShotCopies(1 shots x 2), ShotCopies(2 shots x 3))

    ``ops`` and ``measurements`` are converted to lists upon initialization,
    so those arguments accept any iterable object:

    >>> qscript = QuantumScript((qml.X(i) for i in range(3)))
    >>> qscript.circuit
    [X(0), X(1), X(2)]

    """

    def _flatten(self):
        return (self._ops, self.measurements), (self.shots, tuple(self.trainable_params))

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, shots=metadata[0], trainable_params=metadata[1])

    def __init__(
        self,
        ops: Iterable[Operator] | None = None,
        measurements: Iterable[MeasurementProcess] | None = None,
        shots: ShotsLike | None = None,
        trainable_params: Sequence[int] | None = None,
    ):
        self._ops = [] if ops is None else list(ops)
        self._measurements = [] if measurements is None else list(measurements)
        self._shots = Shots(shots)

        self._trainable_params = trainable_params
        self._graph = None
        self._specs = None
        self._batch_size = _UNSET_BATCH_SIZE

        self._obs_sharing_wires = None
        self._obs_sharing_wires_id = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: wires={self.wires.tolist()}, params={self.num_params}>"

    @cached_property
    def hash(self) -> int:
        """int: returns an integer hash uniquely representing the quantum script"""
        fingerprint = []
        fingerprint.extend(op.hash for op in self.operations)
        fingerprint.extend(m.hash for m in self.measurements)
        fingerprint.extend(self.trainable_params)
        fingerprint.extend(self.shots)
        return hash(tuple(fingerprint))

    def __iter__(self) -> Iterator[Operator | MeasurementProcess]:
        """Iterator[.Operator, .MeasurementProcess]: Return an iterator to the
        underlying quantum circuit object."""
        return iter(self.circuit)

    def __getitem__(self, idx: int) -> Operator | MeasurementProcess:
        """Union[Operator, MeasurementProcess]: Return the indexed operator from underlying quantum
        circuit object."""
        return self.circuit[idx]

    def __len__(self) -> int:
        """int: Return the number of operations and measurements in the
        underlying quantum circuit object."""
        return len(self.circuit)

    # ========================================================
    # QSCRIPT properties
    # ========================================================

    @property
    def circuit(self) -> list[Operator | MeasurementProcess]:
        """Returns the underlying quantum circuit as a list of operations and measurements.

        The circuit is created with the assumptions that:

        * The ``operations`` attribute contains quantum operations and
          mid-circuit measurements and
        * The ``measurements`` attribute contains terminal measurements.

        Note that the resulting list could contain MeasurementProcess objects
        that some devices may not support.

        Returns:

            list[.Operator, .MeasurementProcess]: the quantum circuit
            containing quantum operations and measurements
        """
        return self.operations + self.measurements

    @property
    def operations(self) -> list[Operator]:
        """Returns the state preparations and operations on the quantum script.

        Returns:
            list[.Operator]: quantum operations

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.operations
        [StatePrep(array([0, 1]), wires=[0]), RX(0.432, wires=[0])]
        """
        return self._ops

    @property
    def observables(self) -> list[MeasurementProcess | Operator]:
        """Returns the observables on the quantum script.

        Returns:
            list[.MeasurementProcess, .Operator]]: list of observables

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.observables
        [Z(0)]
        """
        # TODO: modify this property once devices
        # have been refactored to accept and understand recieving
        # measurement processes rather than specific observables.
        obs = []

        for m in self.measurements:
            if m.obs is not None:
                obs.append(m.obs)
            else:
                obs.append(m)

        return obs

    @property
    def measurements(self) -> list[MeasurementProcess]:
        """Returns the measurements on the quantum script.

        Returns:
            list[.MeasurementProcess]: list of measurement processes

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.measurements
        [expval(Z(0))]
        """
        return self._measurements

    @property
    def samples_computational_basis(self) -> bool:
        """Determines if any of the measurements are in the computational basis."""
        return any(o.samples_computational_basis for o in self.measurements)

    @property
    def num_params(self) -> int:
        """Returns the number of trainable parameters on the quantum script."""
        return len(self.trainable_params)

    @property
    def batch_size(self) -> int | None:
        r"""The batch size of the quantum script inferred from the batch sizes
        of the used operations for parameter broadcasting.

        .. seealso:: :attr:`~.Operator.batch_size` for details.

        Returns:
            int or None: The batch size of the quantum script if present, else ``None``.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._update_batch_size()
        return self._batch_size

    @property
    def diagonalizing_gates(self) -> list[Operation]:
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[~.Operation]: the operations that diagonalize the observables

        **Examples**

        For a tape with a single observable, we get the diagonalizing gate of that observable:

        >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))])
        >>> tape.diagonalizing_gates
        [H(0)]

        If the tape includes multiple observables, they are each diagonalized individually:

        >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0)), qml.var(qml.Y(1))])
        >>> tape.diagonalizing_gates
        [H(0), Z(1), S(1), H(1)]

        .. warning::
            If the tape contains multiple observables acting on the same wire,
            then ``tape.diagonalizing_gates`` will include multiple conflicting
            diagonalizations.

            For example:

            >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0)), qml.var(qml.Y(0))])
            >>> tape.diagonalizing_gates
            [H(0), Z(0), S(0), H(0)]

            If it is relevant for your application, applying
            :func:`~.pennylane.transforms.split_non_commuting` to a tape will split it into multiple
            tapes with only qubit-wise commuting observables.

        Generally, composite operators are handled by diagonalizing their component parts, for example:

        >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0)+qml.Y(1))])
        >>> tape.diagonalizing_gates
        [H(0), Z(1), S(1), H(1)]

        However, for operators that contain multiple terms on the same wire, a single diagonalizing
        operator will be returned that diagonalizes the full operator as a unit:

        >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0)+qml.Y(0))])
        >>> tape.diagonalizing_gates[0] # doctest: +SKIP
        QubitUnitary(array([[-0.70710678+0.j ,  0.5       -0.5j],
           [-0.70710678-0.j , -0.5       +0.5j]]), wires=[0])
        """
        rotation_gates = []

        with qml.queuing.QueuingManager.stop_recording():
            for observable in _get_base_obs(self.observables):
                # some observables do not have diagonalizing gates,
                # in which case we just don't append any
                with contextlib.suppress(qml.operation.DiagGatesUndefinedError):
                    rotation_gates.extend(observable.diagonalizing_gates())
        return rotation_gates

    @property
    def shots(self) -> Shots:
        """Returns a ``Shots`` object containing information about the number
        and batches of shots

        Returns:
            ~.Shots: Object with shot information
        """
        return self._shots

    @property
    def num_preps(self) -> int:
        """Returns the index of the first operator that is not an StatePrepBase operator."""
        idx = 0
        num_ops = len(self.operations)
        while idx < num_ops and isinstance(self.operations[idx], qml.operation.StatePrepBase):
            idx += 1
        return idx

    @property
    def op_wires(self) -> Wires:
        """Returns the wires that the tape operations act on."""
        return Wires.all_wires(op.wires for op in self.operations)

    ##### Update METHODS ###############

    def _update(self):
        """Update all internal metadata regarding processed operations and observables"""
        self._graph = None
        self._specs = None
        self._trainable_params = None

        try:
            # Invalidate cached properties so they get recalculated
            del self.wires
            del self.par_info
            del self.hash
        except AttributeError:
            pass

    @cached_property
    def wires(self) -> Wires:
        """Returns the wires used in the quantum script process

        Returns:
            ~.Wires: wires in quantum script process
        """
        return Wires.all_wires(dict.fromkeys(op.wires for op in self))

    @property
    def num_wires(self) -> int:
        """Returns the number of wires in the quantum script process

        Returns:
            int: number of wires in quantum script process
        """
        return len(self.wires)

    @cached_property
    def par_info(self) -> list[dict[str, int | Operator]]:
        """Returns the parameter information of the operations and measurements in the quantum script.

        Returns:
            list[dict[str, Operator or int]]: list of dictionaries with parameter information.

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0), qml.CNOT((0,1))]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.par_info
        [{'op': StatePrep(array([0, 1]), wires=[0]), 'op_idx': 0, 'p_idx': 0},
        {'op': RX(0.432, wires=[0]), 'op_idx': 1, 'p_idx': 0}]

        Note that the operations and measurements included in this list are only the ones that have parameters
        """
        par_info = []
        for idx, op in enumerate(self.operations):
            par_info.extend({"op": op, "op_idx": idx, "p_idx": i} for i, d in enumerate(op.data))

        n_ops = len(self.operations)
        for idx, m in enumerate(self.measurements):
            if m.obs is not None:
                par_info.extend(
                    {"op": m.obs, "op_idx": idx + n_ops, "p_idx": i}
                    for i, d in enumerate(m.obs.data)
                )
        return par_info

    @property
    def obs_sharing_wires(self) -> list[Operator]:
        """Returns the subset of the observables that share wires with another observable,
        i.e., that do not have their own unique set of wires.

        Returns:
            list[~.Operator]: list of observables with shared wires.

        """
        if self._obs_sharing_wires is None:
            self._update_observables()
        return self._obs_sharing_wires

    @property
    def obs_sharing_wires_id(self) -> list[int]:
        """Returns the indices subset of the observables that share wires with another observable,
        i.e., that do not have their own unique set of wires.

        Returns:
            list[int]: list of indices from observables with shared wires.

        """
        if self._obs_sharing_wires_id is None:
            self._update_observables()
        return self._obs_sharing_wires_id

    def _update_observables(self):
        """Update information about observables, including the wires that are acted upon and
        identifying any observables that share wires.

        Sets:
            _obs_sharing_wires (list[~.Operator]): Observables that share wires with
                any other observable
            _obs_sharing_wires_id (list[int]): Indices of the measurements that contain
                the observables in _obs_sharing_wires
        """
        obs_wires = [wire for m in self.measurements for wire in m.wires if m.obs is not None]
        self._obs_sharing_wires = []
        self._obs_sharing_wires_id = []

        if len(obs_wires) != len(set(obs_wires)):
            c = Counter(obs_wires)
            repeated_wires = {w for w in obs_wires if c[w] > 1}

            for i, m in enumerate(self.measurements):
                if m.obs is not None and len(set(m.wires) & repeated_wires) > 0:
                    self._obs_sharing_wires.append(m.obs)
                    self._obs_sharing_wires_id.append(i)

    def _update_batch_size(self):
        """Infer the batch_size of the quantum script from the batch sizes of its operations
        and check the latter for consistency.

        Sets:
            _batch_size (int): The common batch size of the quantum script operations, if any has one
        """
        candidate = None
        for op in self.operations:
            op_batch_size = getattr(op, "batch_size", None)
            if op_batch_size is None:
                continue
            if candidate:
                if op_batch_size != candidate:
                    raise ValueError(
                        "The batch sizes of the quantum script operations do not match, they include "
                        f"{candidate} and {op_batch_size}."
                    )
            else:
                candidate = op_batch_size

        self._batch_size = candidate

    # ========================================================
    # Parameter handling
    # ========================================================

    @property
    def data(self) -> list[TensorLike]:
        """Alias to :meth:`~.get_parameters` for backwards compatibilities with operations."""
        return self.get_parameters(trainable_only=False)

    @property
    def trainable_params(self) -> list[int]:
        r"""Store or return a list containing the indices of parameters that support
        differentiability. The indices provided match the order of appearance in the
        quantum circuit.

        Setting this property can help reduce the number of quantum evaluations needed
        to compute the Jacobian; parameters not marked as trainable will be
        automatically excluded from the Jacobian computation.

        The number of trainable parameters changes the default output size of method :meth:`~.get_parameters()`.

        .. note::

            For devices that support native backpropagation (such as
            ``default.qubit`` and ``default.mixed``), this
            property contains no relevant information when using
            backpropagation to compute gradients.

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.trainable_params
        [0, 1, 2]
        >>> qscript.trainable_params = [0] # set only the first parameter as trainable
        >>> qscript.get_parameters()
        [0.432]
        """
        if self._trainable_params is None:
            self._trainable_params = list(range(len(self.par_info)))
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices: list[int] | set[int]):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (list[int]): parameter indices
        """
        if any(not isinstance(i, int) or i < 0 for i in param_indices):
            raise ValueError("Argument indices must be non-negative integers.")

        num_params = len(self.par_info)
        if any(i > num_params for i in param_indices):
            raise ValueError(f"Quantum Script only has {num_params} parameters.")

        self._trainable_params = sorted(set(param_indices))

    def get_operation(self, idx: int) -> tuple[Operator, int, int]:
        """Returns the trainable operation, the operation index and the corresponding operation argument
        index, for a specified trainable parameter index.

        Args:
            idx (int): the trainable parameter index

        Returns:
            tuple[.Operation, int, int]: tuple containing the corresponding
            operation, operation index and an integer representing the argument index,
            for the provided trainable parameter.
        """
        # get the index of the parameter in the script
        t_idx = self.trainable_params[idx]

        # get the info for the parameter
        info = self.par_info[t_idx]
        return info["op"], info["op_idx"], info["p_idx"]

    def get_parameters(
        self,
        trainable_only: bool = True,
        operations_only: bool = False,
        **kwargs,  # pylint:disable=unused-argument
    ) -> list[TensorLike]:
        """Return the parameters incident on the quantum script operations.

        The returned parameters are provided in order of appearance
        on the quantum script.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            operations_only (bool): if True, returns only the parameters of the
                operations excluding parameters to observables of measurements

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

        By default, all parameters are trainable and will be returned:

        >>> qscript.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> qscript.trainable_params = [1] # set the second parameter as trainable
        >>> qscript.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> qscript.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        if trainable_only:
            params = []
            for p_idx in self.trainable_params:
                par_info = self.par_info[p_idx]
                if operations_only and isinstance(self[par_info["op_idx"]], MeasurementProcess):
                    continue

                op = par_info["op"]
                op_idx = par_info["p_idx"]
                params.append(op.data[op_idx])
            return params

        # If trainable_only=False, return all parameters
        # This is faster than the above and should be used when indexing `par_info` is not needed
        params = [d for op in self.operations for d in op.data]
        if operations_only:
            return params
        for m in self.measurements:
            if m.obs is not None:
                params.extend(m.obs.data)
        return params

    def bind_new_parameters(
        self, params: Sequence[TensorLike], indices: Sequence[int]
    ) -> "QuantumScript":
        """Create a new tape with updated parameters.

        This function takes a list of new parameters as input, and returns
        a new :class:`~.tape.QuantumScript` containing the new parameters at the provided indices,
        with the parameters at all other indices remaining the same.

        Args:
            params (Sequence[TensorLike]): New parameters to create the tape with. This
                must have the same length as ``indices``.
            indices (Sequence[int]): The parameter indices to update with the given parameters.
                The index of a parameter is defined as its index in ``tape.get_parameters()``.

        Returns:
            .tape.QuantumScript: New tape with updated parameters

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

        A new tape can be created by passing new parameters along with the indices
        to be updated. To modify all parameters in the above qscript:

        >>> new_qscript = qscript.bind_new_parameters([0.1, 0.2, 0.3], [0, 1, 2])
        >>> new_qscript.get_parameters()
        [0.1, 0.2, 0.3]

        The original ``qscript`` remains unchanged:

        >>> qscript.get_parameters()
        [0.432, 0.543, 0.133]

        A subset of parameters can be modified as well, defined by the parameter indices:

        >>> newer_qscript = new_qscript.bind_new_parameters([-0.1, 0.5], [0, 2])
        >>> newer_qscript.get_parameters()
        [-0.1, 0.2, 0.5]
        """

        if len(params) != len(indices):
            raise ValueError("Number of provided parameters does not match number of indices")

        # determine the ops that need to be updated
        op_indices = {}
        for param_idx, idx in enumerate(sorted(indices)):
            pinfo = self.par_info[idx]
            op_idx, p_idx = pinfo["op_idx"], pinfo["p_idx"]

            if op_idx not in op_indices:
                op_indices[op_idx] = {}

            op_indices[op_idx][p_idx] = param_idx

        new_ops = self.circuit

        for op_idx, p_indices in op_indices.items():
            op = new_ops[op_idx]
            data = op.data if isinstance(op, Operator) else op.obs.data

            new_params = [params[p_indices[i]] if i in p_indices else d for i, d in enumerate(data)]

            if isinstance(op, Operator):
                new_op = qml.ops.functions.bind_new_parameters(op, new_params)
            else:
                new_obs = qml.ops.functions.bind_new_parameters(op.obs, new_params)
                new_op = op.__class__(obs=new_obs)

            new_ops[op_idx] = new_op

        new_operations = new_ops[: len(self.operations)]
        new_measurements = new_ops[len(self.operations) :]

        return self.__class__(
            new_operations,
            new_measurements,
            shots=self.shots,
            trainable_params=self.trainable_params,
        )

    # ========================================================
    # MEASUREMENT SHAPE
    #
    # We can extract the private static methods to a new class later
    # ========================================================

    def shape(
        self, device: Union["qml.devices.Device", "qml.devices.LegacyDevice"]
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        """Produces the output shape of the quantum script by inspecting its measurements
        and the device used for execution.

        .. note::

            The computed shape is not stored because the output shape may be
            dependent on the device used for execution.

        Args:
            device (pennylane.devices.Device): the device that will be used for the script execution

        Returns:
            Union[tuple[int], tuple[tuple[int]]]: the output shape(s) of the quantum script result

        .. warning::

            The ``QuantumScript.shape`` method is deprecated and will be removed in v0.44.
            Instead, please use ``MeasurementProcess.shape``.

        **Examples**

        >>> dev = qml.device('default.qubit', wires=2)
        >>> qs = QuantumScript(measurements=[qml.state()])
        >>> qs.shape(dev)
        (4,)
        >>> m = [qml.state(), qml.expval(qml.Z(0)), qml.probs((0,1))]
        >>> qs = QuantumScript(measurements=m)
        >>> qs.shape(dev)
        ((4,), (), (4,))

        """
        warnings.warn(
            "``QuantumScript.shape`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``MeasurementProcess.shape``.",
            PennyLaneDeprecationWarning,
        )

        num_device_wires = len(device.wires) if device.wires else len(self.wires)

        def get_shape(mp, _shots):
            # depends on num_device_wires and self.batch_size from closure
            standard_shape = mp.shape(shots=_shots, num_device_wires=num_device_wires)
            if self.batch_size:
                return (self.batch_size, *standard_shape)
            return standard_shape

        shape = []
        for s in self.shots if self.shots else [None]:
            shots_shape = tuple(get_shape(mp, s) for mp in self.measurements)
            shots_shape = shots_shape[0] if len(shots_shape) == 1 else tuple(shots_shape)
            shape.append(shots_shape)

        return tuple(shape) if self.shots.has_partitioned_shots else shape[0]

    @property
    def numeric_type(self) -> type | tuple[type, ...]:
        """Returns the expected numeric type of the quantum script result by inspecting
        its measurements.

        Returns:
            Union[type, Tuple[type]]: The numeric type corresponding to the result type of the
            quantum script, or a tuple of such types if the script contains multiple measurements

        .. warning::

            ``QuantumScript.numeric_type`` is deprecated and will be removed in v0.44.
            Instead, please use ``MeasurementProcess.numeric_type``.

        **Example:**

        >>> dev = qml.device('default.qubit', wires=2)
        >>> qs = QuantumScript(measurements=[qml.state()])
        >>> qs.numeric_type
         <class 'complex'>
        """
        warnings.warn(
            "``QuantumScript.numeric_type`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``MeasurementProcess.numeric_type``.",
            PennyLaneDeprecationWarning,
        )

        types = tuple(observable.numeric_type for observable in self.measurements)

        return types[0] if len(types) == 1 else types

    # ========================================================
    # Transforms: QuantumScript to QuantumScript
    # ========================================================

    def copy(self, copy_operations: bool = False, **update) -> "QuantumScript":
        """Returns a copy of the quantum script. If any attributes are defined via keyword argument,
        those are used on the new tape - otherwise, all attributes match the original tape. The copy
        is a shallow copy if `copy_operations` is False and no tape attributes are updated via keyword
        argument.

        Args:
            copy_operations (bool): If True, the operations are also shallow copied.
                Otherwise, if False, the copied operations will simply be references
                to the original operations; changing the parameters of one script will likewise
                change the parameters of all copies. If any keyword arguments are passed to update,
                this argument will be treated as True.

        Keyword Args:
            operations (Iterable[Operator]): An iterable of the operations to be performed. If provided, these
                operations will replace the copied operations on the new tape.
            measurements (Iterable[MeasurementProcess]): All the measurements to be performed. If provided, these
                measurements will replace the copied measurements on the new tape.
            shots (None, int, Sequence[int], ~.Shots): Number and/or batches of shots for execution. If provided, these
                shots will replace the copied shots on the new tape.
            trainable_params (None, Sequence[int]): The indices for which parameters are trainable. If provided, these
                parameter indices will replace the copied parameter indices on the new tape.

        Returns:
            QuantumScript : A copy of the quantum script, with modified attributes if specified by keyword argument.

        **Example**

        .. code-block:: python

            tape = qml.tape.QuantumScript(
                ops= [qml.X(0), qml.Y(1)],
                measurements=[qml.expval(qml.Z(0))],
                shots=2000)

            new_tape = tape.copy(measurements=[qml.expval(qml.X(1))])

        >>> tape.measurements
        [expval(Z(0))]

        >>> new_tape.measurements
        [expval(X(1))]

        >>> new_tape.shots
        Shots(total_shots=2000, shot_vector=(ShotCopies(2000 shots x 1),))
        """

        if update:
            if "ops" in update:
                update["operations"] = update["ops"]
            for k in update:
                if k not in ["ops", "operations", "measurements", "shots", "trainable_params"]:
                    raise TypeError(
                        f"{self.__class__}.copy() got an unexpected key '{k}' in update dict"
                    )

        if copy_operations or update:
            # Perform a shallow copy of all operations in the operation and measurement
            # queues. The operations will continue to share data with the original script operations
            # unless modified.
            _ops = update.get("operations")
            _measurements = update.get("measurements")
            if _ops is None:
                _ops = (copy.copy(op) for op in self.operations)
            if _measurements is None:
                _measurements = (copy.copy(mp) for mp in self.measurements)
        else:
            # Perform a shallow copy of the operation and measurement queues. The
            # operations within the queues will be references to the original script operations;
            # changing the original operations will always alter the operations on the copied script.

            _ops = self.operations.copy()

            _measurements = self.measurements.copy()

        update_trainable_params = "operations" in update or "measurements" in update
        # passing trainable_params=None will re-calculate trainable_params
        default_trainable_params = None if update_trainable_params else self._trainable_params

        new_qscript = self.__class__(
            ops=_ops,
            measurements=_measurements,
            shots=update.get("shots", self.shots),
            trainable_params=update.get("trainable_params", default_trainable_params),
        )

        # copy cached properties when relevant
        new_qscript._graph = None if copy_operations or update else self._graph
        if not update.get("operations"):
            # batch size may change if operations were updated
            new_qscript._batch_size = self._batch_size
        if not update.get("measurements"):
            # obs may change if measurements were updated
            new_qscript._obs_sharing_wires = self._obs_sharing_wires
            new_qscript._obs_sharing_wires_id = self._obs_sharing_wires_id
        return new_qscript

    def __copy__(self) -> "QuantumScript":
        return self.copy(copy_operations=True)

    def expand(
        self,
        depth: int = 1,
        stop_at: Callable[[Operation | MeasurementProcess], bool] | None = None,
        expand_measurements: bool = False,
    ) -> "QuantumScript":
        """Expand all operations to a specific depth.

        Args:
            depth (int): the depth the script should be expanded
            stop_at (Callable): A function which accepts a queue object,
                and returns ``True`` if this object should *not* be expanded.
                If not provided, all objects that support expansion will be expanded.
            expand_measurements (bool): If ``True``, measurements will be expanded
                to basis rotations and computational basis measurements.

        .. seealso:: :func:`~.pennylane.devices.preprocess.decompose` for a transform that
           performs the same job and fits into the current transform architecture.

        .. warning::

            This method cannot be used with a tape with non-commuting measurements, even if
            ``expand_measurements=False``.

            >>> mps = [qml.expval(qml.X(0)), qml.expval(qml.Y(0))]
            >>> tape = qml.tape.QuantumScript([], mps)
            >>> tape.expand()
            Traceback (most recent call last):
                ...
            pennylane.exceptions.QuantumFunctionError: Only observables that are qubit-wise commuting Pauli words can be returned on the same wire, some of the following measurements do not commute:
            [expval(X(0)), expval(Y(0))]

            Since commutation is determined by pauli word arithmetic, non-pauli words cannot share
            wires with other measurements, even if they commute:

            >>> measurements = [qml.expval(qml.Projector([0], 0)), qml.probs(wires=0)]
            >>> tape = qml.tape.QuantumScript([], measurements)
            >>> tape.expand()
            Traceback (most recent call last):
                ...
            pennylane.exceptions.QuantumFunctionError: Only observables that are qubit-wise commuting Pauli words can be returned on the same wire, some of the following measurements do not commute:
            [expval(Projector(array([0]), wires=[0])), probs(wires=[0])]

            For this reason, we recommend the use of :func:`~.pennylane.devices.preprocess.decompose` instead.

        .. details::
            :title: Usage Details

            >>> ops = [qml.Permute((2,1,0), wires=(0,1,2)), qml.X(0)]
            >>> measurements = [qml.expval(qml.X(0))]
            >>> tape = qml.tape.QuantumScript(ops, measurements)
            >>> expanded_tape = tape.expand()
            >>> print(expanded_tape.draw())
            0: ─╭SWAP──RX─╭GlobalPhase─┤  <X>
            2: ─╰SWAP─────╰GlobalPhase─┤

            Specifying a depth greater than one decomposes operations multiple times.

            >>> expanded_tape2 = tape.expand(depth=2)
            >>> print(expanded_tape2.draw())
            0: ─╭●─╭X─╭●──RX─┤  <X>
            2: ─╰X─╰●─╰X─────┤

            The ``stop_at`` callable allows the specification of terminal
            operations that should no longer be decomposed. In this example, the ``X``
            operator is not decomposed because ``stop_at(qml.X(0)) == True``.

            >>> def stop_at(obj):
            ...     return isinstance(obj, qml.X)
            >>> expanded_tape = tape.expand(stop_at=stop_at)
            >>> print(expanded_tape.draw())
            0: ─╭SWAP──X─┤  <X>
            2: ─╰SWAP────┤

            .. warning::

                If an operator does not have a decomposition, it will not be decomposed, even if
                ``stop_at(obj) == False``.  If you want to decompose to reach a certain gateset,
                you will need an extra validation pass to ensure you have reached the gateset.

                >>> def stop_at(obj):
                ...     return getattr(obj, "name", "") in {"RX", "RY"}
                >>> tape = qml.tape.QuantumScript([qml.RZ(0.1, 0)])
                >>> tape.expand(stop_at=stop_at).circuit
                [RZ(0.1, wires=[0])]

            If more than one observable exists on a wire, the diagonalizing gates will be applied
            and the observable will be substituted for an analogous combination of ``qml.Z`` operators.
            This will happen even if ``expand_measurements=False``.

            >>> mps = [qml.expval(qml.X(0)), qml.expval(qml.X(0) @ qml.X(1))]
            >>> tape = qml.tape.QuantumScript([], mps)
            >>> expanded_tape = tape.expand()
            >>> print(expanded_tape.draw())
            0: ──RY─┤  <Z> ╭<Z@Z>
            1: ──RY─┤      ╰<Z@Z>

            Setting ``expand_measurements=True`` applies any diagonalizing gates and converts
            the measurement into a wires+eigvals representation.

            .. warning::
                Many components of PennyLane do not support the wires + eigvals representation.
                Setting ``expand_measurements=True`` should be used with extreme caution.

            >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))])
            >>> tape.expand(expand_measurements=True).circuit
            [H(0), expval(eigvals=[ 1. -1.], wires=[0])]

        """
        return qml.tape.expand_tape(
            self, depth=depth, stop_at=stop_at, expand_measurements=expand_measurements
        )

    def adjoint(self) -> "QuantumScript":
        """Create a quantum script that is the adjoint of this one.

        Adjointed quantum scripts are the conjugated and transposed version of the
        original script. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        Returns:
            ~.QuantumScript: the adjointed script
        """
        ops = self.operations[self.num_preps :]
        prep = self.operations[: self.num_preps]
        with qml.QueuingManager.stop_recording():
            ops_adj = [qml.adjoint(op, lazy=False) for op in reversed(ops)]
        return self.__class__(ops=prep + ops_adj, measurements=self.measurements, shots=self.shots)

    # ========================================================
    # Transforms: QuantumScript to Information
    # ========================================================

    @property
    def graph(self) -> "qml.CircuitGraph":
        """Returns a directed acyclic graph representation of the recorded
        quantum circuit:

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.graph
        <pennylane.circuit_graph.CircuitGraph object at 0x...>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .CircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = qml.CircuitGraph(
                self.operations,
                self.measurements,
                self.wires,
                self.par_info,
                self.trainable_params,
            )

        return self._graph

    @property
    def specs(self) -> "qml.resource.resource.SpecsDict[str, Any]":
        """Resource information about a quantum circuit.

        Returns:
            SpecsDict[str, Any]: A dictionary containing the specifications of the quantum script.

        **Example**
         >>> ops = [qml.Hadamard(0), qml.RX(0.26, 1), qml.CNOT((1,0)),
         ...         qml.Rot(1.8, -2.7, 0.2, 0), qml.Hadamard(1), qml.CNOT((0, 1))]
         >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0) @ qml.Z(1))])

        Asking for the specs produces a dictionary of useful information about the circuit:

        >>> qscript.specs['num_observables']
        1
        >>> print(qscript.specs['resources'])
        num_wires: 2
        num_gates: 6
        depth: 4
        shots: Shots(total=None)
        gate_types:
        {'Hadamard': 2, 'RX': 1, 'CNOT': 2, 'Rot': 1}
        gate_sizes:
        {1: 4, 2: 2}
        """
        if self._specs is None:
            self._specs = qml.resource.resource.specs_from_tape(self)
        return self._specs

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def draw(
        self,
        wire_order: Iterable[Hashable] | None = None,
        show_all_wires: bool = False,
        decimals: int | None = None,
        max_length: int = 100,
        show_matrices: bool = True,
    ) -> str:
        """Draw the quantum script as a circuit diagram. See :func:`~.drawer.tape_text` for more information.

        Args:
            wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
            show_all_wires (bool): If True, all wires, including empty wires, are printed.
            decimals (int): How many decimal points to include when formatting operation parameters.
                Default ``None`` will omit parameters from operation labels.
            max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
                begin anew beneath the previous lines.
            show_matrices=True (bool): show matrix valued parameters below all circuit diagrams

        Returns:
            str: the circuit representation of the quantum script
        """
        return qml.drawer.tape_text(
            self,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=show_matrices,
        )

    def to_openqasm(
        self,
        wires: WiresLike | None = None,
        rotations: bool = True,
        measure_all: bool = True,
        precision: int | None = None,
    ) -> str:
        """Serialize the circuit as an OpenQASM 2.0 program.

        Measurements are assumed to be performed on all qubits in the computational basis. An
        optional ``rotations`` argument can be provided so that output of the OpenQASM circuit is
        diagonal in the eigenbasis of the quantum script's observables. The measurement outputs can be
        restricted to only those specified in the script by setting ``measure_all=False``.

        .. note::

            The serialized OpenQASM program assumes that gate definitions
            in ``qelib1.inc`` are available.

        .. warning::

            The ``QuantumScript.to_openqasm`` method is deprecated and will be removed in v0.44.
            Instead, please use the :func:`~.to_openqasm` function.

        Args:
            wires (Wires or None): the wires to use when serializing the circuit
            rotations (bool): in addition to serializing user-specified
                operations, also include the gates that diagonalize the
                measured wires such that they are in the eigenbasis of the circuit observables.
            measure_all (bool): whether to perform a computational basis measurement on all qubits
                or just those specified in the script
            precision (int): decimal digits to display for parameters

        Returns:
            str: OpenQASM serialization of the circuit
        """
        warnings.warn(
            "``QuantumScript.to_openqasm`` is deprecated and will be removed in v0.44. "
            "Instead, please use ``qml.to_openqasm``.",
            PennyLaneDeprecationWarning,
        )

        return qml.to_openqasm(
            self,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
        )

    @classmethod
    def from_queue(
        cls: type[QS], queue: qml.queuing.AnnotatedQueue, shots: ShotsLike | None = None
    ) -> QS:
        """Construct a QuantumScript from an AnnotatedQueue."""
        return cls(*process_queue(queue), shots=shots)

    def map_to_standard_wires(self) -> "QuantumScript":
        """
        Map a circuit's wires such that they are in a standard order. If no
        mapping is required, the unmodified circuit is returned.

        Returns:
            QuantumScript: The circuit with wires in the standard order

        The standard order is defined by the operator wires being increasing
        integers starting at zero, to match array indices. If there are any
        measurement wires that are not in any operations, those will be mapped
        to higher values. If there are any work wires that are not used in
        any operations or measurements, those will be mapped to higher values.

        **Example:**

        >>> circuit = qml.tape.QuantumScript([qml.X("a")], [qml.expval(qml.Z("b"))])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), expval(Z(1))]

        If any measured wires are not in any operations, they will be mapped last:

        >>> circuit = qml.tape.QuantumScript([qml.X(1)], [qml.probs(wires=[0, 1])])
        >>> circuit.map_to_standard_wires().circuit
        [X(0), probs(wires=[1, 0])]

        If no wire-mapping is needed, then the returned circuit *is* the inputted circuit:

        >>> circuit = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.Z(1))])
        >>> circuit.map_to_standard_wires() is circuit
        True

        Work wires that are not used in operations or measurements are mapped to the last
        positions in the circuit:

        >>> mcx = qml.MultiControlledX([1, 4, "b", 2], work_wires=[0, 6])
        >>> circuit = qml.tape.QuantumScript([mcx], [qml.probs(wires=[2, 3, 6])])
        >>> mapped_circuit = circuit.map_to_standard_wires()
        >>> mapped_circuit.circuit
        [MultiControlledX(wires=[0, 1, 2, 3], control_values=[True, True, True]),
         probs(wires=[3, 4, 5])]
        >>> mapped_circuit[0].work_wires
        Wires([6, 5])
        """
        wire_map = self._get_standard_wire_map()
        if wire_map is None:
            return self
        tapes, fn = qml.map_wires(self, wire_map)
        return fn(tapes)

    def _get_standard_wire_map(self) -> dict:
        """Helper function to produce the wire map for map_to_standard_wires."""
        op_wires = Wires.all_wires(op.wires for op in self.operations)
        work_wires = Wires.all_wires(getattr(op, "work_wires", []) for op in self.operations)
        meas_wires = Wires.all_wires(mp.wires for mp in self.measurements)
        num_op_wires = len(op_wires)
        meas_only_wires = set(meas_wires) - set(op_wires)
        num_op_meas_wires = num_op_wires + len(meas_only_wires)
        work_only_wires = set(work_wires) - set(op_wires) - meas_only_wires
        # If the op wires are consecutive integers, followed by measurement-only wires, followed
        # by work-only wires, we do not perform a mapping, signaled by returning `None`.
        if (
            set(op_wires) == set(range(num_op_wires))
            and meas_only_wires == set(range(num_op_wires, num_op_wires + len(meas_only_wires)))
            and work_only_wires
            == set(range(num_op_meas_wires, num_op_meas_wires + len(work_only_wires)))
        ):
            return None

        wire_map = {w: i for i, w in enumerate(op_wires + meas_only_wires + work_only_wires)}
        return wire_map


# ParamSpec is used to preserve the exact signature of the input function `fn`
P = ParamSpec("P")
T = TypeVar("T")


def make_qscript(fn: Callable[P, T], shots: ShotsLike | None = None) -> Callable[P, QS]:
    """Returns a function that generates a qscript from a quantum function without any
    operation queuing taking place.

    This is useful when you would like to manipulate or transform
    the qscript created by a quantum function without evaluating it.

    Args:
        fn (function): the quantum function to generate the qscript from
        shots (None, int, Sequence[int], ~.Shots): number and/or
            batches of executions

    Returns:
        function: The returned function takes the same arguments as the quantum
        function. When called, it returns the generated quantum script
        without any queueing occurring.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)

    We can use ``make_qscript`` to extract the qscript generated by this
    quantum function, without any of the operations being queued by
    any existing queuing contexts:

    >>> with qml.queuing.AnnotatedQueue() as active_queue:
    ...     _ = qml.RY(1.0, wires=0)
    ...     qs = make_qscript(qfunc)(0.5)
    >>> qs.operations
    [H(0), CNOT(wires=[0, 1]), RX(0.5, wires=[0])]

    Note that the currently recording queue did not queue any of these quantum operations:

    >>> active_queue.queue
    [RY(1.0, wires=[0])]
    """

    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            fn(*args, **kwargs)

        return QuantumScript.from_queue(q, shots)

    return wrapper


QuantumScriptBatch = Sequence[QuantumScript]
QuantumScriptOrBatch = QuantumScript | QuantumScriptBatch

register_pytree(QuantumScript, QuantumScript._flatten, QuantumScript._unflatten)


def _get_base_obs(observables):

    overlapping_ops_observables = []

    while any(isinstance(o, (qml.ops.CompositeOp, qml.ops.SymbolicOp)) for o in observables):

        new_obs = []

        for observable in observables:

            if isinstance(observable, qml.ops.CompositeOp):
                if any(len(o) > 1 for o in observable.overlapping_ops):
                    overlapping_ops_observables.append(observable)
                else:
                    new_obs.extend(observable.operands)
            elif isinstance(observable, qml.ops.SymbolicOp):
                new_obs.append(observable.base)
            else:
                new_obs.append(observable)

        observables = new_obs

    # removes duplicates from list without disrupting order - basically an ordered set
    return list(dict.fromkeys(observables + overlapping_ops_observables))
