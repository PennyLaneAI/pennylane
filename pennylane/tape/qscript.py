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
from collections import Counter, defaultdict, namedtuple
from collections.abc import Sequence
from typing import List

import numpy as np

import pennylane as qml
from pennylane.measurements import AllCounts, Counts, Sample, Shadow, ShadowExpval
from pennylane.operation import Operator

_empty_wires = qml.wires.Wires([])


OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "S.inv": "sdg",
    "T": "t",
    "T.inv": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
}
"""
dict[str, str]: Maps PennyLane gate names to equivalent QASM gate names.

Note that QASM has two native gates:

- ``U`` (equivalent to :class:`~.U3`)
- ``CX`` (equivalent to :class:`~.CNOT`)

All other gates are defined in the file stdgates.inc:
https://github.com/Qiskit/openqasm/blob/master/examples/stdgates.inc
"""

ShotTuple = namedtuple("ShotTuple", ["shots", "copies"])
"""tuple[int, int]: Represents copies of a shot number."""


def _process_shot_sequence(shot_list):
    """Process the shot sequence, to determine the total
    number of shots and the shot vector.

    Args:
        shot_list (Sequence[int, tuple[int]]): sequence of non-negative shot integers

    Returns:
        tuple[int, list[.ShotTuple[int]]]: A tuple containing the total number
        of shots, as well as a list of shot tuples.

    **Example**

    >>> shot_list = [3, 1, 2, 2, 2, 2, 6, 1, 1, 5, 12, 10, 10]
    >>> _process_shot_sequence(shot_list)
    (57,
     [ShotTuple(shots=3, copies=1),
      ShotTuple(shots=1, copies=1),
      ShotTuple(shots=2, copies=4),
      ShotTuple(shots=6, copies=1),
      ShotTuple(shots=1, copies=2),
      ShotTuple(shots=5, copies=1),
      ShotTuple(shots=12, copies=1),
      ShotTuple(shots=10, copies=2)])

    The total number of shots (57), and a sparse representation of the shot
    sequence is returned, where tuples indicate the number of times a shot
    integer is repeated.
    """
    if all(isinstance(s, int) for s in shot_list):

        if len(set(shot_list)) == 1:
            # All shots are identical, only require a single shot tuple
            shot_vector = [ShotTuple(shots=shot_list[0], copies=len(shot_list))]
        else:
            # Iterate through the shots, and group consecutive identical shots
            split_at_repeated = np.split(shot_list, np.diff(shot_list).nonzero()[0] + 1)
            shot_vector = [ShotTuple(shots=i[0], copies=len(i)) for i in split_at_repeated]

    elif all(isinstance(s, (int, tuple)) for s in shot_list):
        # shot list contains tuples; assume it is already in a sparse representation
        shot_vector = [
            ShotTuple(*i) if isinstance(i, tuple) else ShotTuple(i, 1) for i in shot_list
        ]

    else:
        raise ValueError(f"Unknown shot sequence format {shot_list}")

    total_shots = int(np.sum(np.prod(shot_vector, axis=1)))
    return total_shots, shot_vector


class QuantumScript:
    """The state preparation, operations, and measurements that represent instructions for
    execution on a quantum device.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed
        prep (Iterable[Operator]): Any state preparations to perform at the start of the circuit

    Keyword Args:
        name (str): a name given to the quantum script
        do_queue=False (bool): Whether or not to queue. Defaults to ``False`` for ``QuantumScript``.
        _update=True (bool): Whether or not to set various properties on initialization. Setting
            ``_update=False`` reduces computations if the script is only an intermediary step.

    .. seealso:: :class:`pennylane.tape.QuantumTape`

    **Example:**

    .. code-block:: python

        from pennylane.tape import QuantumScript

        prep = [qml.BasisState(np.array([1,1]), wires=(0,"a"))]

        ops = [qml.RX(0.432, 0),
               qml.RY(0.543, 0),
               qml.CNOT((0,"a")),
               qml.RX(0.133, "a")]

        qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))], prep)

    >>> list(qscript)
    [BasisState(array([1, 1]), wires=[0, "a"]),
    RX(0.432, wires=[0]),
    RY(0.543, wires=[0]),
    CNOT(wires=[0, 'a']),
    RX(0.133, wires=['a']),
    expval(PauliZ(wires=[0]))]
    >>> qscript.operations
    [BasisState(array([1, 1]), wires=[0, "a"]),
    RX(0.432, wires=[0]),
    RY(0.543, wires=[0]),
    CNOT(wires=[0, 'a']),
    RX(0.133, wires=['a'])]
    >>> qscript.measurements
    [expval(PauliZ(wires=[0]))]

    Iterating over the quantum script can be done by:

    >>> for op in qscript:
    ...     print(op)
    BasisState(array([1, 1]), wires=[0, "a"])
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(PauliZ(wires=[0]))'

    Quantum scripts also support indexing and length determination:

    >>> qscript[0]
    BasisState(array([1, 1]), wires=[0, "a"])
    >>> len(qscript)
    6

    Once constructed, the script can be executed directly on a quantum device
    using the :func:`~.pennylane.execute` function:

    >>> dev = qml.device('default.qubit', wires=(0,'a'))
    >>> qml.execute([qscript], dev, gradient_fn=None)
    [array([0.77750694])]

    ``ops``, ``measurements``, and ``prep`` are converted to lists upon initialization,
    so those arguments accept any iterable object:

    >>> qscript = QuantumScript((qml.PauliX(i) for i in range(3)))
    >>> qscript.circuit
    [PauliX(wires=[0]), PauliX(wires=[1]), PauliX(wires=[2])]

    """

    do_queue = False
    """Whether or not to queue the object. Assumed ``False`` for a vanilla Quantum Script, but may be
    True for its child Quantum Tape."""

    def __init__(
        self, ops=None, measurements=None, prep=None, shots=None, name=None, _update=True
    ):  # pylint: disable=too-many-arguments
        self.name = name
        self.shots = shots
        self._prep = [] if prep is None else list(prep)
        self._ops = [] if ops is None else list(ops)
        self._measurements = [] if measurements is None else list(measurements)

        self._par_info = {}
        """dict[int, dict[str, Operator or int]]: Parameter information. Keys are
        parameter indices (in the order they appear on the quantum script), and values are a
        dictionary containing the corresponding operation and operation parameter index."""

        self._trainable_params = []
        self._graph = None
        self._specs = None
        self._output_dim = 0
        self._batch_size = None
        self._qfunc_output = None

        self.wires = _empty_wires
        self.num_wires = 0

        self.is_sampled = False
        self.all_sampled = False

        self._obs_sharing_wires = []
        """list[.Observable]: subset of the observables that share wires with another observable,
        i.e., that do not have their own unique set of wires."""
        self._obs_sharing_wires_id = []

        if _update:
            self._update()

    def __repr__(self):
        return f"<{self.__class__.__name__}: wires={self.wires.tolist()}, params={self.num_params}>"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the quantum script"""
        fingerprint = []
        fingerprint.extend(op.hash for op in self.operations)
        fingerprint.extend(m.hash for m in self.measurements)
        fingerprint.extend(self.trainable_params)
        return hash(tuple(fingerprint))

    def __iter__(self):
        """list[.Operator, .MeasurementProcess]: Return an iterator to the
        underlying quantum circuit object."""
        return iter(self.circuit)

    def __getitem__(self, idx):
        """list[.Operator]: Return the indexed operator from underlying quantum
        circuit object."""
        return self.circuit[idx]

    def __len__(self):
        """int: Return the number of operations and measurements in the
        underlying quantum circuit object."""
        return len(self.circuit)

    # ========================================================
    # QSCRIPT properties
    # ========================================================

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum script (if any)"""
        return None

    @property
    def circuit(self):
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
    def operations(self) -> List[Operator]:
        """Returns the state preparations and operations on the quantum script.

        Returns:
            list[.Operator]: quantum operations

        >>> ops = [qml.QubitStateVector([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        >>> qscript.operations
        [QubitStateVector([0, 1], wires=[0]), RX(0.432, wires=[0])]
        """
        return self._prep + self._ops

    @property
    def observables(self):
        """Returns the observables on the quantum script.

        Returns:
            list[.Operator]]: list of observables

        **Example**

        >>> ops = [qml.QubitStateVector([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        >>> qscript.observables
        [expval(PauliZ(wires=[0]))]
        """
        # TODO: modify this property once devices
        # have been refactored to accept and understand recieving
        # measurement processes rather than specific observables.
        obs = []

        for m in self._measurements:
            if m.obs is not None:
                m.obs.return_type = m.return_type
                obs.append(m.obs)
            else:
                obs.append(m)

        return obs

    @property
    def measurements(self):
        """Returns the measurements on the quantum script.

        Returns:
            list[.MeasurementProcess]: list of measurement processes

        **Example**

        >>> ops = [qml.QubitStateVector([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        >>> qscript.measurements
        [expval(PauliZ(wires=[0]))]
        """
        return self._measurements

    @property
    def num_params(self):
        """Returns the number of trainable parameters on the quantum script."""
        return len(self.trainable_params)

    @property
    def batch_size(self):
        r"""The batch size of the quantum script inferred from the batch sizes
        of the used operations for parameter broadcasting.

        .. seealso:: :attr:`~.Operator.batch_size` for details.

        Returns:
            int or None: The batch size of the quantum script if present, else ``None``.
        """
        return self._batch_size

    @property
    def output_dim(self):
        """The (inferred) output dimension of the quantum script."""
        return self._output_dim

    @property
    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[~.Operation]: the operations that diagonalize the observables
        """
        rotation_gates = []

        for observable in self.observables:
            # some observables do not have diagonalizing gates,
            # in which case we just don't append any
            with contextlib.suppress(qml.operation.DiagGatesUndefinedError):
                rotation_gates.extend(observable.diagonalizing_gates())
        return rotation_gates

    ##### Update METHODS ###############

    def _update(self):
        """Update all internal metadata regarding processed operations and observables"""
        self._graph = None
        self._specs = None
        self._update_circuit_info()  # Updates wires, num_wires, is_sampled, all_sampled; O(ops+obs)
        self._update_par_info()  # Updates the _par_info dictionary; O(ops+obs)

        # The following line requires _par_info to be up to date
        self._update_trainable_params()  # Updates the _trainable_params; O(1)

        self._update_observables()  # Updates _obs_sharing_wires and _obs_sharing_wires_id
        self._update_batch_size()  # Updates _batch_size; O(ops)

        # The following line requires _batch_size to be up to date
        self._update_output_dim()  # Updates _output_dim; O(obs)

    def _update_circuit_info(self):
        """Update circuit metadata

        Sets:
            wires (~.Wires): Wires
            num_wires (int): Number of wires
            is_sampled (bool): Whether any measurement is of type ``Sample`` or ``Counts``
            all_sampled (bool): Whether all measurements are of type ``Sample`` or ``Counts``
        """
        self.wires = qml.wires.Wires.all_wires(dict.fromkeys(op.wires for op in self))
        self.num_wires = len(self.wires)

        is_sample_type = [
            m.return_type in (Sample, Counts, AllCounts, Shadow, ShadowExpval)
            for m in self.measurements
        ]
        self.is_sampled = any(is_sample_type)
        self.all_sampled = all(is_sample_type)

    def _update_par_info(self):
        """Update the parameter information dictionary.

        Sets:
            _par_info (dict): Parameter information dictionary
        """
        param_count = 0

        for obj in self.operations + self.observables:

            for p in range(len(obj.data)):
                info = self._par_info.get(param_count, {})
                info.update({"op": obj, "p_idx": p})

                self._par_info[param_count] = info
                param_count += 1

    def _update_trainable_params(self):
        """Set the trainable parameters

        Sets:
            _trainable_params (list[int]): Script parameter indices of trainable parameters

        self._par_info.keys() is assumed to be sorted and up to date when calling
        this method. This assumes that self._par_info was created in a sorted manner,
        as in _update_par_info.

        Call `_update_par_info` before `_update_trainable_params`
        """
        self._trainable_params = list(self._par_info)

    def _update_observables(self):
        """Update information about observables, including the wires that are acted upon and
        identifying any observables that share wires.

        Sets:
            _obs_sharing_wires (list[~.Observable]): Observables that share wires with
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

    def _update_output_dim(self):
        """Update the dimension of the output of the quantum script.

        Sets:
            self._output_dim (int): Size of the quantum script output (when flattened)

        This method makes use of `self.batch_size`, so that `self._batch_size`
        needs to be up to date when calling it.
        Call `_update_batch_size` before `_update_output_dim`
        """
        self._output_dim = 0
        for m in self.measurements:
            # attempt to infer the output dimension
            if m.return_type is qml.measurements.Probability:
                # TODO: what if we had a CV device here? Having the base as
                # 2 would have to be swapped to the cutoff value
                self._output_dim += 2 ** len(m.wires)
            elif m.return_type is not qml.measurements.State:
                self._output_dim += 1
        if self.batch_size:
            self._output_dim *= self.batch_size

    # ========================================================
    # Parameter handling
    # ========================================================

    @property
    def data(self):
        """Alias to :meth:`~.get_parameters` and :meth:`~.set_parameters`
        for backwards compatibilities with operations."""
        return self.get_parameters(trainable_only=False)

    @data.setter
    def data(self, params):
        self.set_parameters(params, trainable_only=False)

    @property
    def trainable_params(self):
        """Store or return a list containing the indices of parameters that support
        differentiability. The indices provided match the order of appearence in the
        quantum circuit.

        Setting this property can help reduce the number of quantum evaluations needed
        to compute the Jacobian; parameters not marked as trainable will be
        automatically excluded from the Jacobian computation.

        The number of trainable parameters determines the number of parameters passed to
        :meth:`~.set_parameters`, and changes the default output size of method :meth:`~.get_parameters()`.

        .. note::

            For devices that support native backpropagation (such as
            ``default.qubit.tf`` and ``default.qubit.autograd``), this
            property contains no relevant information when using
            backpropagation to compute gradients.

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        >>> qscript.trainable_params
        [0, 1, 2]
        >>> qscript.trainable_params = [0] # set only the first parameter as trainable
        >>> qscript.get_parameters()
        [0.432]
        """
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (list[int]): parameter indices
        """
        if any(not isinstance(i, int) or i < 0 for i in param_indices):
            raise ValueError("Argument indices must be non-negative integers.")

        num_params = len(self._par_info)
        if any(i > num_params for i in param_indices):
            raise ValueError(f"Quantum Script only has {num_params} parameters.")

        self._trainable_params = sorted(set(param_indices))

    def get_operation(self, idx):
        """Returns the trainable operation, and the corresponding operation argument
        index, for a specified trainable parameter index.

        Args:
            idx (int): the trainable parameter index

        Returns:
            tuple[.Operation, int]: tuple containing the corresponding
            operation, and an integer representing the argument index,
            for the provided trainable parameter.
        """
        # get the index of the parameter in the script
        t_idx = self.trainable_params[idx]

        # get the info for the parameter
        info = self._par_info[t_idx]

        # get the corresponding operation
        op = info["op"]

        # get the corresponding operation parameter index
        # (that is, index of the parameter within the operation)
        p_idx = info["p_idx"]
        return op, p_idx

    def get_parameters(
        self, trainable_only=True, operations_only=False, **kwargs
    ):  # pylint:disable=unused-argument
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
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])

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
        params = []
        iterator = self.trainable_params if trainable_only else self._par_info

        for p_idx in iterator:
            op = self._par_info[p_idx]["op"]
            if operations_only and hasattr(op, "return_type"):
                continue

            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])
        return params

    def set_parameters(self, params, trainable_only=True):
        """Set the parameters incident on the quantum script operations.

        Args:
            params (list[float]): A list of real numbers representing the
                parameters of the quantum operations. The parameters should be
                provided in order of appearance in the quantum script.
            trainable_only (bool): if True, set only trainable parameters

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])

        By default, all parameters are trainable and can be modified:

        >>> qscript.set_parameters([0.1, 0.2, 0.3])
        >>> qscript.get_parameters()
        [0.1, 0.2, 0.3]

        Setting the trainable parameter indices will result in only the specified
        parameters being modifiable. Note that this only modifies the number of
        parameters that must be passed.

        >>> qscript.trainable_params = [0, 2] # set the first and third parameter as trainable
        >>> qscript.set_parameters([-0.1, 0.5])
        >>> qscript.get_parameters(trainable_only=False)
        [-0.1, 0.2, 0.5]

        The ``trainable_only`` argument can be set to ``False`` to instead set
        all parameters:

        >>> qscript.set_parameters([4, 1, 6], trainable_only=False)
        >>> qscript.get_parameters(trainable_only=False)
        [4, 1, 6]
        """
        if trainable_only:
            iterator = zip(self.trainable_params, params)
            required_length = self.num_params
        else:
            iterator = enumerate(params)
            required_length = len(self._par_info)

        if len(params) != required_length:
            raise ValueError("Number of provided parameters does not match.")

        for idx, p in iterator:
            op = self._par_info[idx]["op"]
            op.data[self._par_info[idx]["p_idx"]] = p
            op._check_batching(op.data)
        self._update_batch_size()
        self._update_output_dim()

    # ========================================================
    # MEASUREMENT SHAPE
    #
    # We can extract the private static methods to a new class later
    # ========================================================

    @staticmethod
    def _single_measurement_shape(measurement_process, device):
        """Auxiliary function of shape that determines the output
        shape of a quantum script with a single measurement.

        Args:
            measurement_process (MeasurementProcess): the measurement process
                associated with the single measurement
            device (~.Device): a PennyLane device

        Returns:
            tuple: output shape
        """
        return measurement_process.shape(device)

    @staticmethod
    def _multi_homogenous_measurement_shape(mps, device):
        """Auxiliary function of shape that determines the output
        shape of a quantum script with multiple homogenous measurements.

        .. note::

            Assuming multiple probability measurements where not all
            probability measurements have the same number of wires specified,
            the output shape of the quantum script is a sum of the output shapes produced
            by each probability measurement.

            Consider the `qml.probs(wires=[0]), qml.probs(wires=[1,2])`
            multiple probability measurement with an analytic device as an
            example.

            The output shape will be a one element tuple `(6,)`, where the
            element `6` is equal to `2 ** 1 + 2 ** 2 = 6`. The base of each
            term is determined by the number of basis states and the exponent
            of each term comes from the length of the wires specified for the
            probability measurements: `1 == len([0]) and 2 == len([1, 2])`.
        """
        shape = tuple()

        # We know that there's one type of return_type, gather it from the
        # first one
        ret_type = mps[0].return_type
        if ret_type == qml.measurements.State:
            raise ValueError(
                "Getting the output shape of a quantum script with multiple state measurements is not supported."
            )

        shot_vector = device._shot_vector
        if shot_vector is None:
            if ret_type in (qml.measurements.Expectation, qml.measurements.Variance):

                shape = (len(mps),)

            elif ret_type == qml.measurements.Probability:

                wires_num_set = {len(meas.wires) for meas in mps}
                same_num_wires = len(wires_num_set) == 1
                if same_num_wires:
                    # All probability measurements have the same number of
                    # wires, gather the length from the first one

                    len_wires = len(mps[0].wires)
                    dim = mps[0]._get_num_basis_states(len_wires, device)
                    shape = (len(mps), dim)

                else:
                    # There are a varying number of wires that the probability
                    # measurement processes act on
                    shape = (sum(2 ** len(m.wires) for m in mps),)

            elif ret_type == qml.measurements.Sample:

                shape = (len(mps), device.shots)

            # No other measurement type to check

        else:
            shape = QuantumScript._shape_shot_vector_multi_homogenous(mps, device)

        return shape

    @staticmethod
    def _shape_shot_vector_multi_homogenous(mps, device):
        """Auxiliary function for determining the output shape of the quantum script for
        multiple homogenous measurements for a device with a shot vector.

        Note: it is assumed that getting the output shape of a script with
        multiple state measurements is not supported.
        """
        shape = tuple()

        ret_type = mps[0].return_type
        shot_vector = device._shot_vector

        # Shot vector was defined
        if ret_type in (qml.measurements.Expectation, qml.measurements.Variance):
            num = sum(shottup.copies for shottup in shot_vector)
            shape = (num, len(mps))

        elif ret_type == qml.measurements.Probability:

            wires_num_set = {len(meas.wires) for meas in mps}
            same_num_wires = len(wires_num_set) == 1
            if same_num_wires:
                # All probability measurements have the same number of
                # wires, gather the length from the first one

                len_wires = len(mps[0].wires)
                dim = mps[0]._get_num_basis_states(len_wires, device)
                shot_copies_sum = sum(s.copies for s in shot_vector)
                shape = (shot_copies_sum, len(mps), dim)

            else:
                # There is a varying number of wires that the probability
                # measurement processes act on
                # TODO: revisit when issues with this case are resolved
                raise ValueError(
                    "Getting the output shape of a quantum script with multiple probability measurements "
                    "along with a device that defines a shot vector is not supported."
                )

        elif ret_type == qml.measurements.Sample:
            shape = []
            for shot_val in device.shot_vector:
                shots = shot_val.shots
                if shots != 1:
                    shape.extend((shots, len(mps)) for _ in range(shot_val.copies))
                else:
                    shape.extend((len(mps),) for _ in range(shot_val.copies))
        return shape

    def shape(self, device):
        """Produces the output shape of the tape by inspecting its measurements
        and the device used for execution.

        .. note::

            The computed shape is not stored because the output shape may be
            dependent on the device used for execution.

        Args:
            device (.Device): the device that will be used for the tape execution

        Raises:
            ValueError: raised for unsupported cases for
                example when the tape contains heterogeneous measurements

        Returns:
            Union[tuple[int], list[tuple[int]]]: the output shape(s) of the
            tape result

        **Example:**

        .. code-block:: pycon

            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.shape(dev)
            (1, 4)
        """
        if qml.active_return():
            return self._shape_new(device)

        output_shape = tuple()

        if len(self._measurements) == 1:
            output_shape = self._single_measurement_shape(self._measurements[0], device)
        else:
            num_measurements = len({meas.return_type for meas in self._measurements})
            if num_measurements == 1:
                output_shape = self._multi_homogenous_measurement_shape(self._measurements, device)
            else:
                raise ValueError(
                    "Getting the output shape of a tape that contains multiple types of measurements is unsupported."
                )
        return output_shape

    def _shape_new(self, device):
        """Produces the output shape of the tape by inspecting its measurements
        and the device used for execution.

        .. note::

            The computed shape is not stored because the output shape may be
            dependent on the device used for execution.

        Args:
            device (.Device): the device that will be used for the tape execution

        Returns:
            Union[tuple[int], tuple[tuple[int]]]: the output shape(s) of the
            tape result

        **Examples**

        .. code-block:: pycon

            >>> qml.enable_return()
            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.shape(dev)
            (4,)
            >>> m = [qml.state(), qml.expval(qml.PauliZ(0)), qml.probs((0,1))]
            >>> qs = QuantumScript(measurements=m)
            >>> qs.shape(dev)
            ((4,), (), (4,))
        """
        shapes = tuple(meas_process.shape(device) for meas_process in self._measurements)

        if len(shapes) == 1:
            return shapes[0]

        if device.shot_vector is not None:
            # put the shot vector axis before the measurement axis
            shapes = tuple(zip(*shapes))

        return shapes

    @property
    def numeric_type(self):
        """Returns the expected numeric type of the script result by inspecting
        its measurements.

        Raises:
            ValueError: raised for unsupported cases for
                example when the script contains heterogeneous measurements

        Returns:
            type: the numeric type corresponding to the result type of the
            script

        **Example:**

        >>> qscript = QuantumScript(measurements=[qml.state()])
        >>> qscript.numeric_type
        complex
        """
        if qml.active_return():
            return self._numeric_type_new
        measurement_types = {meas.return_type for meas in self._measurements}
        if len(measurement_types) > 1:
            raise ValueError(
                "Getting the numeric type of a quantum script that contains multiple types of measurements is unsupported."
            )

        # Note: if one of the sample measurements contains outputs that
        # are real, then the entire result will be real
        if list(measurement_types)[0] == qml.measurements.Sample:
            return next((float for mp in self._measurements if mp.numeric_type is float), int)

        return self._measurements[0].numeric_type

    @property
    def _numeric_type_new(self):
        """Returns the expected numeric type of the tape result by inspecting
        its measurements.

        Returns:
            Union[type, Tuple[type]]: The numeric type corresponding to the result type of the
            tape, or a tuple of such types if the tape contains multiple measurements

        **Example:**

        .. code-block:: pycon

            >>> qml.enable_return()
            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.numeric_type
            complex
        """
        types = tuple(observable.numeric_type for observable in self._measurements)

        return types[0] if len(types) == 1 else types

    # ========================================================
    # Transforms: Tape to Tape
    # ========================================================

    def copy(self, copy_operations=False):
        """Returns a shallow copy of the quantum script.

        Args:
            copy_operations (bool): If True, the operations are also shallow copied.
                Otherwise, if False, the copied operations will simply be references
                to the original operations; changing the parameters of one script will likewise
                change the parameters of all copies.

        Returns:
            QuantumScript : a shallow copy of the quantum script
        """
        if copy_operations:
            # Perform a shallow copy of all operations in the state prep, operation, and measurement
            # queues. The operations will continue to share data with the original tape operations
            # unless modified.
            _prep = [copy.copy(op) for op in self._prep]
            _ops = [copy.copy(op) for op in self._ops]
            _measurements = [copy.copy(op) for op in self._measurements]
        else:
            # Perform a shallow copy of the state prep, operation, and measurement queues. The
            # operations within the queues will be references to the original tape operations;
            # changing the original operations will always alter the operations on the copied tape.
            _prep = self._prep.copy()
            _ops = self._ops.copy()
            _measurements = self._measurements.copy()

        new_qscript = self.__class__(ops=_ops, measurements=_measurements, prep=_prep)
        new_qscript._graph = None if copy_operations else self._graph
        new_qscript._specs = None
        new_qscript.wires = copy.copy(self.wires)
        new_qscript.num_wires = self.num_wires
        new_qscript.is_sampled = self.is_sampled
        new_qscript.all_sampled = self.all_sampled
        new_qscript._update_par_info()
        new_qscript.trainable_params = self.trainable_params.copy()
        new_qscript._obs_sharing_wires = self._obs_sharing_wires
        new_qscript._obs_sharing_wires_id = self._obs_sharing_wires_id
        new_qscript._batch_size = self.batch_size
        new_qscript._output_dim = self.output_dim

        return new_qscript

    def __copy__(self):
        return self.copy(copy_operations=True)

    def expand(self, depth=1, stop_at=None, expand_measurements=False):
        """Expand all operations to a specific depth.

        Args:
            depth (int): the depth the script should be expanded
            stop_at (Callable): A function which accepts a queue object,
                and returns ``True`` if this object should *not* be expanded.
                If not provided, all objects that support expansion will be expanded.
            expand_measurements (bool): If ``True``, measurements will be expanded
                to basis rotations and computational basis measurements.

        **Example**

        Consider the following nested quantum script:

        >>> prep = [qml.BasisState(np.array([1, 1]), wires=[0, 'a'])]
        >>> nested_script = QuantumScript([qml.Rot(0.543, 0.1, 0.4, wires=0)])
        >>> ops = [nested_script, qml.CNOT(wires=[0, 'a']), qml.RY(0.2, wires='a')]
        >>> measurements = [qml.probs(wires=0), qml.probs(wires='a')]
        >>> qscript = QuantumScript(ops, measurements, prep)

        The nested structure is preserved:

        >>> qscript.operations
        [BasisState(tensor([1, 1], requires_grad=True), wires=[0, 'a']),
        <QuantumScript: wires=[0], params=3>,
        CNOT(wires=[0, 'a']),
        RY(0.2, wires=['a'])]

        Calling ``.expand`` will return a script with all nested scripts
        expanded, resulting in a single script of quantum operations:

        >>> new_qscript = qscript.expand(depth=2)
        >>> new_qscript.operations
        [PauliX(wires=[0]),
        PauliX(wires=['a']),
        RZ(0.543, wires=[0]),
        RY(0.1, wires=[0]),
        RZ(0.4, wires=[0]),
        CNOT(wires=[0, 'a']),
        RY(0.2, wires=['a'])]
        """
        new_script = qml.tape.tape.expand_tape(
            self, depth=depth, stop_at=stop_at, expand_measurements=expand_measurements
        )
        new_script._update()
        return new_script

    # NOT MOVING OVER INV
    # As it will be deprecated soon.

    def adjoint(self):
        """Create a quantum script that is the adjoint of this one.

        Adjointed quantum scripts are the conjugated and transposed version of the
        original script. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        Returns:
            ~.QuantumScript: the adjointed script
        """
        with qml.QueuingManager.stop_recording():
            ops_adj = [qml.adjoint(op, lazy=False) for op in reversed(self._ops)]
        adj = self.__class__(ops=ops_adj, measurements=self._measurements, prep=self._prep)
        if self.do_queue:
            qml.QueuingManager.append(adj)
        return adj

    def unwrap(self):
        """A context manager that unwraps a quantum script with tensor-like parameters
        to NumPy arrays.

        Returns:
            ~.QuantumScript: the unwrapped quantum script

        **Example**

        >>> with tf.GradientTape():
        ...     qscript = QuantumScript([qml.RX(tf.Variable(0.1), 0),
        ...                             qml.RY(tf.constant(0.2), 0),
        ...                             qml.RZ(tf.Variable(0.3), 0)])
        ...     with qscript.unwrap():
        ...         print("Trainable params:", qscript.trainable_params)
        ...         print("Unwrapped params:", qscript.get_parameters())
        Trainable params: [0, 2]
        Unwrapped params: [0.1, 0.3]
        >>> qscript.get_parameters()
        [<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.1>,
        <tf.Tensor: shape=(), dtype=float32, numpy=0.2>,
        <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.3>]
        """
        return qml.tape.UnwrapTape(self)

    # ========================================================
    # Transforms: Tape to Information
    # ========================================================

    @property
    def graph(self):
        """Returns a directed acyclic graph representation of the recorded
        quantum circuit:

        >>> ops = [qml.QubitStateVector([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        >>> qscript.graph
        <pennylane.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .CircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = qml.CircuitGraph(
                self.operations, self.observables, self.wires, self._par_info, self.trainable_params
            )

        return self._graph

    @property
    def specs(self):
        """Resource information about a quantum circuit.

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain quantum script specifications

        **Example**

        >>> ops = [qml.Hadamard(0), qml.RX(0.26, 1), qml.CNOT((1,0)),
        ...         qml.Rot(1.8, -2.7, 0.2, 0), qml.Hadamard(1), qml.CNOT((0,1))]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))])

        Asking for the specs produces a dictionary as shown below:

        >>> qscript.specs['gate_sizes']
        defaultdict(int, {1: 4, 2: 2})
        >>> qscript.specs['gate_types']
        defaultdict(int, {'Hadamard': 2, 'RX': 1, 'CNOT': 2, 'Rot': 1})

        As ``defaultdict`` objects, any key not present in the dictionary returns 0.

        >>> qscript.specs['gate_types']['RZ']
        0

        """
        if self._specs is None:
            self._specs = {"gate_sizes": defaultdict(int), "gate_types": defaultdict(int)}

            for op in self.operations:
                # don't use op.num_wires to allow for flexible gate classes like QubitUnitary
                self._specs["gate_sizes"][len(op.wires)] += 1
                self._specs["gate_types"][op.name] += 1

            self._specs["num_operations"] = len(self.operations)
            self._specs["num_observables"] = len(self.observables)
            self._specs["num_diagonalizing_gates"] = len(self.diagonalizing_gates)
            self._specs["num_used_wires"] = self.num_wires
            self._specs["depth"] = self.graph.get_depth()
            self._specs["num_trainable_params"] = self.num_params

        return self._specs

    # pylint: disable=too-many-arguments
    def draw(
        self,
        wire_order=None,
        show_all_wires=False,
        decimals=None,
        max_length=100,
        show_matrices=False,
    ):
        """Draw the quantum script as a circuit diagram. See :func:`~.drawer.tape_text` for more information.

        Args:
            wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
            show_all_wires (bool): If True, all wires, including empty wires, are printed.
            decimals (int): How many decimal points to include when formatting operation parameters.
                Default ``None`` will omit parameters from operation labels.
            max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
                begin anew beneath the previous lines.
            show_matrices=False (bool): show matrix valued parameters below all circuit diagrams

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

    def to_openqasm(self, wires=None, rotations=True, measure_all=True, precision=None):
        """Serialize the circuit as an OpenQASM 2.0 program.

        Measurements are assumed to be performed on all qubits in the computational basis. An
        optional ``rotations`` argument can be provided so that output of the OpenQASM circuit is
        diagonal in the eigenbasis of the quantum script's observables. The measurement outputs can be
        restricted to only those specified in the script by setting ``measure_all=False``.

        .. note::

            The serialized OpenQASM program assumes that gate definitions
            in ``qelib1.inc`` are available.

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
        wires = wires or self.wires

        # add the QASM headers
        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

        if self.num_wires == 0:
            # empty circuit
            return qasm_str

        # create the quantum and classical registers
        qasm_str += f"qreg q[{len(wires)}];\n"
        qasm_str += f"creg c[{len(wires)}];\n"

        # get the user applied circuit operations
        operations = self.operations

        if rotations:
            # if requested, append diagonalizing gates corresponding
            # to circuit observables
            operations += self.diagonalizing_gates

        # decompose the queue
        # pylint: disable=no-member
        just_ops = QuantumScript(operations)
        operations = just_ops.expand(
            depth=2, stop_at=lambda obj: obj.name in OPENQASM_GATES
        ).operations

        # create the QASM code representing the operations
        for op in operations:
            try:
                gate = OPENQASM_GATES[op.name]
            except KeyError as e:
                raise ValueError(f"Operation {op.name} not supported by the QASM serializer") from e

            wire_labels = ",".join([f"q[{wires.index(w)}]" for w in op.wires.tolist()])
            params = ""

            if op.num_params > 0:
                # If the operation takes parameters, construct a string
                # with parameter values.
                if precision is not None:
                    params = "(" + ",".join([f"{p:.{precision}}" for p in op.parameters]) + ")"
                else:
                    # use default precision
                    params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

            qasm_str += f"{gate}{params} {wire_labels};\n"

        # apply computational basis measurements to each quantum register
        # NOTE: This is not strictly necessary, we could inspect self.observables,
        # and then only measure wires which are requested by the user. However,
        # some devices which consume QASM require all registers be measured, so
        # measure all wires by default to be safe.
        if measure_all:
            for wire in range(len(wires)):
                qasm_str += f"measure q[{wire}] -> c[{wire}];\n"
        else:
            measured_wires = qml.wires.Wires.all_wires([m.wires for m in self.measurements])

            for w in measured_wires:
                wire_indx = self.wires.index(w)
                qasm_str += f"measure q[{wire_indx}] -> c[{wire_indx}];\n"

        return qasm_str

    @property
    def shots(self):
        """Shots property.

        Returns:
            int: Number of circuit evaluations/random samples used to estimate
                expectation values of observables.
        """
        return self._shots

    @shots.setter
    def shots(self, shots):
        """Changes the number of shots.

        Args:
            shots (int): number of circuit evaluations/random samples used to estimate
                expectation values of observables

        Raises:
            DeviceError: if number of shots is less than 1
        """
        self.raw_shots = shots
        if shots is None:
            # device is in analytic mode
            self._shots = shots
            self._shot_vector = None

        elif isinstance(shots, int):
            # device is in sampling mode (unbatched)
            if shots < 1:
                raise ValueError(
                    f"The specified number of shots needs to be at least 1. Got {shots}."
                )

            self._shots = shots
            self._shot_vector = None

        elif isinstance(shots, Sequence) and not isinstance(shots, str):
            # device is in batched sampling mode
            self._shots, self._shot_vector = _process_shot_sequence(shots)

        else:
            raise ValueError(
                "Shots must be a single non-negative integer or a sequence of non-negative integers."
            )

    @property
    def shot_vector(self):
        """Returns a sparse representation of the shot sequence.

        Returns:
            list[.ShotTuple[int, int]]: sparse representation of the shot sequence

        **Example**

        >>> tape = QuantumTape(shots=[3, 1, 2, 2, 2, 2, 6, 1, 1, 5, 12, 10, 10])
        >>> tape.shots
        57
        >>> tape.shot_vector
        [ShotTuple(shots=3, copies=1),
         ShotTuple(shots=1, copies=1),
         ShotTuple(shots=2, copies=4),
         ShotTuple(shots=6, copies=1),
         ShotTuple(shots=1, copies=2),
         ShotTuple(shots=5, copies=1),
         ShotTuple(shots=12, copies=1),
         ShotTuple(shots=10, copies=2)]

        The sparse representation of the shot
        sequence is returned, where tuples indicate the number of times a shot
        integer is repeated.
        """
        return self._shot_vector

    @property
    def analytic(self):
        """Whether shots is None or not. Kept for backwards compatability."""
        return self._shots is None

    def has_partitioned_shots(self):
        """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
        return self._shot_vector is not None and (
            len(self._shot_vector) > 1 or self._shot_vector[0].copies > 1
        )
