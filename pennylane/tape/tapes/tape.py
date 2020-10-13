# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the base quantum tape.
"""
# pylint: disable=too-many-instance-attributes,protected-access,too-many-branches,too-many-public-methods
from collections import OrderedDict
import contextlib

import numpy as np

import pennylane as qml

from pennylane.tape.circuit_graph import TapeCircuitGraph
from pennylane.tape.operation import mock_operations
from pennylane.tape.queuing import AnnotatedQueue, QueuingContext


STATE_PREP_OPS = (
    qml.BasisState,
    qml.QubitStateVector,
    qml.CatState,
    qml.CoherentState,
    qml.FockDensityMatrix,
    qml.DisplacedSqueezedState,
    qml.FockState,
    qml.FockStateVector,
    qml.ThermalState,
    qml.GaussianState,
)


def expand_tape(tape, depth=1, stop_at=None, expand_measurements=False):
    """Expand all objects in a tape to a specific depth.

    Args:
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.
        expand_measurements (bool): If ``True``, measurements will be expanded
            to basis rotations and computational basis measurements.

    **Example**

    Consider the following nested tape:

    .. code-block:: python

        with JacobianTape() as tape:
            qml.BasisState(np.array([1, 1]), wires=[0, 'a'])

            with JacobianTape() as tape2:
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, 'a'])
            qml.RY(0.2, wires='a')
            probs(wires=0), probs(wires='a')

    The nested structure is preserved:

    >>> tape.operations
    [BasisState(array([1, 1]), wires=[0, 'a']),
     <JacobianTape: wires=[0], params=3>,
     CNOT(wires=[0, 'a']),
     RY(0.2, wires=['a'])]

    Calling ``expand_tape`` will return a tape with all nested tapes
    expanded, resulting in a single tape of quantum operations:

    >>> new_tape = expand_tape(tape)
    >>> new_tape.operations
    [PauliX(wires=[0]),
     PauliX(wires=['a']),
     Rot(0.543, 0.1, 0.4, wires=[0]),
     CNOT(wires=[0, 'a']),
     RY(0.2, wires=['a'])]
    """
    if depth == 0:
        return tape

    if stop_at is None:
        # by default expand all objects
        stop_at = lambda obj: False

    new_tape = tape.__class__()

    for queue in ("_prep", "_ops", "_measurements"):
        for obj in getattr(tape, queue):

            stop = stop_at(obj)

            if not expand_measurements:
                # Measurements should not be expanded; treat measurements
                # as a stopping condition
                stop = stop or isinstance(obj, qml.tape.measure.MeasurementProcess)

            if stop:
                # do not expand out the object; append it to the
                # new tape, and continue to the next object in the queue
                getattr(new_tape, queue).append(obj)
                continue

            if isinstance(obj, (qml.operation.Operation, qml.tape.measure.MeasurementProcess)):
                # Object is an operation; query it for its expansion
                try:
                    obj = obj.expand()
                except NotImplementedError:
                    # Object does not define an expansion; treat this as
                    # a stopping condition.
                    getattr(new_tape, queue).append(obj)
                    continue

            # recursively expand out the newly created tape
            expanded_tape = expand_tape(obj, stop_at=stop_at, depth=depth - 1)

            new_tape._prep += expanded_tape._prep
            new_tape._ops += expanded_tape._ops
            new_tape._measurements += expanded_tape._measurements

    return new_tape


# pylint: disable=too-many-public-methods
class QuantumTape(AnnotatedQueue):
    """A quantum tape recorder, that records, validates and executes variational quantum programs.

    .. note::

        As the quantum tape is a *beta* feature. See :mod:`pennylane.tape`
        for more details.

    Args:
        name (str): a name given to the quantum tape
        caching (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.

    **Example**

    .. code-block:: python

        import pennylane.tape

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            expval(qml.PauliZ(wires=[0]))

    Once constructed, information about the quantum circuit can be queried:

    >>> tape.operations
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
    >>> tape.observables
    [expval(PauliZ(wires=[0]))]
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> tape.wires
    <Wires = [0, 'a']>
    >>> tape.num_params
    3

    The :class:`~.TapeCircuitGraph` can also be accessed:

    >>> tape.graph
    <pennylane.tape.circuit_graph.TapeCircuitGraph object at 0x7fcc0433a690>

    Once constructed, the quantum tape can be executed directly on a supported
    device:

    >>> dev = qml.device("default.qubit", wires=[0, 'a'])

    Execution can take place either using the in-place constructed parameters,

    >>> tape.execute(dev)
    [0.77750694]

    or by providing parameters at run time:

    >>> tape.execute(dev, params=[0.1, 0.1, 0.1])
    [0.99003329]

    The trainable parameters of the tape can be explicitly set, and the values of
    the parameters modified in-place:

    >>> tape.trainable_params = {0} # set only the first parameter as free
    >>> tape.set_parameters(0.56)
    >>> tape.get_parameters()
    [0.56]
    >>> tape.get_parameters(trainable_only=False)
    [0.56, 0.543, 0.133]
    """

    def __init__(self, name=None, caching=0):
        super().__init__()
        self.name = name

        self._prep = []
        """list[.Operation]: Tape state preparations."""

        self._ops = []
        """list[.Operation]: quantum operations recorded by the tape."""

        self._measurements = []
        """list[.MeasurementProcess]: measurement processes recorded by the tape."""

        self._par_info = {}
        """dict[int, dict[str, Operation or int]]: Parameter information. Keys are
        parameter indices (in the order they appear on the tape), and values are a
        dictionary containing the corresponding operation and operation parameter index."""

        self._trainable_params = set()
        self._graph = None
        self._output_dim = 0

        self.wires = qml.wires.Wires([])
        self.num_wires = 0

        self.hash = 0
        self.is_sampled = False
        self.inverse = False

        self._stack = None

        self._caching = caching
        """float: number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""

        self._cache_execute = OrderedDict()
        """OrderedDict[int: Any]: Mapping from hashes of the circuit to results of executing the
        device."""

    def __repr__(self):
        return f"<{self.__class__.__name__}: wires={self.wires.tolist()}, params={self.num_params}>"

    def __enter__(self):
        if not QueuingContext.recording():
            # if the tape is the first active queuing context
            # monkeypatch the operations to support the new queuing context
            with contextlib.ExitStack() as stack:
                for mock in mock_operations():
                    stack.enter_context(mock)
                self._stack = stack.pop_all()

        QueuingContext.append(self)
        return super().__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)

        if not QueuingContext.recording():
            # remove the monkeypatching
            self._stack.__exit__(exception_type, exception_value, traceback)

        self._process_queue()

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum tape (if any)"""
        return None

    # ========================================================
    # construction methods
    # ========================================================

    def _process_queue(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes.

        This method sets the following attributes:

        * ``_ops``
        * ``_measurements``
        * ``_par_info``
        * ``_output_dim``
        * ``_trainable_params``
        * ``is_sampled``
        """
        self._prep = []
        self._ops = []
        self._measurements = []
        self._output_dim = 0

        for obj, info in self._queue.items():

            if isinstance(obj, QuantumTape):
                self._ops.append(obj)

            elif isinstance(obj, qml.operation.Operation) and not info.get("owner", False):
                # operation objects with no owners

                if self._measurements:
                    raise ValueError(
                        f"Quantum operation {obj} must occur prior to any measurements."
                    )

                # invert the operation if required
                obj.inverse = info.get("inverse", False)

                if isinstance(obj, STATE_PREP_OPS):
                    if self._ops:
                        raise ValueError(
                            f"State preparation operation {obj} must occur prior to any quantum operations."
                        )

                    self._prep.append(obj)
                else:
                    self._ops.append(obj)

            elif isinstance(obj, qml.tape.measure.MeasurementProcess):
                # measurement process
                self._measurements.append(obj)

                # attempt to infer the output dimension
                if obj.return_type is qml.operation.Probability:
                    self._output_dim += 2 ** len(obj.wires)
                elif obj.return_type is qml.operation.State:
                    continue  # the output_dim is worked out automatically
                else:
                    self._output_dim += 1

                # check if any sampling is occuring
                if obj.return_type is qml.operation.Sample:
                    self.is_sampled = True

            elif isinstance(obj, qml.operation.Observable) and "owner" not in info:
                raise ValueError(f"Observable {obj} does not have a measurement type specified.")

        self._update()

    def _update_circuit_info(self):
        """Update circuit metadata"""
        self.wires = qml.wires.Wires.all_wires(
            [op.wires for op in self.operations + self.observables]
        )
        self.num_wires = len(self.wires)

    def _update_par_info(self):
        """Update the parameter information dictionary"""
        param_count = 0

        for obj in self.operations + self.observables:

            for p in range(len(obj.data)):
                info = self._par_info.get(param_count, {})
                info.update({"op": obj, "p_idx": p})

                self._par_info[param_count] = info
                param_count += 1

    def _update_trainable_params(self):
        """Set the trainable parameters"""
        self._trainable_params = set(self._par_info)

    def _update(self):
        """Update all internal tape metadata regarding processed operations and observables"""
        self._graph = None
        self._update_circuit_info()
        self._update_par_info()
        self._update_trainable_params()

    def expand(self, depth=1, stop_at=None, expand_measurements=False):
        """Expand all operations in the processed queue to a specific depth.

        Args:
            depth (int): the depth the tape should be expanded
            stop_at (Callable): A function which accepts a queue object,
                and returns ``True`` if this object should *not* be expanded.
                If not provided, all objects that support expansion will be expanded.
            expand_measurements (bool): If ``True``, measurements will be expanded
                to basis rotations and computational basis measurements.

        **Example**

        Consider the following nested tape:

        .. code-block:: python

            with JacobianTape() as tape:
                qml.BasisState(np.array([1, 1]), wires=[0, 'a'])

                with JacobianTape() as tape2:
                    qml.Rot(0.543, 0.1, 0.4, wires=0)

                qml.CNOT(wires=[0, 'a'])
                qml.RY(0.2, wires='a')
                probs(wires=0), probs(wires='a')

        The nested structure is preserved:

        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         <JacobianTape: wires=[0], params=3>,
         CNOT(wires=[0, 'a']),
         RY(0.2, wires=['a'])]

        Calling ``.expand`` will return a tape with all nested tapes
        expanded, resulting in a single tape of quantum operations:

        >>> new_tape = tape.expand()
        >>> new_tape.operations
        [PauliX(wires=[0]),
         PauliX(wires=['a']),
         Rot(0.543, 0.1, 0.4, wires=[0]),
         CNOT(wires=[0, 'a']),
         RY(0.2, wires=['a'])]
        """
        new_tape = expand_tape(
            self, depth=depth, stop_at=stop_at, expand_measurements=expand_measurements
        )
        new_tape._update()
        return new_tape

    def inv(self):
        """Inverts the processed operations.

        Inversion is performed in-place.

        .. note::

            This method only inverts the quantum operations/unitary recorded
            by the quantum tape; state preparations and measurements are left unchanged.

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.BasisState(np.array([1, 1]), wires=[0, 'a'])
                qml.RX(0.432, wires=0)
                qml.Rot(0.543, 0.1, 0.4, wires=0).inv()
                qml.CNOT(wires=[0, 'a'])
                probs(wires=0), probs(wires='a')

        This tape has the following properties:

        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         RX(0.432, wires=[0]),
         Rot.inv(0.543, 0.1, 0.4, wires=[0]),
         CNOT(wires=[0, 'a'])]
        >>> tape.get_parameters()
        [array([1, 1]), 0.432, 0.543, 0.1, 0.4]

        Here, let's set some trainable parameters:

        >>> tape.trainable_params = {1, 2}
        >>> tape.get_parameters()
        [0.432, 0.543]

        Inverting the tape:

        >>> tape.inv()
        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         CNOT.inv(wires=[0, 'a']),
         Rot(0.543, 0.1, 0.4, wires=[0]),
         RX.inv(0.432, wires=[0])]

        Tape inversion also modifies the order of tape parameters:

        >>> tape.get_parameters(trainable_only=False)
        [array([1, 1]), 0.543, 0.1, 0.4, 0.432]
        >>> tape.get_parameters(trainable_only=True)
        [0.543, 0.432]
        >>> tape.trainable_params
        {1, 4}
        """
        # we must remap the old parameter
        # indices to the new ones after the operation order is reversed.
        parameter_indices = []
        param_count = 0

        for queue in [self._prep, self._ops, self.observables]:
            # iterate through all queues

            obj_params = []

            for obj in queue:
                # index the number of parameters on each operation
                num_obj_params = len(obj.data)
                obj_params.append(list(range(param_count, param_count + num_obj_params)))

                # keep track of the total number of parameters encountered so far
                param_count += num_obj_params

            if queue == self._ops:
                # reverse the list representing operator parameters
                obj_params = obj_params[::-1]

            parameter_indices.extend(obj_params)

        # flatten the list of parameter indices after the reversal
        parameter_indices = [item for sublist in parameter_indices for item in sublist]
        parameter_mapping = dict(zip(parameter_indices, range(len(parameter_indices))))

        # map the params
        self.trainable_params = {parameter_mapping[i] for i in self.trainable_params}
        self._par_info = {parameter_mapping[k]: v for k, v in self._par_info.items()}

        for op in self._ops:
            op.inverse = not op.inverse

        self._ops = list(reversed(self._ops))

    # ========================================================
    # Parameter handling
    # ========================================================

    @property
    def trainable_params(self):
        """Store or return a set containing the indices of parameters that support
        differentiability. The indices provided match the order of appearence in the
        quantum circuit.

        Setting this property can help reduce the number of quantum evaluations needed
        to compute the Jacobian; parameters not marked as trainable will be
        automatically excluded from the Jacobian computation.

        The number of trainable parameters determines the number of parameters passed to
        :meth:`~.set_parameters`, :meth:`~.execute`, and :meth:`~.JacobianTape.jacobian`,
        and changes the default output size of methods :meth:`~.JacobianTape.jacobian` and
        :meth:`~.get_parameters()`.

        .. note::

            Since the :meth:`~.JacobianTape.jacobian` method is not called for devices that support
            native backpropagation (such as ``default.qubit.tf`` and ``default.qubit.autograd``),
            this property contains no relevant information when using backpropagation to compute gradients.

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.trainable_params
        {0, 1, 2}
        >>> tape.trainable_params = {0} # set only the first parameter as free
        >>> tape.get_parameters()
        [0.432]

        Args:
            param_indices (set[int]): parameter indices
        """
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (set[int]): parameter indices
        """
        if any(not isinstance(i, int) or i < 0 for i in param_indices):
            raise ValueError("Argument indices must be positive integers.")

        if any(i > len(self._par_info) for i in param_indices):
            raise ValueError(f"Tape has at most {self.num_params} parameters.")

        self._trainable_params = param_indices

    def get_parameters(self, trainable_only=True):
        """Return the parameters incident on the tape operations.

        The returned parameters are provided in order of appearance
        on the tape.

        Args:
            trainable_only (bool): if True, returns only trainable parameters

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and will be returned:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> tape.trainable_params = {1} # set the second parameter as free
        >>> tape.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> tape.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        params = []
        iterator = self.trainable_params if trainable_only else self._par_info

        for p_idx in iterator:
            op = self._par_info[p_idx]["op"]
            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])

        return params

    def set_parameters(self, params, trainable_only=True):
        """Set the parameters incident on the tape operations.

        Args:
            params (list[float]): A list of real numbers representing the
                parameters of the quantum operations. The parameters should be
                provided in order of appearance in the quantum tape.
            trainable_only (bool): if True, set only trainable parameters

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and can be modified:

        >>> tape.set_parameters([0.1, 0.2, 0.3])
        >>> tape.get_parameters()
        [0.1, 0.2, 0.3]

        Setting the trainable parameter indices will result in only the specified
        parameters being modifiable. Note that this only modifies the number of
        parameters that must be passed.

        >>> tape.trainable_params = {0, 2} # set the first and third parameter as free
        >>> tape.set_parameters([-0.1, 0.5])
        >>> tape.get_parameters(trainable_only=False)
        [-0.1, 0.2, 0.5]

        The ``trainable_only`` argument can be set to ``False`` to instead set
        all parameters:

        >>> tape.set_parameters([4, 1, 6], trainable_only=False)
        >>> tape.get_parameters(trainable_only=False)
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

    # ========================================================
    # Tape properties
    # ========================================================

    @property
    def operations(self):
        """Returns the operations on the quantum tape.

        Returns:
            list[.Operation]: recorded quantum operations

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.operations
        [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
        """
        return self._prep + self._ops

    @property
    def observables(self):
        """Returns the observables on the quantum tape.

        Returns:
            list[.Observable]: list of recorded quantum operations

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.observables
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
        """Returns the measurements on the quantum tape.

        Returns:
            list[.MeasurementProcess]: list of recorded measurement processess

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.measurements
        [<pennylane.tape.measure.MeasurementProcess object at 0x7f10b2150c10>]
        """
        return self._measurements

    @property
    def num_params(self):
        """Returns the number of trainable parameters on the quantum tape."""
        return len(self.trainable_params)

    @property
    def output_dim(self):
        """The (inferred) output dimension of the quantum tape."""
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
            rotation_gates.extend(observable.diagonalizing_gates())

        return rotation_gates

    @property
    def graph(self):
        """Returns a directed acyclic graph representation of the recorded
        quantum circuit:

        >>> tape.graph
        <pennylane.tape.circuit_graph.TapeCircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .TapeCircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = TapeCircuitGraph(self.operations, self.observables, self.wires)

        return self._graph

    def draw(self, charset="unicode"):
        """Draw the quantum tape as a circuit diagram.

        Consider the following circuit as an example:

        .. code-block:: python3

            with QuantumTape() as tape:
                qml.Hadamard(0)
                qml.CRX(2.3, wires=[0, 1])
                qml.Rot(1.2, 3.2, 0.7, wires=[1])
                qml.CRX(-2.3, wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        We can draw the tape after construction:

        >>> print(tape.draw())
        0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
        1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
        >>> print(tape.draw(charset="ascii"))
        0: --H--+C----------------------------+C---------+| <Z @ Z>
        1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>

        Args:
            charset (str, optional): The charset that should be used. Currently, "unicode" and
                "ascii" are supported.

        Raises:
            ValueError: if the given charset is not supported

        Returns:
            str: the circuit representation of the tape
        """
        return self.graph.draw(charset=charset, show_variable_names=False)

    @property
    def data(self):
        """Alias to :meth:`~.get_parameters` and :meth:`~.set_parameters`
        for backwards compatibilities with operations."""
        return self.get_parameters(trainable_only=False)

    @data.setter
    def data(self, params):
        self.set_parameters(params, trainable_only=False)

    def copy(self):
        """Returns a shallow copy of the quantum tape."""
        tape = self.__class__()
        tape._prep = self._prep.copy()
        tape._ops = self._ops.copy()
        tape._measurements = self._measurements.copy()

        tape._update()

        tape._par_info = self._par_info.copy()
        tape.trainable_params = self.trainable_params.copy()
        return tape

    # ========================================================
    # execution methods
    # ========================================================

    def execute(self, device, params=None):
        """Execute the tape on a quantum device.

        Args:
            device (.Device): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameters are used (via :meth:`~.get_parameters`).

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                probs(wires=[0, 'a'])

        If parameters are not provided, the existing tape parameters are used:

        >>> dev = qml.device("default.qubit", wires=[0, 'a'])
        >>> tape.execute(dev)
        array([[8.84828969e-01, 3.92449987e-03, 4.91235209e-04, 1.10755296e-01]])

        Parameters can be optionally passed during execution:

        >>> tape.execute(dev, params=[1.0, 0.0, 1.0])
        array([[0.5931328 , 0.17701835, 0.05283049, 0.17701835]])

        Parameters provided for execution are temporary, and do not affect
        the tapes' parameters in-place:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]
        """
        if params is None:
            params = self.get_parameters()

        return self._execute(params, device=device)

    def execute_device(self, params, device):
        """Execute the tape on a quantum device.

        This is a low-level method, intended to be called by an interface,
        and does not support autodifferentiation.

        For more details on differentiable tape execution, see :meth:`~.execute`.

        Args:
            device (~.Device): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).
        """

        device.reset()

        # backup the current parameters
        saved_parameters = self.get_parameters()

        # temporarily mutate the in-place parameters
        self.set_parameters(params)

        if self._caching:
            circuit_hash = self.graph.hash
            if circuit_hash in self._cache_execute:
                self.set_parameters(saved_parameters)
                return self._cache_execute[circuit_hash]

        if isinstance(device, qml.QubitDevice):
            res = device.execute(self)
        else:
            res = device.execute(self.operations, self.observables, {})

        # Update output dim if incorrect.
        # Note that we cannot assume the type of `res`, so
        # we use duck typing to catch any 'array like' object.
        try:
            if isinstance(res, np.ndarray) and res.dtype is np.dtype("object"):
                output_dim = sum([len(i) for i in res])
            else:
                output_dim = np.prod(res.shape)

            if self.output_dim != output_dim:
                # update the inferred output dimension with the correct value
                self._output_dim = output_dim

        except (AttributeError, TypeError):
            # unable to determine the output dimension
            pass

        # restore original parameters
        self.set_parameters(saved_parameters)

        if self._caching and circuit_hash not in self._cache_execute:
            self._cache_execute[circuit_hash] = res
            if len(self._cache_execute) > self._caching:
                self._cache_execute.popitem(last=False)

        return res

    # interfaces can optionally override the _execute method
    # if they need to perform any logic in between the user's
    # call to tape.execute and the internal call to tape.execute_device.
    _execute = execute_device

    @property
    def caching(self):
        """float: number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""
        return self._caching
