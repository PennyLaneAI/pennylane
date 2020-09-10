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
# pylint: disable=too-many-instance-attributes,protected-access,too-many-branches
import contextlib

import numpy as np

import pennylane as qml

from pennylane.beta.queuing import MeasurementProcess
from pennylane.beta.queuing import AnnotatedQueue, QueuingContext
from pennylane.beta.queuing import mock_operations

from .circuit_graph import NewCircuitGraph


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


class QuantumTape(AnnotatedQueue):
    """A quantum tape recorder, that records, validates, executes,
    and differentiates variational quantum programs.

    .. note::

        As the quantum tape is a *beta* feature, the standard PennyLane
        measurement functions cannot be used. You will need to instead
        import modified measurement functions within the quantum tape:

        >>> from pennylane.beta.queuing import expval, var, sample, probs

    **Example**

    .. code-block:: python

        from pennylane.beta.tapes import QuantumTape
        from pennylane.beta.queuing import expval, var, sample, probs

        with QuantumTape() as tape:
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

    The :class:`~.beta.tapes.NewCircuitGraph` can also be accessed:

    >>> tape.graph
    <pennylane.beta.tapes.circuit_graph.NewCircuitGraph object at 0x7fcc0433a690>

    Once constructed, the quantum tape can be executed directly on a supported
    device:

    >>> dev = qml.device("default.qubit", wires=[0, 'a'])

    Execution can take place either using the in-place constructed parameters,
    >>> tape.execute(dev)
    [0.77750694]

    or by providing parameters at run time:

    >>> tape.execute(dev, params=[0.1, 0.1, 0.1])
    [0.99003329]

    The Jacobian can also be computed using finite difference:

    >>> tape.jacobian(dev)
    [[-0.35846484 -0.46923704  0.        ]]
    >>> tape.jacobian(dev, params=[0.1, 0.1, 0.1])
    [[-0.09933471 -0.09933471  0.        ]]

    Finally, the trainable parameters can be explicitly set, and the values of
    the parameters modified in-place:

    >>> tape.trainable_params = {0} # set only the first parameter as free
    >>> tape.set_parameters(0.56)
    >>> tape.get_parameters()
    [0.56]
    >>> tape.get_parameters(free_only=False)
    [0.56, 0.543, 0.133]

    Trainable parameters are taken into account when calculating the Jacobian,
    avoiding unnecessary calculations:

    >>> tape.jacobian(dev)
    [[-0.45478169]]
    """

    cast = staticmethod(np.array)

    def __init__(self, name=None):
        super().__init__()
        self.name = name

        self._prep = []
        """list[~.Operation]: Tape state preparations."""

        self._ops = []
        """list[~.Operation]: quantum operations recorded by the tape."""

        self._obs = []
        """list[tuple[~.MeasurementProcess, ~.Observable]]: measurement processes and
        coresponding observables recorded by the tape."""

        self._par_info = {}
        """dict[int, dict[str, Operation or int]]: Parameter information. Keys are
        parameter indices (in the order they appear on the tape), and values are a
        dictionary containing the corresponding operation and operation parameter index."""

        self._trainable_params = set()
        self._graph = None
        self._output_dim = 0

        self.wires = qml.wires.Wires([])
        self.num_wires = 0

        self.jacobian_options = {}

        self.hash = 0
        self.is_sampled = False

        self._stack = None

    def __repr__(self):
        return f"<{self.__class__.__name__}: wires={self.wires}, params={self.num_params}>"

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
        """str, None: automatic differentiation interface used by the quantum tap (if any)"""
        return None

    # ========================================================
    # construction methods
    # ========================================================

    def _process_queue(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes.

        This method sets the following attributes:

        * ``_ops``
        * ``_obs``
        * ``_par_info``
        * ``_output_dim``
        * ``_trainable_params``
        * ``is_sampled``
        """
        self._prep = []
        self._ops = []
        self._obs = []
        self._output_dim = 0

        for obj, info in self._queue.items():

            if isinstance(obj, QuantumTape):
                self._ops.append(obj)

            elif isinstance(obj, qml.operation.Operation) and not info.get("owner", False):
                # operation objects with no owners
                if self._obs:
                    raise ValueError(
                        f"Quantum operation {obj} must occur prior to any measurements."
                    )

                obj.do_check_domain = False

                # invert the operation if required
                obj.inverse = info.get("inverse", False)

                if isinstance(obj, STATE_PREP_OPS):
                    if self._ops:
                        raise ValueError(
                            f"State preperation operation {obj} must occur prior to any quantum operations."
                        )

                    self._prep.append(obj)
                else:
                    self._ops.append(obj)

            elif isinstance(obj, MeasurementProcess):
                # measurement process

                if obj.return_type is qml.operation.Probability:
                    self._obs.append((obj, obj))
                    self._output_dim += 2 ** len(obj.wires)

                elif "owns" in info:
                    # TODO: remove the following line once devices
                    # have been refactored to no longer use obs.return_type.
                    # Monkeypatch the observable to have the same return
                    # type as the measurement process.
                    info["owns"].return_type = obj.return_type
                    info["owns"].do_check_domain = False

                    self._obs.append((obj, info["owns"]))
                    self._output_dim += 1

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

    def _expand(self, stop_at=None):
        """Expand all operations and tapes in the processed queue.

        Args:
            stop_at (Sequence[str]): Sequence of PennyLane operation or tape names.
                An operation or tape appearing in this list of names is not expanded.
        """
        new_tape = self.__class__()
        stop_at = stop_at or []

        for op in self.operations:
            if op.name in stop_at:
                new_tape._ops.append(op)
                continue

            t = op if isinstance(op, QuantumTape) else op.expand()
            new_tape._prep += t._prep
            new_tape._ops += t._ops

        new_tape._obs = self._obs
        return new_tape

    def expand(self, depth=1, stop_at=None):
        """Expand all operations in the processed to a specific depth.

        Args:
            depth (int): the depth the tape should be expanded
            stop_at (Sequence[str]): Sequence of PennyLane operation or tape names.
                An operation or tape appearing in this list of names is not expanded.

        **Example**

        Consider the following nested tape:

        .. code-block:: python

            with QuantumTape() as tape:
                qml.BasisState(np.array([1, 1]), wires=[0, 'a'])

                with QuantumTape() as tape2:
                    qml.Rot(0.543, 0.1, 0.4, wires=0)

                qml.CNOT(wires=[0, 'a'])
                qml.RY(0.2, wires='a')
                probs(wires=0), probs(wires='a')

        The nested structure is preserved:

        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         <QuantumTape: params=3, wires=<Wires = [0]>>,
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
        old_tape = self

        for _ in range(depth):
            new_tape = old_tape._expand(stop_at=stop_at)

            if len(new_tape.operations) == len(old_tape.operations) and [
                o1.name == o2.name for o1, o2 in zip(new_tape.operations, old_tape.operations)
            ]:
                # expansion has found a fixed point
                break

            old_tape = new_tape

        new_tape._update()
        return new_tape

    def inv(self):
        """Inverts the processed operations.

        Inversion is performed in-place.

        .. note::

            This method only inverts the quantum operations/unitary recorded
            by the quantum tape; state preprations and measurements are left unchanged.

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
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

        Here, lets set some trainable parameters:

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

        Note that the state preparation remains as-is, while the operations have been
        inverted and their order reversed.

        Tape inversion also modifies the order of tape parameters:

        >>> tape.get_parameters(free_only=False)
        [array([1, 1]), 0.543, 0.1, 0.4, 0.432]
        >>> tape.get_parameters(free_only=True)
        [0.543, 0.432]
        >>> tape.trainable_params
        {1, 4}
        """
        for op in self._ops:
            op.inverse = not op.inverse

        if self.trainable_params != set(range(len(self._par_info))):
            # if the trainable parameters have been set to a subset
            # of all parameters, we must remap the old trainable parameter
            # indices to the new ones after the operation order is reversed.
            parameter_indices = []
            param_count = 0

            for idx, queue in enumerate([self._prep, self._ops, self.observables]):
                # iterate through all queues

                obj_params = []

                for obj in queue:
                    # index the number of parameters on each operation
                    num_obj_params = len(obj.data)
                    obj_params.append(list(range(param_count, param_count + num_obj_params)))

                    # keep track of the total number of parameters encountered so far
                    param_count += num_obj_params

                if idx == 1:
                    # reverse the list representing operator parameters
                    obj_params = obj_params[::-1]

                parameter_indices.extend(obj_params)

            # flatten the list of parameter indices after the reversal
            parameter_indices = [item for sublist in parameter_indices for item in sublist]

            # remap the trainable parameter information
            trainable_params = set()

            for old_idx, new_idx in zip(parameter_indices, range(len(parameter_indices))):
                if old_idx in self.trainable_params:
                    trainable_params.add(new_idx)

            self.trainable_params = trainable_params

        self._ops = list(reversed(self._ops))

    # ========================================================
    # properties, setters, and getters
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
        :meth:`~.set_parameters`, :meth:`~.QuantumTape.execute`, and :meth:`~.QuantumTape.jacobian`,
        and changes the default output size of methods :meth:`~.QuantumTape.jacobian` and
        :meth:`~.get_parameters()`.

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
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

    @property
    def operations(self):
        """Returns the operations on the quantum tape.

        Returns:
            list[.Operation]: list of recorded quantum operations

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
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

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.operations
        [expval(PauliZ(wires=[0]))]
        """
        return [m[1] for m in self._obs]

    @property
    def measurements(self):
        """Returns the measurements on the quantum tape.

        Returns:
            list[.MeasurementProcess]: list of recorded measurement processess

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        >>> tape.measurements
        [<pennylane.beta.queuing.measure.MeasurementProcess object at 0x7f10b2150c10>]
        """
        return [m[0] for m in self._obs]

    @property
    def num_params(self):
        """Returns the number of trainable parameters on the quantum tape."""
        return len(self.trainable_params)

    @property
    def output_dim(self):
        """The (estimated) output dimension of the quantum tape."""
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
        <pennylane.beta.tapes.circuit_graph.NewCircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .beta.tapes.NewCircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = NewCircuitGraph(self.operations, self.observables, self.wires)

        return self._graph

    def get_parameters(self, free_only=True):
        """Return the parameters incident on the tape operations.

        The returned parameters are provided in order of appearance
        on the tape.

        Args:
            free_only (bool): if True, returns only trainable parameters

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
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

        The ``free_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> tape.get_parameters(free_only=False)
        [0.432, 0.543, 0.133]
        """
        params = [o.data for o in self.operations + self.observables]
        params = [item for sublist in params for item in sublist]

        if not free_only:
            return params

        return [p for idx, p in enumerate(params) if idx in self.trainable_params]

    def set_parameters(self, params, free_only=True):
        """Set the parameters incident on the tape operations.

        Args:
            params (list[float]): A list of real numbers representing the
                parameters of the quantum operations. The parameters should be
                provided in order of appearance in the quantum tape.
            free_only (bool): if True, set only trainable parameters

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
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
        >>> tape.get_parameters(free_only=False)
        [-0.1, 0.2, 0.5]

        The ``free_only`` argument can be set to ``False`` to instead set
        all parameters:

        >>> tape.set_parameters([4, 1, 6])
        >>> tape.get_parameters(free_only=False)
        [4, 1, 6]
        """
        if free_only:
            iterator = zip(self.trainable_params, params)
            required_length = self.num_params
        else:
            iterator = enumerate(params)
            required_length = len(self._par_info)

        if len(params) != required_length:
            raise ValueError("Number of provided parameters invalid.")

        for idx, p in iterator:
            op = self._par_info[idx]["op"]
            op.data[self._par_info[idx]["p_idx"]] = p

    @property
    def data(self):
        """Alias to :meth:`~.get_parameters` and :meth:`~.set_parameters`
        for backwards compatibilities with operations."""
        return self.get_parameters(free_only=False)

    @data.setter
    def data(self, params):
        self.set_parameters(params, free_only=False)
