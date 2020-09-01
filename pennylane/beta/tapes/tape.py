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
import numpy as np

import pennylane as qml

from pennylane.beta.queuing import MeasurementProcess
from pennylane.beta.queuing import AnnotatedQueue, QueuingContext
from pennylane.beta.queuing import monkeypatch_operations, unmonkeypatch_operations

from .circuit_graph import CircuitGraph


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

    The :class:`~.beta.tapes.CircuitGraph` can also be accessed:

    >>> tape.graph
    <pennylane.beta.tapes.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

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
        self.grad_method = None

        self.hash = 0
        self.is_sampled = False

    def __repr__(self):
        return f"<{self.__class__.__name__}: wires={self.wires}, params={self.num_params}>"

    def __enter__(self):
        monkeypatch_operations()
        QueuingContext.append(self)
        return super().__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)

        if not QueuingContext.recording:
            unmonkeypatch_operations()

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

                    self._obs.append((obj, info["owns"]))
                    self._output_dim += 1

                    if obj.return_type is qml.operation.Sample:
                        self.is_sampled = True

        self._update_circuit_info()
        self._update_par_info()
        self._update_gradient_info()
        self._update_trainable_params()

    def _update_circuit_info(self):
        """Update circuit metadata"""
        self.wires = qml.wires.Wires.all_wires(
            [op.wires for op in self.operations + self.observables]
        )
        self.num_wires = len(self.wires)

    def _update_gradient_info(self):
        """Update the parameter information dictionary with gradient information
        of each parameter"""
        gmeth = []

        for i, info in self._par_info.items():
            gmeth.append(self._grad_method(i, use_graph=True))
            info["grad_method"] = gmeth[-1]

    def _update_par_info(self):
        """Update the parameter information dictionary"""
        param_count = 0

        for obj in self.operations + self.observables:

            for p in range(len(obj.data)):
                info = self._par_info.get(param_count, {})
                info.update(
                    {
                        "op": obj,
                        "p_idx": p,
                    }
                )

                self._par_info[param_count] = info
                param_count += 1

    def _update_trainable_params(self):
        """Set the trainable parameters"""
        self._trainable_params = set(self._par_info)

    def _expand(self, stop_at=None):
        """Expand all operations and tapes in the processed queue.

        Args:
            stop_at (Sequence[str]): Sequence of PennyLane operation or tape names.
                An operation or tape appearing in this list of names is not expanded.
        """
        new_tape = QuantumTape()
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
        new_tape = self

        for _ in range(depth):
            new_tape = new_tape._expand(stop_at=stop_at)

        new_tape._update_circuit_info()
        new_tape._update_par_info()
        new_tape._update_gradient_info()
        new_tape._update_trainable_params()
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
        :meth:`~.set_parameters`, :meth:`~.execute`, and :meth:`~.jacobian`, and changes the default
        output size of methods :meth:`~.jacobian` and :meth:`~.get_parameters()`.

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
        <pennylane.beta.tapes.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .beta.tapes.CircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = CircuitGraph(self.operations, self.observables, self.wires)

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

    # ========================================================
    # execution methods
    # ========================================================

    def execute(self, device, params=None):
        """Execute the tape on a quantum device.

        Args:
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameters are used (via :meth:`~.get_parameters`).

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
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

        For more details, see :meth:`~.execute`.

        Args:
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).
        """
        device.reset()

        # backup the current parameters
        current_parameters = self.get_parameters()

        # temporarily mutate the in-place parameters
        self.set_parameters(params)

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
                # update the output dimension estimate with the correct value
                self._output_dim = output_dim

        except (AttributeError, TypeError):
            # unable to determine the output dimension
            pass

        # restore original parameters
        self.set_parameters(current_parameters)
        return res

    _execute = execute_device

    # ========================================================
    # gradient methods
    # ========================================================

    def _grad_method(self, idx, use_graph=True):
        """Determine the correct partial derivative computation method for each gate parameter.

        Parameter gradient methods include:

        * ``None``: the parameter does not support differentiation.

        * ``"0"``: the variational circuit output does not depend on this
            parameter (the partial derivative is zero).

        * ``"F"``: the parameter has a non-zero derivative that should be computed
            using finite-differences.

        * ``"A"``: the parameter has a non-zero derivative that should be computed
            using an analytic method.

        .. note::

            The base ``QuantumTape`` class only supports numerical differentiation, so
            this method will always return either ``"F"`` or ``None``. If an inheriting
            tape supports analytic differentiation for certain operations, make sure
            that this method is overwritten appropriately to return ``"A"`` where
            required.

        Args:
            idx (int): parameter index
            use_graph: whether to use a directed-acyclic graph to determine
                if the parameter has a gradient of 0

        Returns:
            str: partial derivative method to be used
        """
        op = self._par_info[idx]["op"]

        if op.grad_method is None:
            return None

        if (self._graph is not None) or use_graph:
            # an empty list to store the 'best' partial derivative method
            # for each observable
            best = []

            # loop over all observables
            for ob in self.observables:
                # get the set of operations betweens the
                # operation and the observable
                S = self.graph.nodes_between(op, ob)

                # If there is no path between them, gradient is zero
                # Otherwise, use finite differences
                best.append("0" if not S else "F")

            if all(k == "0" for k in best):
                return "0"

        return "F"

    def numeric_pd(self, idx, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        a single trainable tape parameter using numerical finite-differences.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Keyword Args:
            h=1e-7 (float): finite difference method step size
            order=1 (int): The order of the finite difference method to use. ``1`` corresponds
                to forward finite differences, ``2`` to centered finite differences.

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
        """
        if params is None:
            params = np.array(self.get_parameters())

        order = options.get("order", 1)
        h = options.get("h", 1e-7)

        shift = np.zeros_like(params)
        shift[idx] = h

        if order == 1:
            # forward finite-difference
            y0 = options.get("y0", None)

            if y0 is None:
                y0 = np.asarray(self.execute_device(params, device))

            y = np.array(self.execute_device(params + shift, device))
            return (y - y0) / h

        if order == 2:
            # central finite difference
            shift_forward = np.array(self.execute_device(params + shift / 2, device))
            shift_backward = np.array(self.execute_device(params - shift / 2, device))
            return (shift_forward - shift_backward) / h

        raise ValueError("Order must be 1 or 2.")

    def device_pd(self, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        a single trainable tape parameter by querying the provided device.

        Args:
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).
        """
        # pylint:disable=unused-argument
        if params is None:
            params = np.array(self.get_parameters())

        current_parameters = self.get_parameters()

        # temporarily mutate the in-place parameters
        self.set_parameters(params)

        # TODO: modify devices that have device Jacobian methods to
        # accept the quantum tape as an argument
        jac = device.jacobian(self)

        # restore original parameters
        self.set_parameters(current_parameters)
        return jac

    def analytic_pd(self, idx, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        a single trainable tape parameter using an analytic method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
        """
        raise NotImplementedError

    def jacobian(self, device, params=None, method="best", **options):
        r"""Compute the Jacobian of the parametrized quantum circuit recorded by the quantum tape.

        The quantum tape can be interpreted as a simple :math:`\mathbb{R}^m \to \mathbb{R}^n` function,
        mapping :math:`m` (trainable) gate parameters to :math:`n` measurement statistics,
        such as expectation values or probabilities.

        By default, the Jacobian will be computed with respect to all parameters on the quantum tape.
        This can be modified by setting the :attr:`~.trainable_params` attribute of the tape.

        The Jacobian can be computed using several methods:

        * Finite differences (``'numeric'``). The first-order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second-order method at :math:`2n` points,
          where ``n = tape.num_params``.

        * Analytic method (``'analytic'``). Analytic, if implemented by the inheriting quantum tape.

        * Best known method for each parameter (``'best'``): uses the analytic method if
          possible, otherwise finite difference.

        * Device method (``'device'``): Delegates the computation of the Jacobian to the
          device executing the circuit. Only supported by devices that provide their
          own method for computing derivatives; support can be checked by
          querying the device capabilities: ``dev.capabilities()['provides_jacobian']`` must
          return ``True``. Examples of supported devices include the experimental
          :class:`"default.tensor.tf" <~.DefaultTensorTF>` device.

        .. note::

            The finite difference method is sensitive to statistical noise in the circuit output,
            since it compares the output at two points infinitesimally close to each other. Hence
            the ``'F'`` method works best with exact expectation values when using simulator
            devices.

        Args:
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Returns:
            array[float]: 2-dimensional array of shape ``(tape.num_params, tape.output_dim)``

        **Example**

        .. code-block:: python

            from pennylane.beta.tapes import QuantumTape
            from pennylane.beta.queuing import expval, var, sample, probs

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                probs(wires=[0, 'a'])

        If parameters are not provided, the existing tape parameters are used:

        >>> dev = qml.device("default.qubit", wires=[0, 'a'])
        >>> tape.jacobian(dev)
        array([[-0.178441  , -0.23358253, -0.05892804],
               [-0.00079144, -0.00103601,  0.05892804],
               [ 0.00079144,  0.00103601,  0.00737611],
               [ 0.178441  ,  0.23358253, -0.00737611]])

        Parameters can be optionally passed during execution:

        >>> tape.jacobian(dev, params=[1.0, 0.0, 1.0])
        array([[-3.24029934e-01, -9.99200722e-09, -3.24029934e-01],
               [-9.67055711e-02, -2.77555756e-09,  3.24029935e-01],
               [ 9.67055709e-02,  3.05311332e-09,  9.67055709e-02],
               [ 3.24029935e-01,  1.08246745e-08, -9.67055711e-02]])

        Parameters provided for execution are temporary, and do not affect
        the tapes' parameters in-place:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Explicitly setting the trainable parameters can significantly reduce
        computational resources, as non-trainable parameters are ignored
        during the computation:

        >>> tape.trainable_params = {0} # set only the first parameter as trainable
        >>> tape.jacobian(dev)
        array([[-0.178441  ],
               [-0.00079144],
               [ 0.00079144],
               [ 0.178441  ]])

        If a tape has no trainable parameters, the Jacobian will be empty:

        >>> tape.trainable_params = {}
        >>> tape.jacobian(dev)
        array([], shape=(4, 0), dtype=float64)
        """
        if method not in ("best", "numeric", "analytic", "device"):
            raise ValueError(f"Unknown gradient method '{method}'")

        if params is None:
            params = self.get_parameters()

        params = np.array(params)

        # check and raise an error if any parameters are non-differentiable
        nondiff_params = {
            idx
            for idx, info in self._par_info.items()
            if info["grad_method"] is None and idx in self.trainable_params
        }

        if nondiff_params:
            raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

        if method == "analytic":
            # If explicitly using analytic mode, ensure that all parameters
            # support analytic differentiation.
            numeric_params = {
                idx
                for idx, info in self._par_info.items()
                if info["grad_method"] == "F" and idx in self.trainable_params
            }

            if numeric_params:
                raise ValueError(
                    f"The analytic gradient method cannot be used with the argument(s) {numeric_params}."
                )

        elif method == "device":
            # Using device mode; query the device for the Jacobian
            return self.device_pd(device, **options)

        if options.get("order", 1) == 1:
            # the value of the circuit at current params, computed only once here
            options["y0"] = np.asarray(self.execute_device(params, device))

        jac = np.zeros((self.output_dim, len(params)), dtype=float)
        p_ind = list(np.ndindex(*params.shape))

        # loop through each parameter and compute the gradient
        for idx, (l, p) in enumerate(zip(p_ind, self.trainable_params)):
            param_method = self._par_info[p]["grad_method"]

            if param_method == "0":
                # independent parameter; skip.
                continue

            if (method == "best" and param_method == "F") or (method == "numeric"):
                # finite difference method
                g = self.numeric_pd(l, device, params=params, **options)

            elif (method == "best" and param_method == "A") or (method == "analytic"):
                # analytic method
                g = self.analytic_pd(l, device, params=params, **options)

            if g.dtype is np.dtype("object"):
                # object arrays cannot be flattened; must hstack them
                g = np.hstack(g)

            try:
                jac[:, idx] = g.flatten()
            except ValueError as e:
                if "could not broadcast input array from shape" in str(e):
                    # the value of self._output_dim, which was estimated during
                    # construction, is incorrect. A device execution is required
                    # to properly infer output dimension.
                    raise ValueError(
                        "The quantum tape could not infer the correct output dimension "
                        "of the quantum computation. Please execute the tape before "
                        "computing the Jacobian."
                    )
                raise e

        return jac
