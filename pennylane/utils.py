# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Utilities
=========

**Module name:** :mod:`pennylane.utils`

.. currentmodule:: pennylane.utils

This module contains utilities and auxiliary functions, which are shared
across the PennyLane submodules.

.. raw:: html

    <h3>Summary</h3>

.. autosummary::
    _flatten
    _unflatten
    unflatten
    _inv_dict
    _get_default_args
    expand
    CircuitGraph

.. raw:: html

    <h3>Code details</h3>
"""
from collections import namedtuple
from collections.abc import Iterable
import numbers
import inspect
import itertools

import autograd.numpy as np
import networkx as nx

from pennylane.variable import Variable


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, (numbers.Number, Variable, str)):
        return flat[0], flat[1:]
    elif isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]
    elif isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat
    else:
        raise TypeError("Unsupported type in the model: {}".format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return res


def _inv_dict(d):
    """Reverse a dictionary mapping.

    Returns multimap where the keys are the former values,
    and values are sets of the former keys.

    Args:
        d (dict[a->b]): mapping to reverse

    Returns:
        dict[b->set[a]]: reversed mapping
    """
    ret = {}
    for k, v in d.items():
        ret.setdefault(v, set()).add(k)
    return ret


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (function): a valid Python function

    Returns:
        dict: dictionary containing the argument name and tuple
        (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {
        k: (idx, v.default)
        for idx, (k, v) in enumerate(signature.parameters.items())
        if v.default is not inspect.Parameter.empty
    }


def expand(U, wires, num_wires):
    r"""Expand a multi-qubit operator into a full system operator.

    Args:
        U (array): :math:`2^n \times 2^n` matrix where n = len(wires).
        wires (Sequence[int]): Target subsystems (order matters! the
            left-most Hilbert space is at index 0).

    Returns:
        array: :math:`2^N\times 2^N` matrix. The full system operator.
    """
    if num_wires == 1:
        # total number of wires is 1, simply return the matrix
        return U

    N = num_wires
    wires = np.asarray(wires)

    if np.any(wires < 0) or np.any(wires >= N) or len(set(wires)) != len(wires):
        raise ValueError("Invalid target subsystems provided in 'wires' argument.")

    if U.shape != (2 ** len(wires), 2 ** len(wires)):
        raise ValueError("Matrix parameter must be of size (2**len(wires), 2**len(wires))")

    # generate N qubit basis states via the cartesian product
    tuples = np.array(list(itertools.product([0, 1], repeat=N)))

    # wires not acted on by the operator
    inactive_wires = list(set(range(N)) - set(wires))

    # expand U to act on the entire system
    U = np.kron(U, np.identity(2 ** len(inactive_wires)))

    # move active wires to beginning of the list of wires
    rearranged_wires = np.array(list(wires) + inactive_wires)

    # convert to computational basis
    # i.e., converting the list of basis state bit strings into
    # a list of decimal numbers that correspond to the computational
    # basis state. For example, [0, 1, 0, 1, 1] = 2^3+2^1+2^0 = 11.
    perm = np.ravel_multi_index(tuples[:, rearranged_wires].T, [2] * N)

    # permute U to take into account rearranged wires
    return U[:, perm][perm]


################################################
# Graph functions


Command = namedtuple("Command", ["name", "op", "return_type"])
Layer = namedtuple("Layer", ["op_idx", "param_idx"])


class CircuitGraph:
    """Represents a queue of operations and observables
    as a directed acyclic graph.

    Args:
        queue (list[Operation]): the quantum operations to apply
        observables (list[Observable]): the quantum observables to measure
        parameters (dict[int->list[(int, int)]]): Specifies the free parameters
            of the quantum circuit. The dictionary key is the parameter index.
            The first element of the value tuple is the operation index,
            the second the index of the parameter within the operation.
    """

    def __init__(self, queue, observables, parameters=None):
        self.queue = queue
        self.observables = observables
        self.parameters = parameters or {}

        self._grid = {}
        """dict[int, list[int, Command]]: dictionary representing the quantum circuit
        as a grid. Here, the key is the wire number, and the value is a list
        containing the operation index (that is, where in the queue it occured)
        as well as a Command object containing the operation/observable itself.
        """

        for idx, op in enumerate(queue + observables):
            cmd = Command(name=op.name, op=op, return_type=getattr(op, "return_type", None))

            for w in set(op.wires):
                if w not in self._grid:
                    # wire is not already in the grid;
                    # add the corresponding wire to the grid
                    self._grid[w] = []

                # Add the operation to the grid, to the end of the specified wire
                self._grid[w].append([idx, cmd])

        self._graph = nx.DiGraph()

        # iterate over each wire in the grid
        for _, cmds in self._grid.items():
            if cmds:
                # Add the first operation on the wire to the graph
                # This operation does not depend on any others
                attrs = cmds[0][1]._asdict()
                self._graph.add_node(cmds[0][0], **attrs)

            for i in range(1, len(cmds)):
                # For subsequent operations on the wire:

                if cmds[i][0] not in self._graph:
                    # add them to the graph if they are not already
                    # in the graph (multi-qubit operations might already have been placed)
                    attrs = cmds[i][1]._asdict()
                    self._graph.add_node(cmds[i][0], **attrs)

                # create an edge between this operation and the
                # previous operation
                self._graph.add_edge(cmds[i - 1][0], cmds[i][0])

    @property
    def graph(self):
        """The graph representation of the quantum program.

        The resulting graph has nodes representing quantum operations,
        and edges representing dependent/successor operations.

        Each node is labelled by an integer corresponding to the position
        in the queue; node attributes are used to store information about the node:

        * ``'name'`` *(str)*: name of the quantum operation (e.g., ``'PauliZ'``)

        * ``'op'`` *(Operation or Observable)*: the quantum operation/observable object

        * ``'return_type'`` *(pennylane.operation.ObservableReturnTypes)*: The observable
          return type. If an operation, the return type is simply ``None``.

        Returns:
            networkx.DiGraph: the directed acyclic graph representing
            the quantum program
        """
        return self._graph

    def get_op_indices(self, wire):
        """The operation indices on the given wire.

        Args:
            wire (int): the wire to examine

        Returns:
            list (int): all operation indices on the wire,
            in temporal order
        """
        return list(zip(*self._grid[wire]))[0]

    def ancestors(self, ops):
        """Returns all ancestor operations of a given set of operations.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            set[int]: integer position of all operations
            in the queue that are ancestors of the given operations
        """
        ancestors = set()

        for o in ops:
            subG = self.graph.subgraph(nx.dag.ancestors(self.graph, o))
            ancestors |= set(subG.nodes())

        return ancestors - set(ops)

    def descendants(self, ops):
        """Returns all descendant operations of a given set of operations.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            set[int]: integer position of all operations
            in the queue that are descendants of the given operations
        """
        descendants = set()

        for o in ops:
            subG = self.graph.subgraph(nx.dag.descendants(self.graph, o))
            descendants |= set(subG.nodes())

        return descendants - set(ops)

    def get_ops(self, ops):
        """Given a set of operation indices, return the operation objects.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            List[Operation, Observable]: operations or observables
            corresponding to given integer positions in the queue
        """
        return [self.graph.nodes(data="op")[i] for i in ops]

    def get_names(self, ops):
        """Given a set of operation indices, return the operation names.

        Args:
            ops (Iterable[int]): a given set of operations labelled by integer
                position in the queue

        Returns:
            List[str]: operations or observables
            corresponding to given integer positions in the queue
        """
        return [self.graph.nodes(data="name")[i] for i in ops]

    def layers(self):
        """Identifies and returns a metadata list describing the
        layer structure of the circuit.

        Each layer is a named tuple containing:

        * ``op_idx`` *(list[int])*: the list of operation indices in the layer

        * ``param_idx`` *(list[int])*: the list of parameter indices used within the layer

        Returns:
            list[Layer]: a list of layers
        """
        # keep track of the layer number
        layer = 0
        layer_ops = {0: ([], [])}

        variable_ops_sorted = sorted(list(self.parameters.items()), key=lambda x: x[1][0][0])

        for param_idx, gate_param_tuple in variable_ops_sorted:
            # iterate over all parameters
            for op_idx, _ in gate_param_tuple:
                # get all dependents of the existing parameter
                sub = set(nx.dag.topological_sort(self.graph.subgraph(nx.dag.ancestors(self.graph, op_idx)).copy()))

                # check if any of the dependents are in the
                # existing layer
                if set(layer_ops[layer][0]) & sub:
                    # operation depends on previous layer,
                    # start a new layer count
                    layer += 1

                # store the parameters and ops indices for the layer
                layer_ops.setdefault(layer, ([], []))
                layer_ops[layer][0].append(op_idx)
                layer_ops[layer][1].append(param_idx)

        return [Layer(*k) for _, k in sorted(list(layer_ops.items()))]

    def iterate_layers(self):
        """Identifies and returns an iterable containing
        the parametrized layers.

        Returns:
            Iterable[tuple[list, list, tuple, list]]: an iterable that returns a tuple
            ``(pre_queue, layer, param_idx, post_queue)`` at each iteration.

            * ``pre_queue`` (*list[Operation]*): all operations that precede the layer

            * ``layer`` (*list[Operation]*): the parametrized gates in the layer

            * ``param_idx`` (*tuple[int]*): The integer indices corresponding
              to the free parameters of this layer, in the order they appear in
              this layer.

            * ``post_queue`` (*list[Operation, Observable]*): all operations that succeed the layer
        """
        # iterate through each layer
        for ops, param_idx in self.layers():

            # get the ops in this layer
            layer = self.get_ops(ops)
            pre_queue = self.get_ops(self.ancestors(ops))
            post_queue = self.get_ops(self.descendants(ops))

            yield pre_queue, layer, tuple(param_idx), post_queue
