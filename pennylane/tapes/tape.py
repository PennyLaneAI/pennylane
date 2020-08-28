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

from collections import deque, Sequence
import numpy as np

import pennylane as qml
from pennylane.beta.queuing.measure import MeasurementProcess
from .queuing import AnnotatedQueue, QueuingContext
from .circuit_graph import CircuitGraph



class QuantumTape(AnnotatedQueue):
    cast = staticmethod(np.array)

    def __init__(self):
        super().__init__()
        self._prep = []
        self._ops = []
        self._obs = []

        self._par_info = {}
        self._trainable_params = set()
        self._graph = None
        self._output_dim = 0

        self.hash = 0
        self.is_sampled = False

    def __enter__(self):
        QueuingContext.append(self)
        return super().__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)
        self._construct()

    def _construct(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes."""
        param_count = 0
        op_count = 0

        for obj, info in self._queue.items():
            if not info:
                self._ops.append(obj)

                for p in range(len(obj.data)):
                    self._par_info[param_count] = {"op": obj, "p_idx": p}
                    param_count += 1

                op_count += 1

            if isinstance(obj, MeasurementProcess):
                # TODO: remove the following line once devices
                # have been refactored to no longer use obs.return_type
                info["owns"].return_type = obj.return_type

                self._obs.append((obj, info["owns"]))
                self._output_dim += 1

                if obj.return_type is qml.operation.Sample:
                    self.is_sampled = True

        self.wires = qml.wires.Wires.all_wires([op.wires for op in self.operations + self.observables])
        self._trainable_params = set(range(param_count))

    @property
    def trainable_params(self):
        """Store or return the indices of parameters that support differentiability.
        Avalue of ``None`` means that all argument indices are differentiable.
        """
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (Set[int]): Parameter indices.
        """
        if any(not isinstance(i, int) or i < 0 for i in param_indices):
            raise ValueError("Argument indices must be positive integers.")

        if any(i > self.num_params for i in param_indices):
            raise ValueError(f"Tape has at most {self.num_params} trainable parameters.")

        self._trainable_params = param_indices

    @property
    def operations(self):
        return self._ops

    @property
    def observables(self):
        return [m[1] for m in self._obs]

    @property
    def num_params(self):
        if self.trainable_params:
            return len(self.trainable_params)

        return len(self._par_info)

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
        if self._graph is None:
            self._graph = CircuitGraph(self.operations, self.observables, self.wires)

        return self._graph

    def get_parameters(self, free_only=True):
        """Return the parameters incident on the tape operations"""
        params = [o.data for o in self._ops]
        params = [item for sublist in params for item in sublist]

        if not free_only:
            return params

        return [p for idx, p in enumerate(params) if idx in self.trainable_params]

    def set_parameters(self, parameters, free_only=True):
        """Set the parameters incident on the tape operations"""
        for idx, p in enumerate(parameters):
            if free_only and idx not in self.trainable_params:
                continue

            op = self._par_info[idx]["op"]
            op.data[self._par_info[idx]["p_idx"]] = p

    def execute(self, device, params=None):
        """Execute the tape on `device` with gate input `params`"""
        if params is None:
            params = self.get_parameters()

        return self.cast(self._execute(params, device=device))

    def execute_device(self, params, device):
        """Execute the tape on `device` with gate input `params`"""
        device.reset()

        self.set_parameters(params)

        if isinstance(device, qml.QubitDevice):
            return device.execute(self)

        return device.execute(self.operations, self.observables, {})

    _execute = execute_device

    # ========================================================
    # gradient methods
    # ========================================================

    def _grad_method(self, idx, use_graph=True):
        """Determine the correct partial derivative computation method for each gate argument.

        .. note::

            The ``QuantumTape`` only supports numerical differentiation, so
            this method will always return either ``"F"`` or ``None``. If an inheriting
            QNode supports analytic differentiation for certain operations, make sure
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

    def igrad_numeric(self, idx, device, params=None, **options):
        """Evaluate the gradient for the ith parameter in params
        using finite differences."""
        if params is None:
            params = np.array(self.get_parameters())

        order = options.get("order", 1)
        h = options.get("h", 1e-7)

        shift = np.zeros_like(params)
        shift[idx] = h

        if order == 1:
            # forward finite-difference
            y0 = options.get("y0", np.asarray(self.execute_device(params, device)))
            y = np.array(self.execute_device(params + shift, device))
            return (y - y0) / h

        if order == 2:
            # central finite difference
            shift_forward = np.array(self.execute_device(params + shift / 2, device))
            shift_backward = np.array(self.execute_device(params - shift / 2, device))
            return (shift_forward - shift_backward) / h

        raise ValueError("Order must be 1 or 2.")

    def jacobian(self, device, params=None, method="best", **options):
        """Compute the Jacobian via the parameter-shift rule
        on `device` with gate input `params`"""
        if params is None:
            params = self.get_parameters()

        params = np.array(params)

        if options.get("order", 1) == 1:
            # the value of the circuit at current params, computed only once here
            options["y0"] = np.asarray(self.execute_device(params, device))

        jac = np.zeros((self._output_dim, len(params)), dtype=float)

        p_ind = list(np.ndindex(*params.shape))

        for idx, l in enumerate(p_ind):
            # loop through each parameter and compute the gradient
            method = self._grad_method(l[0], use_graph=options.get("use_graph", True))

            if method == "F":
                jac[:, idx] = self.igrad_numeric(l, device, params=params, **options)

        return jac
