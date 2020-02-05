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
PassthruQNode class
"""
import pennylane as qml
import pennylane.operation
import pennylane.circuit_graph
from .base import BaseQNode, QuantumFunctionError


"""
Design notes
------------

PassthruQNode requires some changes to the way other PennyLane components work:

1. :class:`Operator` must not do domain checking for its parameters, or it must let the ADT pass the check.
2. The simulator device must return the result as the ADT instead of plain Python/NumPy types.
3. Any output_conversion in :meth:`BaseQNode.evaluate` must be skipped.

Additionally, any array-like ADT needs to be able to handle (1) scalar multiplication,
(2) indexing/slicing, and possibly (3) iteration, as these are the things qfuncs expect of
array-like parameters.

PassthruQNode does not have a Jacobian method, so it does not HAVE to use VariableRefs or scalar linear indexing of input parameters.
Two options:
1. Use VariableRefs anyway, re-use most BaseQNode methods.
   Problem: after evaluating the VariableRefs, stacking sliced/indexed Tensors in Operation.parameters should somehow result in a Tensor, not an object array.
2. Do not use VariableRefs, call the qfunc each time :meth:`PassthruQNode.evaluate` is called (always mutable).
   Problem: tensornet_tf requires variable_deps?

TODO rethink output_conversion? should require device to return things in a fixed form, but either as arrays or as AD Tensors, do conversion in interface (if necessary...)
"""

class PassthruQNode(BaseQNode):
    """Differentiable quantum node that appears as a white box to an external autodiff framework.

    In PennyLane, the QNode classes work as black box functions with respect to any
    autodiff (AD) framework (such as TensorFlow or PyTorch). This means that the QNode
    converts all its inputs (which may come in data types specific to the
    AD framework used, which we denote *ADT* here) into plain Python/NumPy types,
    computes the required :ref:`quantum function <intro_vcirc_qfunc>` value or Jacobian,
    and converts the result back into the ADT if necessary.

    In contrast, PassthruQNode works as a white box: it preserves the ADT
    throughout the computation. This requires that the quantum function is computed
    using a simulator device that is compatible with the AD framework used (typically
    implemented using that same framework), and returns the result as the ADT instead
    of plain Python/NumPy types.

    The advantages of this approach are that the qfunc can be differentiated using its AD framework
    without requiring a separate method for computing the Jacobian, and that the internals
    of the simulation are visible in the computational graph.

    Args:
        func (callable): The *quantum function* of the QNode.
            A Python function containing :class:`~.operation.Operation` constructor calls,
            and returning a tuple of measured :class:`~.operation.Observable` instances.
        device (~pennylane._device.Device): computational device to execute the function on
        properties (dict[str, Any] or None): additional keyword properties for adjusting the QNode behavior
    """
    def __init__(self, func, device, properties=None):
        # make the device return the result in its native type
        properties = properties or {}
        properties.setdefault('use_native_type', True)
        super().__init__(func, device, mutable=True, properties=properties)

    def __repr__(self):
        """String representation."""
        detail = "<PassthruQNode: device='{}', func={}, wires={}>"
        return detail.format(self.device.short_name, self.func.__name__, self.num_wires)

    def _set_variables(self, args, kwargs):
        # do nothing, since we do not use VariableRefs
        pass

    def _construct(self, args, kwargs):
        """Construct the quantum circuit graph by calling the quantum function.

        Like :class:`.BaseQNode._construct`, but does not use VariableRefs.
        """
        # temporary queues for operations and observables
        self.queue = []  #: list[Operation]: applied operations
        self.obs_queue = []  #: list[Observable]: applied observables

        # set up the context for Operator entry
        if qml._current_context is None:
            qml._current_context = self
        else:
            raise QuantumFunctionError(
                "qml._current_context must not be modified outside this method."
            )
        try:
            # turn off domain checking since PassthruQNode qfuncs can take any class as input
            pennylane.operation.Operator.do_check_domain = False
            # generate the program queue by executing the quantum circuit function
            res = self.func(*args, **kwargs)
        finally:
            qml._current_context = None
            pennylane.operation.Operator.do_check_domain = True

        # check the validity of the circuit
        self._check_circuit(res)
        del self.queue
        del self.obs_queue

        # no output conversion
        self.output_conversion = lambda x: x

        # no VariableRefs, self.variable_deps is empty!
        # generate the DAG
        self.circuit = pennylane.circuit_graph.CircuitGraph(self.ops, self.variable_deps)

        # check for operations that cannot affect the output
        if self.properties.get("vis_check", False):
            self.check_visibility()
