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
"""Context manager for applying custom decompositions."""

import contextlib
import types

import pennylane as qml
from pennylane.transforms.qfunc_transforms import NonQueuingTape


NonQueuingTape = type("NonQueuingTape", (NonQueuingTape, qml.tape.QuantumTape), {})


@contextlib.contextmanager
def custom_decomposition(decomp_dict, dev=None):
    """A context manager to enable custom decompositions to be applied to Operations.

    Args:
        decomp_dict (Dict[Union(str, .Operator), Callable]): a dictionary detailing the
            pairings of operations and their custom decompositions. The functions
            must have the signature fn(*params, wires).
        dev (Device): an optional device on which to apply the decomposition.

    **Example**

    Suppose we would like to swap out decompositions of the Hadamard and
    CNOT. We must define new decomposition functions with the signature
    ``decomp(params, wires)``.

    .. code-block:: python3

        def custom_cnot(params, wires):
            return [
                qml.Hadamard(wires=wires[1],
                qml.CZ(wires=[wires[0], wires[1]])
                qml.Hadamard(wires=wires[1]
            ]

        def custom_hadamard(params, wires):
            return [qml.RZ(np.pi, wires=wires), qml.RY(np.pi / 2, wires=wires)]

        decomp_dict = {"CNOT" : custom_cnot, "Hadamard" : custom_hadamard}

    We can apply these decompositions during the execution of a QNode by passing
    them to the ``custom_decomp`` context manager:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.beta.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RX(x, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with custom_decomposition(decomp_dict, dev):
            print(qml.draw(circuit, expansion_strategy="device")(0.5))
    """
    try:
        with contextlib.ExitStack() as stack:
            for obj, fn in decomp_dict.items():
                # We enter a new context each decomposition the user passes
                stack.enter_context(_custom_decomposition(obj, fn, dev=dev))

            stack = stack.pop_all()

        yield

    finally:
        stack.close()


@contextlib.contextmanager
def _custom_decomposition(obj, fn, dev=None):
    # We might be providing a decomposition for an Operation that is allowed
    # by a given device, so we need to trick the device into thinking it doesn't
    # support it so it will apply our decomposition.
    obj_supported_by_device = False

    # Covers the case where the user passes a string to indicate the Operator
    if isinstance(obj, str):
        obj = getattr(qml, obj)

    original_decomp_method = obj.expand

    # This is the method that will override the current .decomposition method of
    # the given operation; it has the same signature as .decomposition and takes
    # only self, but then passes its own parameters and wires to the custom
    # decomposition function.
    def instance_method(self):
        with NonQueuingTape() as tape:
            fn(*self.parameters, self.wires)

        return tape

    if isinstance(obj, qml.operation.Operation):
        # Provided object is an instance, not a class.
        # Need to convert `fn` into an instance method.
        instance_method = types.MethodType(instance_method, obj)

    try:
        # Actually set the new decomp function; note that we set
        # expand rather than `decomposition` proper.
        obj.expand = instance_method

        if dev is not None and obj.__name__ in dev.operations:
            obj_supported_by_device = True
            dev.operations.remove(obj.__name__)

        yield

    finally:
        obj.expand = original_decomp_method

        if dev is not None and obj_supported_by_device:
            dev.operations.add(obj.__name__)
