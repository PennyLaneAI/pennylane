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
"""Code for resource estimation"""
import inspect

import pennylane as qml


def specs(qnode, max_expansion=None):
    """Resource information about a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit.

    Args:
        qnode (.QNode): the QNode to calculate the specifications for

    Keyword Args:
        max_expansion (int): The number of times the internal circuit should be expanded when
            calculating the specification. Defaults to ``qnode.max_expansion``.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a dictionary of information about qnode structure.

    **Example**

    .. code-block:: python3

        x = np.array([0.1, 0.2])

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x, add_ry=True):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=(0,1))
            if add_ry:
                qml.RY(x[1], wires=1)
            return qml.probs(wires=(0,1))

    >>> qml.specs(circuit)(x, add_ry=False)
    {'gate_sizes': defaultdict(int, {1: 1, 2: 1}),
    'gate_types': defaultdict(int, {'RX': 1, 'CNOT': 1}),
    'num_operations': 2,
    'num_observables': 1,
    'num_diagonalizing_gates': 0,
    'num_used_wires': 2,
    'depth': 2,
    'num_device_wires': 2,
    'device_name': 'default.qubit.autograd',
    'diff_method': 'backprop'}

    """

    def specs_qnode(*args, **kwargs):
        """Returns information on the structure and makeup of provided QNode.

        Dictionary keys:
            * ``"num_operations"``
            * ``"num_observables"``
            * ``"num_diagonalizing_gates"``
            * ``"gate_sizes"``: dictionary mapping gate number of wires to number of occurances
            * ``"gate_types"``: dictionary mapping gate types to number of occurances
            * ``"num_used_wires"``: number of wires used by the circuit
            * ``"num_device_wires"``: number of wires in device
            * ``"depth"``: longest path in directed acyclic graph representation
            * ``"dev_short_name"``: name of QNode device
            * ``"diff_method"``

        Potential Additional Information:
            * ``"num_trainable_params"``: number of individual scalars that are trainable
            * ``"num_parameter_shift_executions"``: number of times circuit will execute when
                    calculating the derivative

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain QNode specifications
        """
        initial_max_expansion = qnode.max_expansion
        qnode.max_expansion = max_expansion

        try:
            qnode.construct(args, kwargs)
        finally:
            qnode.max_expansion = initial_max_expansion

        if isinstance(qnode, qml.QNode):
            # TODO: remove when the old QNode is removed
            return qnode.specs

        if qnode.qtape is None:
            raise qml.QuantumFunctionError(
                "The QNode specifications can only be calculated after its quantum tape has been constructed."
            )

        info = qnode.qtape.specs.copy()

        info["num_device_wires"] = qnode.device.num_wires
        info["device_name"] = qnode.device.short_name
        info["diff_method"] = qnode.diff_method
        info["expansion_strategy"] = qnode.expansion_strategy
        info["gradient_options"] = qnode.gradient_kwargs
        info["execution_options"] = qnode.execute_kwargs
        info["interface"] = qnode.interface

        if callable(qnode.gradient_fn):
            info["gradient_fn"] = inspect.getmodule(qnode.gradient_fn).__name__

            try:
                if isinstance(qnode.gradient_fn, qml.gradients.gradient_transform):
                    info["num_gradient_executions"] = len(qnode.gradient_fn(qnode.qtape)[0])
            except Exception as e:
                info["num_gradient_executions"] = "NotSupported"
        else:
            info["gradient_fn"] = qnode.gradient_fn

        return info

    return specs_qnode
