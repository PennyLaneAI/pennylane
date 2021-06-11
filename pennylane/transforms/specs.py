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


def specs(qnode, max_expansion=None):
    """

    Args:
        qnode (.QNode): the QNode to calculate the specifications for

    Keyword Args:
        max_expansion (int): The number of times the internal circuit should be expanded when
            calculating the specification. Defaults to ``qnode.max_expansion``.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a dictionary of information about qnode structure

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

    >>> info = qml.specs(circuit)(x, add_ry=False)
    {'by_size': defaultdict(int, {1: 1, 2: 1}),
    'by_name': defaultdict(int, {'RX': 1, 'CNOT': 1}),
    'total_operations': 2,
    'total_observables': 1,
    'num_tape_wires': 2,
    'depth': 2,
    'num_device_wires': 2,
    'device_name': 'default.qubit.autograd',
    'diff_method': 'backprop'}

    """

    def specs_qnode(*args, **kwargs):
        """Returns information on the structure and makeup of provided QNode.

        Dictionary keys:
            * ``"total_operations"``
            * ``"total_observables"``
            * ``"by_size"``: dictionary mapping gate number of wires to number of occurances
            * ``"by_name"``: dictionary mapping gate types to number of occurances
            * ``"num_tape_wires"``: number of wires used by the circuit
            * ``"num_wires"``: number of wires in device
            * ``"depth"``: longest path in directed acyclic graph representation
            * ``"dev_short_name"``: name of QNode device
            * ``"diff_method"``

        Potential Additional Information:
            * ``"num_trainable_params"``: number of individual scalars that are trainable
            * ``"num_parameter_shift_executions"``: number of times circuit will execute when
                    calculating the derivative

        Returns:
            dict: information about qnode structure
        """
        if max_expansion is not None:
            initial_max_expansion = qnode.max_expansion
            qnode.max_expansion = max_expansion

        qnode.construct(args, kwargs)

        if max_expansion is not None:
            qnode.max_expansion = initial_max_expansion

        return qnode.specs

    return specs_qnode
