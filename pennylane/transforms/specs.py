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

def specs(qnode):
    """

    Args:
        qnode (qml.QNode)

    Returns:
        A function that has the same argument signature as ``qnode``. This function 
        returns a dictionary of information about qnode structure

    **Example**

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x, add_ry=True):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=(0,1))
            if add_ry:
                qml.RY(x[1], wires=1)
            return qml.probs(wires=(0,1))

    >>> x = np.array([0.1, 0.2])
    >>> info = qml.specs(circuit)(x, add_ry=False)
    """
    def specs_qnode(*args, **kwargs):
        """Returns information on the structure and makeup of provided QNode.

        Dictionary keys:
            * ``"total_operations"``: total gates in circuit
            * ``"by_size"``: dictionary mapping gate number of wires to number of occurances
            * ``"by_name"``: dictionary mapping gate types to number of occurances 
            * ``"num_tape_wires"``: number of wires used by the circuit
            * ``"num_wires"``: number of wires in device
            * ``"num_trainable_params"``: number of individual scalars that are trainable
            * ``"depth"``: longest path in directed acyclic graph representation
            * ``"dev_short_name"``: name of QNode device
        
        Potential Additional Information:
            *``"num_parameter_shift_executions"``: number of times circuit will execute when 
                    calculating the derivative
    

        Returns:
            dict: information about qnode structure

        **Example**

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=2)
            @qml.qnode(dev)
            def circuit(x, add_ry=True):
                qml.RX(x[0], wires=0)
                qml.CNOT(wires=(0,1))
                if add_ry:
                    qml.RY(x[1], wires=1)
                return qml.probs(wires=(0,1))

        >>> x = np.array([0.1, 0.2])
        >>> info = qml.specs(circuit)(x, add_ry=False)
        """
        qnode.construct(args, kwargs)

        return qnode.specs
    return specs_qnode