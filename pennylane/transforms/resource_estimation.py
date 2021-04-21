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


def resource_estimation(qnode, expand_depth=1):
    """ Create a function that provides resource estimation for given parameter
    values.

    Args:
        qnode (qml.QNode): a PL Qnode
    
    Returns:
        function: a function of the same parameters as the qnode

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

        estimates = resource_estimation(circuit)
        x = np.array([0.1, 0.2])

    >>> estimate_dict = esimates(x, add_ry=True)

    """

    def estimate(*args, **kwargs):
        """
        info slots:
            dev_short_name
            num_wires
            num_gates
            num_ops_by_size

        Returns:
            dictionary

        """
        info = dict()
        qnode.construct(args, kwargs)
        tape = qnode.qtape
        tape.expand(depth=expand_depth)

        info['dev_short_name'] = qnode.device.short_name
        info['num_wires'] = qnode.device.num_wires

        num_gates = 0
        op_by_size = dict()
        for op in tape.operations:
            op_by_size[op.num_wires]=op_by_size.get(op.num_wires, 0)+1
            num_gates +=1

        info['num_gates'] = num_gates
        info['num_ops_by_size'] = op_by_size
        



        return info

        

    return estimate