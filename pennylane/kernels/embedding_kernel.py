# Copyright

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
This file contains functionalities for embedding kernels.
"""
import pennylane as qml

class EmbeddingKernel:
    """
    Args:
        ansatz (callable): The ansatz for the circuit before the final measurement step.
            Note that the ansatz **must** have the following signature:

            .. code-block:: python

                @qml.template
                ansatz(x, params, **kwargs)

            where ``x`` represents the datapoint, ``params`` are the trainable weights of the
            variational circuit, and ``kwargs`` are any additional keyword arguments that need
            to be passed to the template. It is absolutely necessary that the ansatz is a qml.template.
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            cost function should be executed. This can either be a single device, or a list
            of devices of length matching the number of terms in the Hamiltonian.
        interface (str, None): Which interface to use.
            This affects the types of objects that can be passed to/returned to the cost function.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): The method of differentiation to use with the created cost function.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.
    """   
    
    def __init__(
        self,
        ansatz,
        device,
        interface="autograd",
        diff_method="best",
        **kwargs,
    ):
        self.probs_qnode = None
        """QNode: The QNode representing the quantum embedding kernel."""

        def circuit(x1, x2, params, **kwargs):
            ansatz(x1, params, **kwargs)
            qml.inv(ansatz(x2, params, **kwargs))

            return qml.probs(wires=device.wires)

        self.probs_qnode = qml.QNode(circuit, device, interface=interface, diff_method=diff_method, **kwargs)

    def __call__(self, x1, x2, params, **kwargs):
        """
        Evaluate the embedding kernel between two datapoints at a
        specific point in parameter space.

        Args:
            x1 (array): The first datapoint
            x2 (array): The second datapoint
            params (array): The variational parameters of the circuit

        Returns:
            float: Overlap of the embedded states, the value lies in the interval [0, 1]
        """

        return self.probs_qnode(x1, x2, params, **kwargs)[0]