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
from .cost_functions import (
    square_kernel_matrix,
    kernel_matrix,
    kernel_polarization,
    kernel_target_alignment,
)


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

        self.probs_qnode = qml.QNode(
            circuit, device, interface=interface, diff_method=diff_method, **kwargs
        )

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

    def kernel_matrix(self, X1, X2, params, **kwargs):
        """Return the kernel matrix for two given sets of datapoints.

        Args:
            X1 (list[datapoint]): List of datapoints (first argument)
            X2 (list[datapoint]): List of datapoints (second argument)
            params (array[float]): Circuit parameters

        Returns:
            array[float]: Kernel matrix for the given datapoints
        """
        return kernel_matrix(
            X1,
            X2,
            lambda x1, x2: self(x1, x2, params, **kwargs),
        )

    def square_kernel_matrix(self, X, params, **kwargs):
        """Return the kernel matrix for a given set of datapoints.

        Args:
            X (list[datapoint]): List of datapoints
            params (array[float]): Circuit parameters

        Returns:
            array[float]: Kernel matrix for the given datapoints
        """
        return square_kernel_matrix(
            X, lambda x1, x2: self(x1, x2, params, **kwargs), assume_normalized_kernel=True
        )

    def polarization(self, X, Y, params, **kwargs):
        """Kernel polarization relative to a given set of labels.

        Args:
            X (list[datapoint]): List of datapoints
            Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
            kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.

        Keyword Args:
            rescale_class_labels (bool, optional): Rescale the class labels during the computation
                of the polarization. This is important to take care of unbalanced datasets.
                Defaults to True.

        Returns:
            float: The kernel polarization.
        """
        return kernel_polarization(
            X,
            Y,
            lambda x1, x2: self(x1, x2, params, **kwargs),
            assume_normalized_kernel=True,
            rescale_class_labels=kwargs.get("rescale_class_labels", True),
        )

    def target_alignment(self, X, Y, params, **kwargs):
        """Kernel target alignment relative to a given set of labels.

        Args:
            X (list[datapoint]): List of datapoints
            Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
            kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.

        Keyword Args:
            rescale_class_labels (bool, optional): Rescale the class labels during the computation
                of the polarization. This is important to take care of unbalanced datasets.
                Defaults to True.

        Returns:
            float: The kernel target alignment.
        """
        return kernel_target_alignment(
            X,
            Y,
            lambda x1, x2: self(x1, x2, params, **kwargs),
            assume_normalized_kernel=True,
            rescale_class_labels=kwargs.get("rescale_class_labels", True),
        )
