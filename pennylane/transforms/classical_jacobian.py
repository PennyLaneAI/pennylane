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
"""
Contains the classical Jacobian transform
"""
# pylint: disable=import-outside-toplevel
import pennylane as qml


def classical_jacobian(qnode):
    """Function to extract the Jacobian
    matrix of the classical part of a QNode"""

    def classical_preprocessing(*args, **kwargs):
        """Returns the trainable gate parameters for
        a given QNode input"""
        qnode.construct(args, kwargs)
        return qml.math.stack(qnode.qtape.get_parameters())

    if qnode.interface == "autograd":
        return qml.jacobian(classical_preprocessing)

    if qnode.interface == "torch":
        import torch

        def _jacobian(*args, **kwargs):  # pylint: disable=unused-argument
            return torch.autograd.functional.jacobian(classical_preprocessing, args)

        return _jacobian

    if qnode.interface == "jax":
        import jax

        return jax.jacobian(classical_preprocessing)

    if qnode.interface == "tf":
        import tensorflow as tf

        def _jacobian(*args, **kwargs):
            with tf.GradientTape() as tape:
                tape.watch(args)
                gate_params = classical_preprocessing(*args, **kwargs)

            return tape.jacobian(gate_params, args)

        return _jacobian
