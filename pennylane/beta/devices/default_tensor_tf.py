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
r"""
Experimental simulator plugin based on tensor network contractions,
using the TensorFlow backend for Jacobian computations.
"""
import copy


try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        raise ImportError("default.tensor.tf device requires TensorFlow>=2.0")

except ImportError as e:
    raise ImportError("default.tensor.tf device requires TensorFlow>=2.0") from e

from pennylane.beta.devices.default_tensor import DefaultTensor
from pennylane.devices import tf_ops as ops

# tolerance for numerical errors
tolerance = 1e-10


class DefaultTensorTF(DefaultTensor):
    """Experimental TensorFlow Tensor Network simulator device for PennyLane.

    **Short name:** ``default.tensor.tf``

    This experimental device extends ``default.tensor`` by making use of
    the TensorFlow backend of TensorNetwork. As a result, it supports
    classical backpropagation as a means to compute the Jacobian. This can
    be faster than the parameter-shift rule for analytic quantum gradients
    when the number of parameters to be optimized is large.

    To use this device, you will need to install TensorFlow and TensorNetwork:

    .. code-block:: bash

        pip install tensornetwork>=0.2 tensorflow>=2.0

    **Example**

    The ``default.tensor.tf`` device supports end-to-end classical backpropagation with the TensorFlow interface.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your TensorFlow computation:

    >>> dev = qml.device("default.tensor.tf", wires=1)
    >>> @qml.qnode(dev, interface="tf", diff_method="backprop")
    >>> def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> vars = tf.Variable([0.2, 0.5, 0.1])
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(vars)
    >>> tape.gradient(res, vars)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-2.2526717e-01, -1.0086454e+00,  1.3877788e-17], dtype=float32)>

    In this mode, you must use the ``"tf"`` interface, as TensorFlow
    is used as the device backend.

    Args:
        wires (int): number of subsystems in the quantum state represented by the device
        shots (None, int): Number of circuit evaluations/random samples to return when sampling from the device.
            Defaults to ``None`` if not specified, which means that the device returns analytical results.
        representation (str): Underlying representation used for the tensor network simulation.
            Valid options are "exact" (no approximations made) or "mps" (simulated quantum
            state is approximated as a Matrix Product State).
        contraction_method (str): Method used to perform tensor network contractions. Only applicable
            for the "exact" representation. Valid options are "auto", "greedy", "branch", or "optimal".
            See documentation of the `TensorNetwork library <https://tensornetwork.readthedocs.io/en/latest/>`_
            for more information about contraction methods.
    """

    # pylint: disable=too-many-instance-attributes
    name = "PennyLane TensorNetwork (TensorFlow) simulator plugin"
    short_name = "default.tensor.tf"

    _operation_map = copy.copy(DefaultTensor._operation_map)
    _operation_map.update(
        {
            "PhaseShift": lambda phi: tf.linalg.diag(ops.PhaseShift(phi)),
            "RX": ops.RX,
            "RY": ops.RY,
            "RZ": lambda theta: tf.linalg.diag(ops.RZ(theta)),
            "Rot": ops.Rot,
            "CRX": ops.CRX,
            "CRY": ops.CRY,
            "CRZ": lambda theta: tf.linalg.diag(ops.CRZ(theta)),
            "CRot": ops.CRot,
        }
    )

    backend = "tensorflow"
    _reshape = staticmethod(tf.reshape)
    _array = staticmethod(tf.constant)
    _asarray = staticmethod(tf.convert_to_tensor)
    _real = staticmethod(tf.math.real)
    _imag = staticmethod(tf.math.imag)
    _abs = staticmethod(tf.abs)
    _squeeze = staticmethod(tf.squeeze)
    _expand_dims = staticmethod(tf.expand_dims)

    C_DTYPE = ops.C_DTYPE
    R_DTYPE = ops.R_DTYPE

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="tf",
        )
        return capabilities
