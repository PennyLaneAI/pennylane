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
    raise ImportError("default.tensor.tf device requires TensorFlow>=2.0")

from pennylane.variable import Variable
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

    The ``default.tensor.tf`` device supports various differentiation modes.

    * *End-to-end classical backpropagation with the TensorFlow interface*.
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

    * *Device differentiation*. Using this method, the created QNode
      is a 'black-box' to your classical computation. PennyLane will automatically
      accept classical tensors from any supported interface, and query the
      device directly for the quantum gradient when required.

      >>> dev = qml.device("default.tensor.tf", wires=1)
      >>> @qml.qnode(dev, interface="autograd", diff_method="device")
      >>> def circuit(x):
      ...     qml.RX(x[1], wires=0)
      ...     qml.Rot(x[0], x[1], x[2], wires=0)
      ...     return qml.expval(qml.PauliZ(0))
      >>> grad_fn = qml.grad(circuit, argnum=[0])
      >>> print(grad_fn([0.2, 0.5, 0.1]))
      ([array(-0.22526717), array(-1.00864546), array(6.9388939e-18)],)

      In this mode, even though TensorFlow is used as the device backend, it
      is independent of the chosen QNode interface. In the example above, we combine
      ``default.tensor.tf`` with the ``autograd`` interface.
      It can also be used with the ``torch`` and the ``tf`` interface.

    In addition to end-to-end classical backpropagation and device differentiation,
    the ``default.tensor.tf`` device also supports ``parameter-shift`` and
    ``finite-diff`` differentiation methods.

    Args:
        wires (int): number of subsystems in the quantum state represented by the device
        shots (int): Number of circuit evaluations/random samples to return when sampling from the device.
            Defaults to 1000 if not specified.
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

    def __init__(self, wires, shots=1000, representation="exact", contraction_method="auto"):
        self.variables = []
        """List[tf.Variable]: Free parameters, cast to TensorFlow variables,
        for this circuit."""

        self.res = None
        """tf.tensor[tf.float64]: result from the last circuit execution"""

        self.op_params = {}
        """dict[Operation, List[Any, tf.Variable]]: A mapping from each operation
        in the queue, to the corresponding list of parameter values. These
        values can be Python numeric types, NumPy arrays, or TensorFlow variables."""

        self.tape = None
        """tf.GradientTape: the gradient tape under which all tensor network
        modifications must be made"""

        super().__init__(
            wires,
            shots=shots,
            representation=representation,
            contraction_method=contraction_method,
        )

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            provides_jacobian=True,
            passthru_interface="tf",
        )
        return capabilities

    def reset(self):
        self.res = None
        self.variables = []
        super().reset()

    def execution_context(self):
        self.tape = tf.GradientTape(persistent=True)
        return self.tape

    def pre_apply(self):
        super().pre_apply()

        self.op_params = {}

        for operation in self.op_queue:
            # Copy the operation parameters to the op_params dictionary.
            # Note that these are the unwrapped parameters, so PennyLane
            # free parameters will be represented as Variable instances.
            self.op_params[operation] = operation.data[:]

        # Loop through the free parameter reference dictionary
        for _, par_dep_list in self.parameters.items():
            if not par_dep_list:
                # parameter is not used within circuit
                v = tf.Variable(0, dtype=self.R_DTYPE)
                self.variables.append(v)
                continue

            # get the first parameter dependency for each free parameter
            first = par_dep_list[0]

            # For the above parameter dependency, get the corresponding
            # operation parameter variable, and get the numeric value.
            # Convert the resulting value to a TensorFlow tensor.
            val = first.op.data[first.par_idx].val
            mult = first.op.data[first.par_idx].mult
            v = tf.Variable(val / mult, dtype=self.R_DTYPE)

            # Mark the variable to be watched by the gradient tape,
            # and append it to the variable list.
            self.variables.append(v)

            for p in par_dep_list:
                # Replace the existing Variable free parameter in the op_params dictionary
                # with the corresponding tf.Variable parameter.
                # Note that the free parameter might be scaled by the
                # variable.mult scaling factor.
                mult = p.op.data[p.par_idx].mult
                self.op_params[p.op][p.par_idx] = v * mult

        # check that no Variables remain in the op_params dictionary
        values = [item for sublist in self.op_params.values() for item in sublist]
        assert not any(
            isinstance(v, Variable) for v in values
        ), "A pennylane.Variable instance was not correctly converted to a tf.Variable"

        # flatten the variables list in case of nesting
        self.variables = tf.nest.flatten(self.variables)
        self.tape.watch(self.variables)

        for operation in self.op_queue:
            # Apply each operation, but instead of passing operation.parameters
            # (which contains the evaluated numeric parameter values),
            # pass op_params[operation], which contains numeric values
            # for fixed parameters, and tf.Variable objects for free parameters.
            super().apply(operation.name, operation.wires, self.op_params[operation])

    def apply(self, operation, wires, par):
        # individual operations are already applied inside self.pre_apply()
        pass

    def execute(self, queue, observables, parameters=None, **kwargs):
        # pylint: disable=bad-super-call
        results = super(DefaultTensor, self).execute(queue, observables, parameters=parameters)

        with self.tape:
            # convert the results list into a single tensor
            self.res = tf.stack(results)

        if kwargs.get("return_native_type", False):
            return self.res
        # return the results as a NumPy array
        return self.res.numpy()

    def jacobian(self, queue, observables, parameters):
        """Calculates the Jacobian of the device circuit using TensorFlow
        backpropagation.

        Args:
            queue (list[Operation]): operations to be applied to the device
            observables (list[Observable]): observables to be measured
            parameters (dict[int, ParameterDependency]): reference dictionary
                mapping free parameter values to the operations that
                depend on them

        Returns:
            array[float]: Jacobian matrix of size (``num_params``, ``num_wires``)
        """
        self.reset()
        self.execute(queue, observables, parameters=parameters)
        jac = self.tape.jacobian(self.res, self.variables, experimental_use_pfor=False)
        # TODO use unconnected_gradients=tf.UnconnectedGradients.ZERO instead of the following?
        jac = [i if i is not None else tf.zeros(self.res.shape, dtype=tf.float64) for i in jac]
        jac = tf.stack(jac)
        return jac.numpy().T
