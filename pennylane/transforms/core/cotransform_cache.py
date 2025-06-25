# Copyright 2023 Xanadu Quantum Technologies Inc.

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
fsdf
"""
from functools import partial

import pennylane as qml
from pennylane import math
from pennylane.exceptions import QuantumFunctionError


def _numpy_jac(*_, **__) -> qml.typing.TensorLike:
    raise QuantumFunctionError("No trainable parameters.")


def _autograd_jac(classical_function, argnums, *args, **kwargs) -> qml.typing.TensorLike:
    if not math.get_trainable_indices(args) and argnums is None:
        raise QuantumFunctionError("No trainable parameters.")
    return qml.jacobian(classical_function, argnum=argnums)(*args, **kwargs)


# pylint: disable=import-outside-toplevel, unused-argument
def _tf_jac(classical_function, argnums, *args, **kwargs) -> qml.typing.TensorLike:
    if not math.get_trainable_indices(args):
        raise QuantumFunctionError("No trainable parameters.")
    import tensorflow as tf

    with tf.GradientTape() as tape:
        gate_params = classical_function(*args, **kwargs)
    return tape.jacobian(gate_params, args)


# pylint: disable=import-outside-toplevel, unused-argument
def _torch_jac(classical_function, argnums, *args, **kwargs) -> qml.typing.TensorLike:
    if not math.get_trainable_indices(args):
        raise QuantumFunctionError("No trainable parameters.")
    from torch.autograd.functional import jacobian

    return jacobian(partial(classical_function, **kwargs), args)


# pylint: disable=import-outside-toplevel
def _jax_jac(classical_function, argnums, *args, **kwargs) -> qml.typing.TensorLike:
    import jax

    if argnums is None:
        argnums = 0
    return jax.jacobian(classical_function, argnums=argnums)(*args, **kwargs)


_jac_map = {
    None: _numpy_jac,
    "numpy": _numpy_jac,
    "autograd": _autograd_jac,
    "tf": _tf_jac,
    "torch": _torch_jac,
    "jax": _jax_jac,
    "jax-jit": _jax_jac,
}


# pylint: disable=unused-argument
def _classical_preprocessing(qnode, program, tape_idx: int, *args, argnums=None, **kwargs):
    """Returns the trainable gate parameters for a given QNode input.

    While differentiating this again for each tape in the batch may be less efficient than desireable for large batches,
    it cleanly works with all interfaces.
    """
    tape = qml.workflow.construct_tape(qnode, level=0)(*args, **kwargs)
    tapes, _ = program((tape,))
    return math.stack(tapes[tape_idx].get_parameters(trainable_only=True))


def _jax_argnums_to_tape_trainable(qnode, argnums, program, args, kwargs):
    """This function gets the tape parameters from the QNode construction given some argnums (only for Jax).
    The tape parameters are transformed to JVPTracer if they are from argnums. This function imitates the behaviour
    of Jax in order to mark trainable parameters.

    Args:
        qnode(qml.QNode): the quantum node.
        argnums(int, list[int]): the parameters that we want to set as trainable (on the QNode level).
        program(qml.transforms.core.TransformProgram): the transform program to be applied on the tape.

    Return:
        list[float, jax.JVPTracer]: List of parameters where the trainable one are `JVPTracer`.
    """
    import jax  # pylint: disable=import-outside-toplevel

    tag = jax.core.TraceTag()
    with jax.core.take_current_trace() as parent_trace:
        trace = jax.interpreters.ad.JVPTrace(parent_trace, tag)
        args_jvp = [
            (
                jax.interpreters.ad.JVPTracer(trace, arg, jax.numpy.zeros(arg.shape))
                if i in argnums
                else arg
            )
            for i, arg in enumerate(args)
        ]
        with jax.core.set_current_trace(trace):
            tape = qml.workflow.construct_tape(qnode, level=0)(*args_jvp, **kwargs)
            tapes, _ = program((tape,))

    return tuple(tape.get_parameters(trainable_only=False) for tape in tapes)


def _get_interface(qnode, args, kwargs) -> str:
    if qnode.interface == "auto":
        interface = math.get_interface(*args, *list(kwargs.values()))
        try:
            interface = math.get_canonical_interface_name(interface).value
        except ValueError:
            interface = "numpy"
    else:
        interface = qnode.interface
    return interface


class CotransformCache:

    def __init__(self, qnode, args, kwargs):
        self.qnode = qnode
        self.args = args
        self.kwargs = kwargs

    @property
    def program(self):
        return self.qnode.transform_program

    def _get_idx_for_transform(self, transform):
        for i, t in self.program:
            if t.transform == transform:
                return i
        raise ValueError("Could not find {transform}.")

    def get_classical_jacobian(self, transform, tape_idx: int):
        transform_index = self._get_idx_for_transform(transform)
        if not transform.classical_cotransform:
            return None
        argnums = self.program[-1].kwargs.get("argnums", None)  # pylint: disable=no-member

        interface = _get_interface(self.qnode, self.args, self.kwargs)

        subprogram = self.program[:transform_index]
        f = partial(_classical_preprocessing, self.qnode, subprogram, tape_idx)
        classical_jacobian = _jac_map[interface](f, argnums, *self.args, **self.kwargs)
        return classical_jacobian

    def get_argnums(self, transform):
        transform_index = self._get_idx_for_transform(transform)
        interface = _get_interface(self.qnode, self.args, self.kwargs)
        if interface not in ["jax", "jax-jit"]:
            return None

        if "argnum" in self.program[transform_index].kwargs:
            raise QuantumFunctionError(
                "argnum does not work with the Jax interface. You should use argnums instead."
            )

        transform = self.program[transform_index]
        argnums = self.program[-1].kwargs.get("argnums", None)  # pylint: disable=no-member

        if argnums is None and math.get_interface(self.args[0]) != "jax":
            raise QuantumFunctionError("No trainable parameters.")

        argnums = [0] if argnums is None else argnums
        # pylint: disable=protected-access
        if (transform._use_argnum or transform.classical_cotransform) and argnums:
            subprogram = self.program[:transform_index]
            params = _jax_argnums_to_tape_trainable(
                self.qnode, argnums, subprogram, self.args, self.kwargs
            )
            return [math.get_trainable_indices(param) for param in params]
        return None
