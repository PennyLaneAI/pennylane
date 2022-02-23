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
This module contains the mixin interface class for creating differentiable quantum tapes with
PyTorch.
"""
# pylint: disable=protected-access, attribute-defined-outside-init, arguments-differ, no-member, import-self, too-many-statements
import numpy as np
import semantic_version
import torch

import pennylane as qml
from pennylane.queuing import AnnotatedQueue

COMPLEX_SUPPORT = semantic_version.match(">=1.8.0", torch.__version__)


def args_to_numpy(args):
    """Converts all Torch tensors in a list to NumPy arrays

    Args:
        args (list): list containing QNode arguments, including Torch tensors

    Returns:
        list: returns the same list, with all Torch tensors converted to NumPy arrays
    """
    res = []

    for i in args:
        if isinstance(i, torch.Tensor):
            if i.is_cuda:  # pragma: no cover
                res.append(i.cpu().detach().numpy())
            else:
                res.append(i.detach().numpy())
        else:
            res.append(i)

    # if NumPy array is scalar, convert to a Python float
    res = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in res]

    return res


class _TorchInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_kwargs, *input_):
        """Implements the forward pass QNode evaluation"""
        # detach all input tensors, convert to NumPy array
        ctx.args = args_to_numpy(input_)
        ctx.kwargs = input_kwargs
        ctx.save_for_backward(*input_)

        tape = ctx.kwargs["tape"]
        device = ctx.kwargs["device"]

        # unwrap constant parameters
        ctx.all_params = tape.get_parameters(trainable_only=False)
        ctx.all_params_unwrapped = args_to_numpy(ctx.all_params)

        # evaluate the tape
        tape.set_parameters(ctx.all_params_unwrapped, trainable_only=False)
        res = tape.execute_device(ctx.args, device)
        tape.set_parameters(ctx.all_params, trainable_only=False)

        if hasattr(res, "numpy"):
            res = qml.math.to_numpy(res)

        use_adjoint_cached_state = False
        # tape might not be a jacobian tape
        jac_options = getattr(tape, "jacobian_options", {})
        # cache state for adjoint jacobian computation
        if jac_options.get("jacobian_method", None) == "adjoint_jacobian":
            if jac_options.get("adjoint_cache", True):
                use_adjoint_cached_state = True
                state = device._pre_rotated_state

        ctx.saved_grad_matrices = {}

        def _evaluate_grad_matrix(grad_matrix_fn):
            """Convenience function for generating gradient matrices
            for the given parameter values.

            This function serves two purposes:

            * Avoids duplicating logic surrounding parameter unwrapping/wrapping

            * Takes advantage of closure, to cache computed gradient matrices via
              the ctx.saved_grad_matrices attribute, to avoid gradient matrices being
              computed multiple redundant times.

              This is particularly useful when differentiating vector-valued QNodes.
              Because PyTorch requests the vector-GradMatrix product,
              and *not* the full GradMatrix, differentiating vector-valued
              functions will result in multiple backward passes.

            Args:
                grad_matrix_fn (str): Name of the gradient matrix function. Should correspond to an existing
                    tape method. Currently allowed values include ``"jacobian"`` and ``"hessian"``.

                Returns:
                    array[float]: the gradient matrix
            """
            if grad_matrix_fn in ctx.saved_grad_matrices:
                return ctx.saved_grad_matrices[grad_matrix_fn]

            if use_adjoint_cached_state:
                tape.jacobian_options["device_pd_options"] = {"starting_state": state}

            tape.set_parameters(ctx.all_params_unwrapped, trainable_only=False)
            grad_matrix = getattr(tape, grad_matrix_fn)(
                device, params=ctx.args, **tape.jacobian_options
            )
            tape.set_parameters(ctx.all_params, trainable_only=False)

            grad_matrix = torch.as_tensor(torch.from_numpy(grad_matrix), dtype=tape.dtype)
            ctx.saved_grad_matrices[grad_matrix_fn] = grad_matrix

            return grad_matrix

        class _Jacobian(torch.autograd.Function):
            @staticmethod
            def forward(ctx_, parent_ctx, *input_):
                """Implements the forward pass QNode Jacobian evaluation"""
                ctx_.dy = parent_ctx.dy
                ctx_.save_for_backward(*input_)
                jacobian = _evaluate_grad_matrix("jacobian")
                return jacobian

            @staticmethod
            def backward(ctx_, ddy):  # pragma: no cover
                """Implements the backward pass QNode vector-Hessian product"""
                hessian = _evaluate_grad_matrix("hessian")

                if torch.squeeze(ddy).ndim > 1:
                    vhp = ctx_.dy.view(1, -1) @ ddy @ hessian @ ctx_.dy.view(-1, 1)
                    vhp = vhp / torch.linalg.norm(ctx_.dy) ** 2
                else:
                    vhp = ddy @ hessian

                vhp = torch.unbind(vhp.view(-1))

                grad_input = []

                # match the type and device of the input tensors
                for i, j in zip(vhp, ctx_.saved_tensors):
                    res = torch.as_tensor(i, dtype=tape.dtype)
                    if j.is_cuda:  # pragma: no cover
                        cuda_device = j.get_device()
                        res = torch.as_tensor(res, device=cuda_device)
                    grad_input.append(res)

                return (None,) + tuple(grad_input)

        ctx.jacobian = _Jacobian

        # if any input tensor uses the GPU, the output should as well
        for i in input_:
            if isinstance(i, torch.Tensor):
                if i.is_cuda:  # pragma: no cover
                    cuda_device = i.get_device()
                    return torch.as_tensor(
                        torch.from_numpy(res), device=cuda_device, dtype=tape.dtype
                    )

        if tape.is_sampled and not tape.all_sampled:
            return tuple(torch.as_tensor(t, dtype=tape.dtype) for t in res)

        if res.dtype == np.dtype("object"):
            res = np.hstack(res)

        return torch.as_tensor(torch.from_numpy(res), dtype=tape.dtype)

    @staticmethod
    def backward(ctx, dy):  # pragma: no cover
        """Implements the backwards pass QNode vector-Jacobian product"""
        ctx.dy = dy

        dyv = dy.view(1, -1)
        jac_res = ctx.jacobian.apply(ctx, *ctx.saved_tensors)

        # When using CUDA, dyv seems to remain on the GPU, while the result
        # of jac_res is returned on CPU, even though the saved_tensors arguments are
        # themselves on the GPU. Check whether this has happened, and move things
        # back to the GPU if required.
        if dyv.is_cuda or jac_res.is_cuda:
            if not dyv.is_cuda:
                dyv = torch.as_tensor(dyv, device=jac_res.get_device())
            if not jac_res.is_cuda:
                jac_res = torch.as_tensor(jac_res, device=dyv.get_device())

        vjp = dyv @ jac_res
        vjp = torch.unbind(vjp.view(-1))
        return (None,) + tuple(vjp)


class TorchInterface(AnnotatedQueue):
    """Mixin class for applying an Torch interface to a :class:`~.JacobianTape`.

    Torch-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyTorchQuantumTape(TorchInterface, JacobianTape):

    Alternatively, the Torch interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the Torch interface can be used to perform quantum-classical
    differentiable programming.

    **Example**

    Once a Torch quantum tape has been created, it can be evaluated and differentiated:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

        with TorchInterface.apply(JacobianTape()) as qtape:
            qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * torch.sin(p[2]), wires=0)
            expval(qml.PauliX(0))

        result = qtape.execute(dev)

    >>> print(result)
    tensor([0.0698], dtype=torch.float64, grad_fn=<_TorchInterfaceBackward>)
    >>> result.backward()
    >>> print(p.grad)
    tensor([0.2987, 0.3971, 0.0988])

    The Torch interface defaults to ``torch.float64`` output. This can be modified by
    providing the ``dtype`` argument when applying the interface:

    >>> p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    >>> with TorchInterface.apply(JacobianTape(), dtype=torch.float32) as qtape:
    ...     qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * torch.sin(p[2]), wires=0)
    ...     expval(qml.PauliX(0))
    >>> result = qtape.execute(dev)
    >>> print(result)
    tensor([0.0698], grad_fn=<_TorchInterfaceBackward>)
    >>> print(result.dtype)
    torch.float32
    >>> result.backward()
    >>> print(p.grad)
    tensor([0.2987, 0.3971, 0.0988])
    >>> print(p.grad.dtype)
    torch.float32
    """

    dtype = torch.float64

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "torch"

    def _update_trainable_params(self):
        params = self.get_parameters(trainable_only=False)

        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", False):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        return params

    def _execute(self, params, **kwargs):
        kwargs["tape"] = self
        res = _TorchInterface.apply(kwargs, *params)
        return res

    @classmethod
    def apply(cls, tape, dtype=torch.float64):
        """Apply the Torch interface to an existing tape in-place.

        Args:
            tape (.JacobianTape): a quantum tape to apply the Torch interface to
            dtype (torch.dtype): the dtype that the returned quantum tape should
                output

        **Example**

        >>> with JacobianTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> TorchInterface.apply(tape)
        >>> tape
        <TorchQuantumTape: wires=<Wires = [0]>, params=1>
        """
        if (dtype is torch.complex64 or dtype is torch.complex128) and not COMPLEX_SUPPORT:
            raise qml.QuantumFunctionError(
                "Version 1.8.0 or above of PyTorch must be installed for complex support, "
                "which is required for quantum functions that return the state."
            )

        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("TorchQuantumTape", (cls, tape_class), {"dtype": dtype})
        tape._update_trainable_params()
        return tape
