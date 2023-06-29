import torch

import torch.utils._pytree as pytree


from ..executor import Executor
from ..gradient_layers import DerivativeExecutor


def pytreeify(cls):
    """Pytrees refer to a tree-like structure built out of container-like Python objects. The pytreeify class is used
    to bypass some PyTorch limitation of `autograd.Function`. The forward pass can only return tuple of tensors but
    not any other nested structure. This class apply flatten to the forward pass and unflatten the results in the
    apply function. In this way, it is possible to treat multiple tapes with multiple measurements.
    """
    orig_fw = cls.forward
    orig_bw = cls.backward
    orig_apply = cls.apply

    def new_apply(*inp):
        # Inputs already flat
        out_struct_holder = []
        flat_out = orig_apply(out_struct_holder, *inp)
        return pytree.tree_unflatten(flat_out, out_struct_holder[0])

    def new_forward(ctx, out_struct_holder, *inp):
        out = orig_fw(ctx, *inp)
        flat_out, out_struct = pytree.tree_flatten(out)
        ctx._out_struct = out_struct
        out_struct_holder.append(out_struct)
        return tuple(flat_out)

    def new_backward(ctx, *flat_grad_outputs):
        grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
        grad_inputs = orig_bw(ctx, *grad_outputs)
        # None corresponds to the diff of out_struct_holder
        return (None,) + tuple(grad_inputs)

    cls.apply = new_apply
    cls.forward = new_forward
    cls.backward = new_backward
    return cls


@pytreeify
class TorchBoundary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, tapes, next_executor, vjp_function, *parameters
    ):  # pylint: disable=arguments-differ
        """Implements the forward pass batch tape evaluation."""
        ctx.tapes = tapes
        ctx.next_executor = next_executor
        ctx.vjp_function = vjp_function

        return ctx.next_executor(ctx.tapes)

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        vjps = ctx.vjp_function(ctx.tapes, dy, reduction="extend")

        # Remove empty vjps (from tape with non trainable params)
        vjps = tuple(vjp for vjp in vjps if list(vjp.shape) != [0])
        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None, None, None) + tuple(vjps)


class TorchLayer(Executor):
    def __init__(self, next_executor, derivative_executor: DerivativeExecutor):
        self._next_executor = next_executor
        self._derivative_executor = derivative_executor

    def __call__(self, circuits):
        parameters = []
        for tape in circuits:
            parameters.extend(tape.get_parameters())

        return TorchBoundary.apply(
            circuits,
            self._next_executor,
            self._derivative_executor.execute_and_compute_vjp,
            *parameters
        )

    @property
    def next_layer(self):
        return self._next_executor

    @property
    def configuration(self):
        return (self._next_executor, self._derivative_executor)
