from ..executor import Executor


class NullLayer(Executor):
    def __init__(self, next_executor: Executor, *_, **__):
        self._next_executor = next_executor

    def __call__(self, circuits):
        return self._next_executor(circuits)

    def __repr__(self):
        return f"NullExecutor(\n\t{self._next_executor}\n)"


def _get_null_boundary():
    return NullLayer


def _get_torch_boundary():
    from .torch_boundary import TorchLayer

    return TorchLayer


def _get_jax_boundary():
    from .jax_boundary import JaxLayer

    return JaxLayer


def _get_autograd_boundary():
    from .autograd_boundary import AutogradLayer

    return AutogradLayer


def _get_tf_boundary():
    from .tf_boundary import TFLayer

    return TFLayer


_get_boundary_map = {
    None: _get_null_boundary,
    "numpy": _get_null_boundary,
    "torch": _get_torch_boundary,
    "jax": _get_jax_boundary,
    "autograd": _get_autograd_boundary,
    "tf": _get_tf_boundary,
    "tensorflow": _get_tf_boundary,
}


def get_interface_boundary(ml_interface: str) -> Executor:
    """Chooses the proper executor for the requested interface."""
    return _get_boundary_map[ml_interface]()
