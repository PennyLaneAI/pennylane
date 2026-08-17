# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the switches to (de)activate the capturing mechanism, and a
status reporting function on whether it is enabled or not.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

has_jax = True
is_jax_compatible = True

REQUIRED_JAX_VERSION = "0.7.1"

try:
    import jax  # pylint: disable=unused-import
    from packaging import version

    jax_version = version.parse(jax.__version__)
    required_version = version.parse(REQUIRED_JAX_VERSION)
    if jax_version != required_version:  # pragma: no cover
        is_jax_compatible = False
except ImportError:  # pragma: no cover
    has_jax = False
    is_jax_compatible = False


def _make_switches() -> tuple[Callable, Callable, Callable[[], bool], Callable]:
    r"""Create four functions, corresponding to an activation switch, a deactivation switch
    and a status query, and a context manager, in that order.

    .. note::

        While the internal variable is named in some context, this function
        can be used to make switches for any context.
    """

    _FEATURE_ENABLED = ContextVar("_FEATURE_ENABLED", default=False)
    # since this changes what happens with tracing, we need to turn the behaviour
    # off by default to preserve our ability to jit pennylane circuits.

    def enable_fn() -> None:
        """Enable the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr)."""
        if not has_jax:
            raise ImportError("plxpr requires JAX to be installed.")
        if not is_jax_compatible:  # pragma: no cover
            raise ImportError(
                f"PennyLane's program capture requires JAX=={REQUIRED_JAX_VERSION} to be installed to ensure functionality. "
                f"You have JAX {jax.__version__} installed. "
                f"Please pin JAX by running: pip install --upgrade jax=={REQUIRED_JAX_VERSION} jaxlib=={REQUIRED_JAX_VERSION}"
            )
        _FEATURE_ENABLED.set(True)

    def disable_fn() -> None:
        """Disable the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr)."""
        _FEATURE_ENABLED.set(False)

    def status_fn() -> bool:
        """Return whether the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr) is enabled."""
        return _FEATURE_ENABLED.get()

    @contextmanager
    def toggle_ctx_fn(new_state: bool):
        """A context manager in which capture is enabled or disabled temporarily."""
        token = _FEATURE_ENABLED.set(new_state)
        try:
            yield
        finally:
            _FEATURE_ENABLED.reset(token)

    return enable_fn, disable_fn, status_fn, toggle_ctx_fn


enable, disable, enabled, toggle_ctx = _make_switches()

pause = partial(toggle_ctx, new_state=False)
