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
from typing import Callable

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


def _make_switches() -> [Callable[[], None], Callable[[], None], Callable[[], bool]]:
    r"""Create three functions, corresponding to an activation switch, a deactivation switch
    and a status query, in that order.

    .. note::

        While the internal variable is named in some context, this function
        can be used to make switches for any context.
    """

    _PLXPR_ENABLED = False
    # since this changes what happens with tracing, we need to turn the behavior
    # off by default to preserve our ability to jit pennylane circuits.

    def enable() -> None:
        """Enable the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr)."""
        if not has_jax:
            raise ImportError("plxpr requires JAX to be installed.")
        nonlocal _PLXPR_ENABLED
        _PLXPR_ENABLED = True

    def disable() -> None:
        """Disable the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr)."""
        nonlocal _PLXPR_ENABLED
        _PLXPR_ENABLED = False

    def status() -> bool:
        """Return whether the capturing mechanism of hybrid quantum-classical programs
        in a PennyLane Program Representation (plxpr) is enabled."""
        nonlocal _PLXPR_ENABLED
        return _PLXPR_ENABLED

    return enable, disable, status


enable_plxpr, disable_plxpr, plxpr_enabled = _make_switches()
