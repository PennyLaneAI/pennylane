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
This module provides the circuit drawing functionality used to display circuits visually.

.. currentmodule:: pennylane.drawer
.. automodule::
    :toctree: api

"""

from .draw import draw, draw_mpl
from .label import label
from .mpldrawer import MPLDrawer
from .style import available_styles, use_style
from .tape_mpl import tape_mpl
from .tape_text import tape_text

from importlib import metadata, import_module

_eps = metadata.entry_points(group="catalyst.graph_drawer")


def _load_catalyst_drawers():
    drawers = {name: _eps[name].load() for name in _eps.names}
    return drawers


_catalyst_drawers = _load_catalyst_drawers()

_pl_drawers = [
    name for name, obj in globals().items() if callable(obj) and not name.startswith("_")
]

__all__ = _pl_drawers + list(_catalyst_drawers.keys())


def __dir__():
    return __all__ + list(globals().keys())


def __getattr__(name):
    if name in _catalyst_drawers:
        func = _catalyst_drawers[name]
        func.__module__ = __name__
        setattr(import_module(__name__), name, func)
        return func

    else:
        raise AttributeError(f"module 'pennylane.drawer' has no attribute {name}")
