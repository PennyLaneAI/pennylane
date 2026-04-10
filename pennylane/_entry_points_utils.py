# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This contains a single utility function for calling entry-points from PennyLane directly. This
should be called in __init__.py files wherein a given group of entry-points is desired to be
accessed from in PennyLane.
"""

import sys
from importlib import metadata


def _setup_entry_points(module_name, group_name):  # pragma: no cover
    """Returns dunder methods required to import entry-points from the module in which this function is called. Entry-point
    elements are lazy-loaded.

    Args:
        module_name (str): The name of the module that this function gets called in.
        group_name (str | list(str)): The entry-point group name(s).

    Returns:
        Tuple(Callable):
            The module's modified ``__all__``, ``__getattr__``, and ``__dir__`` methods.
    """
    # Get entry points from the given group name (or group names)
    if isinstance(group_name, list):
        eps = metadata.entry_points(group=group_name[0])
        for i in range(1, len(group_name)):
            eps += metadata.entry_points(group=group_name[i])
    else:
        eps = metadata.entry_points(group=group_name)

    ep_dict = {ep.name: ep for ep in eps}
    ep_names = list(ep_dict.keys())

    current_module = sys.modules[module_name]

    # all public functions and classes
    current_module_funcs = [
        name
        for name, obj in current_module.__dict__.items()
        if callable(obj) and not name.startswith("_")
    ]

    def module_getattr(name):
        """The new __getattr__ method for the current_module"""
        if name in ep_dict:
            # lazy load the entry point
            func = ep_dict[name].load()
            func.__module__ = module_name
            setattr(current_module, name, func)
            return func

        if current_module.__name__ == "pennylane":
            if name == "plugin_devices":
                # pylint: disable=import-outside-toplevel
                from pennylane.devices.device_constructor import plugin_devices

                return plugin_devices

        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    # add entry points to the module
    module_all = list(current_module_funcs + ep_names)

    def module_dir():
        """The new __dir__ method for the current_module"""
        return sorted(set(module_all) | set(current_module.__dict__.keys()))

    return module_all, module_getattr, module_dir
