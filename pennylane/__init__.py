# Copyright 2018 Xanadu Quantum Technologies Inc.

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
This is the top level module from which all basic functions and classes of
PennyLane can be directly imported.
"""
import os
from pkg_resources import iter_entry_points

from autograd import numpy
from autograd import grad as _grad
from autograd import jacobian as _jacobian

from semantic_version import Version, Spec

import pennylane.operation

import pennylane.init
import pennylane.templates.layers
import pennylane.templates.embeddings
from pennylane.about import about


from .configuration import Configuration
from ._device import Device, DeviceError
from .measure import expval, var, sample
from .ops import *
from .optimize import *
from .qnode import QNode, QuantumFunctionError
from ._version import __version__

# NOTE: this has to be imported last,
# otherwise it will clash with the .qnode import.
from .decorator import qnode


# overwrite module docstrings
numpy.__doc__ = "NumPy with automatic differentiation support, provided by Autograd."


# Look for an existing configuration file
default_config = Configuration("config.toml")


# get list of installed plugin devices
plugin_devices = {entry.name: entry for entry in iter_entry_points("pennylane.plugins")}


def device(name, *args, **kwargs):
    r"""device(name, wires=1, *args, **kwargs)
    Load a plugin :class:`~.Device` and return the instance.

    This function is used to load a particular quantum device,
    which can then be used to construct QNodes.

    PennyLane comes with support for the following two devices:

    * :mod:`'default.qubit' <pennylane.plugins.default_qubit>`: a simple pure
      state simulator of qubit-based quantum circuit architectures.

    * :mod:`'default.gaussian' <pennylane.plugins.default_gaussian>`: a simple simulator
      of Gaussian states and operations on continuous-variable circuit architectures.

    In addition, additional devices are supported through plugins — see
    :ref:`plugins` for more details.

    All devices must be loaded by specifying their **short-name** as listed above,
    followed by the number of *wires* (subsystems) you wish to initialize.

    Some devices may accept additional arguments. For instance,
    ``default.gaussian`` accepts the keyword argument ``hbar``, to set
    the convention used in the commutation relation :math:`[\x,\p]=i\hbar`
    (by default set to 2).

    Please refer to the documentation for the individual devices to see any
    additional arguments that might be required or supported.

    Args:
        name (str): the name of the device to load
        wires (int): the number of wires (subsystems) to initialise
            the device with

    Keyword Args:
        config (pennylane.Configuration): a PennyLane configuration object
            that contains global and/or device specific configurations.
    """
    if name in plugin_devices:
        options = {}

        # load global configuration settings if available
        config = kwargs.get("config", default_config)

        if config:
            # combine configuration options with keyword arguments.
            # Keyword arguments take preference, followed by device options,
            # followed by plugin options, followed by global options.
            options.update(config["main"])
            options.update(config[name.split(".")[0] + ".global"])
            options.update(config[name])

        kwargs.pop("config", None)
        options.update(kwargs)

        # loads the plugin device class
        plugin_device_class = plugin_devices[name].load()

        if Version(version()) not in Spec(plugin_device_class.pennylane_requires):
            raise DeviceError(
                "The {} plugin requires PennyLane versions {}, however PennyLane "
                "version {} is installed.".format(
                    name, plugin_device_class.pennylane_requires, __version__
                )
            )

        # load plugin device
        return plugin_device_class(*args, **options)

    raise DeviceError(
        "Device does not exist. Make sure the required plugin is installed."
    )


def grad(func, argnum):
    """Returns the gradient as a callable function of (functions of) QNodes.

    This is a wrapper around the :mod:`autograd.grad` functions.

    Args:
        func (function): a Python function or QNode that contains
            a combination of quantum and classical nodes
        argnum (int or list(int)): which argument(s) to take the gradient
            with respect to

    Returns:
        function: the function that returns the gradient of the input
        function with respect to the arguments in argnum
    """
    # pylint: disable=no-value-for-parameter
    return _grad(func, argnum)


def jacobian(func, argnum):
    """Returns the Jacobian as a callable function of vector-valued
    (functions of) QNodes.

    This is a wrapper around the :mod:`autograd.jacobian` function.

    Args:
        func (function): a vector-valued Python function or QNode that contains
            a combination of quantum and classical nodes. The output of the computation
            must consist of a single NumPy array (if classical) or a tuple of
            expectation values (if a quantum node)
        argnum (int or Sequence[int]): which argument to take the gradient
            with respect to. If a sequence is given, the Jacobian matrix
            corresponding to all input elements and all output elements is returned.

    Returns:
        function: the function that returns the Jacobian of the input
        function with respect to the arguments in argnum
    """
    # pylint: disable=no-value-for-parameter
    if isinstance(argnum, int):
        return _jacobian(func, argnum)
    return lambda *args, **kwargs: numpy.stack([_jacobian(func, arg)(*args, **kwargs) for arg in argnum]).T


def version():
    """Returns the PennyLane version number."""
    return __version__
