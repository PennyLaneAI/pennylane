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
Top level OpenQML module
========================
"""
import os
import logging as log
from pkg_resources import iter_entry_points

from autograd import numpy
from autograd import grad as _grad


from .configuration import Configuration
from .device import Device, DeviceError, QuantumFunctionError
import openqml.operation
from .ops import *
import openqml.expval
from .optimize import *
from .qnode import QNode
from ._version import __version__


# NOTE: this has to be imported last,
# otherwise it will clash with the .qnode import.
from .decorator import qnode


# set up logging
if "LOGGING" in os.environ:
    logLevel = os.environ["LOGGING"]
    print('Logging:', logLevel)
    numeric_level = getattr(log, logLevel.upper(), 10)
else:
    numeric_level = 100 # info

log.basicConfig(
    level=numeric_level,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log.captureWarnings(True)


# Look for an existing configuration file
default_config = Configuration()


# get list of installed plugin devices
plugin_devices = {
    entry.name: entry for entry in iter_entry_points('openqml.plugins')
}


def device(name, *args, **kwargs):
    """Load a plugin Device class and return the instance to the user.

    Args:
        name (str): the name of the device to load.

    Keyword Args:
        config (openqml.Configuration): an OpenQML configuration object
            that contains global and/or device specific configurations.
    """
    if name in plugin_devices:
        options = {}

        # load global configuration settings if available
        config = kwargs.get('config', default_config)

        if config:
            # combine configuration options with keyword arguments.
            # Keyword arguments take preference, followed by device options,
            # followed by plugin options, followed by global options.
            options.update(config['main'])
            options.update(config[name.split('.')[0]+'.global'])
            options.update(config[name])

        kwargs.pop("config", None)
        options.update(kwargs)

        # load plugin device
        p = plugin_devices[name].load()(*args, **options)

        if p.api_version != __version__:
            log.warning('Plugin API version {} does not match OpenQML version {}.'.format(p.api_version, __version__))

        return p
    else:
        raise DeviceError('Device does not exist. Make sure the required plugin is installed.')


def grad(func):
    """Wrapper around the autograd.grad function."""
    return _grad(func)


def version():
    """Version number"""
    return __version__
