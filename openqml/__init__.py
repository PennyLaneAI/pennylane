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
"""Top level OpenQML module"""
import os
import logging as log
from pkg_resources import iter_entry_points

import toml
from appdirs import user_config_dir

from autograd import numpy
from autograd import grad as _grad

from .device import Device, DeviceError, QuantumFunctionError
from .expectation import Expectation
from .ops import *
from .qfunc import qfunc
from .qnode import QNode
from .optimizer import Optimizer
from .variable import Variable
from ._version import __version__

# set up logger
logLevel = 'info'
numeric_level = getattr(log, logLevel.upper(), 10)
log.basicConfig(
    level=numeric_level,
    format='\n%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log.captureWarnings(True)


def safe_get(dct, *keys):
    """Safely return value from a nested dictionary."""
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return {}
    return dct


def load_config(path):
    """Load a configuration file."""
    with open(path) as f:
        config = toml.load(f)


# Look for an existing configuration file
config = None
for directory in os.curdir, user_config_dir('openqml'), os.environ.get("OPENQML_CONF", ""):
    try:
        load_config(os.path.join(directory, 'config.toml'))
    except FileNotFoundError:
        log.warning('No OpenQML configuration file found.')


# get list of installed plugin devices
plugin_devices = {
    entry.name: entry for entry in iter_entry_points('openqml.plugins')
}


def device(name, *args, **kwargs):
    """Load a plugin Device class and return the instance to the user."""
    if name in plugin_devices:
        options = {}

        if config is not None:
            # load global configuration settings if available
            global_config = safe_get(config, 'main')
            # load plugin configuration settings if available
            plugin_config = safe_get(config, name.split('.')[0], 'global')
            # load device configuration settings if available
            device_config = safe_get(config, *name.split('.'))

            # combine with configuration options with keyword arguments.
            # Keyword arguments take preference, followed by device options,
            # followed by plugin options, followed by global options.
            options.update(global_config)
            options.update(plugin_config)
            options.update(device_config)

        options.update(kwargs)

        # load plugin device
        p = plugin_devices[name].load()(*args, **options)

        if p.api_version != __version__:
            log.warning('Plugin API version {} does not match OpenQML version {}.'.format(p.plugin_api_version, temp))

        return p
    else:
        raise DeviceError('Device does not exist. Make sure the required plugin is installed.')


def grad(func, args):
    """Wrapper around the autograd.grad function."""
    return _grad(func)(*args)


def version():
    """Version number"""
    return __version__

# short names
Op = Operation
Ex = Expectation
