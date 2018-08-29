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
import logging as log
from pkg_resources import iter_entry_points

from autograd import numpy
from autograd import grad as _grad

from .device import Device, DeviceError
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


# get list of installed plugin devices
plugin_devices = {
    entry.name: entry for entry in iter_entry_points('openqml.plugins')
}


def device(name, *args, **kwargs):
    """Load a plugin Device class and return the instance to the user."""
    if name in plugin_devices:
        # load plugin device
        p = plugin_devices[name].load()(*args, **kwargs)

        if p.api_version != __version__:
            log.warning('Plugin API version {} does not match OpenQML version {}.'.format(p.plugin_api_version, temp))

        return p
    else:
        raise DeviceError('Device does not exist. Make sure the required plugin is installed.')


def grad(func, *args):
    """Wrapper around the autograd.grad function."""
    return _grad(func)(numpy.asarray(args))


def version():
    """Version number"""
    return __version__

# short names
Op = Operation
Ex = Expectation
