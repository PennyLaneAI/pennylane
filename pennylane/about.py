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
r"""
About
=====

**Module name:** :mod:`pennylane.about`

.. currentmodule:: pennylane.about

A simple module to display all the details of the `pennylane` installation,
e.g., OS, version, `Numpy` and `Scipy` versions, installation method.

Behaviour
---------

The module simply prints the information on screen and can be accessed as
`pennylane.about()`

Summary of methods
------------------

.. currentmodule:: pennylane.about

.. autosummary::
    about

Code details
~~~~~~~~~~~~

.. currentmodule:: pennylane.about

"""
import sys
import platform
from pip import _internal
from pkg_resources import iter_entry_points
import numpy
import scipy


def about():
    """
    Prints the information for pennylane installation.
    """
    plugin_devices = [entry.name for entry in iter_entry_points("pennylane.plugins")]
    _internal.main(["show", "pennylane"])
    print("Python Version:          {0}.{1}.{2}".format(*sys.version_info[0:3]))
    print("Platform Info:           {}{}".format(platform.system(), platform.machine()))
    print("Installed plugins:       {}".format(plugin_devices))
    print("Numpy Version:           {}".format(numpy.__version__))
    print("Scipy Version:           {}".format(scipy.__version__))


if __name__ == "__main__":
    about()
