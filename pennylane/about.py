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
r"""
This module contains the :func:`about` function to display all the details of the PennyLane installation,
e.g., OS, version, `Numpy` and `Scipy` versions, installation method.
"""
import platform
import sys
from subprocess import check_output
from importlib import metadata
from sys import version_info
import numpy
import scipy


def about():
    """
    Prints the information for pennylane installation.
    """
    if version_info[:2] == (3, 9):
        from pkg_resources import iter_entry_points  # pylint:disable=import-outside-toplevel

        plugin_devices = iter_entry_points("pennylane.plugins")
        dist_name = "project_name"
    else:  # pragma: no cover
        plugin_devices = metadata.entry_points(  # pylint:disable=unexpected-keyword-arg
            group="pennylane.plugins"
        )
        dist_name = "name"
    print(check_output([sys.executable, "-m", "pip", "show", "pennylane"]).decode())
    print(f"Platform info:           {platform.platform(aliased=True)}")
    print(
        f"Python version:          {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version:           {numpy.__version__}")
    print(f"Scipy version:           {scipy.__version__}")

    print("Installed devices:")

    for d in plugin_devices:
        print(f"- {d.name} ({getattr(d.dist, dist_name)}-{d.dist.version})")


if __name__ == "__main__":
    about()
