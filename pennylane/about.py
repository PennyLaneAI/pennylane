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
import importlib
import sys
from pkg_resources import iter_entry_points
import numpy
import scipy

# The following if/else block enables support for pip versions 19.3.x
_parent_module = importlib.util.find_spec("pip._internal.main") or importlib.util.find_spec(
    "pip._internal"
)
_internal_main = importlib.util.module_from_spec(_parent_module)
_parent_module.loader.exec_module(_internal_main)


def about():
    """
    Prints the information for pennylane installation.
    """
    plugin_devices = iter_entry_points("pennylane.plugins")
    _internal_main.main(["show", "pennylane"])
    print(f"Platform info:           {platform.platform(aliased=True)}")
    print(
        f"Python version:          {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version:           {numpy.__version__}")
    print(f"Scipy version:           {scipy.__version__}")

    print("Installed devices:")

    for d in plugin_devices:
        print(f"- {d.name} ({d.dist.project_name}-{d.dist.version})")


if __name__ == "__main__":
    about()
