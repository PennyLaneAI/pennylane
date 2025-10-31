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
import os
import json
from importlib import metadata
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
from sys import version_info

import numpy
import scipy

if find_spec("jax"):
    jax_version = version("jax")
else:
    jax_version = None

def _pkg_location():
    """Return absolute path to the installed PennyLane package."""
    try:
        dist = metadata.distribution("pennylane")
        return os.path.abspath(str(dist.locate_file("")))
    except (PackageNotFoundError, OSError):
        # Use imported module path if available
        mod = sys.modules.get("pennylane")
        if mod and getattr(mod, "__file__", None):
            return os.path.abspath(os.path.dirname(mod.__file__))
        return "(unknown)"


def about():
    """
    Prints the information for pennylane installation.
    """
    if version_info[:2] == (3, 9):
        from pkg_resources import iter_entry_points  # pylint:disable=import-outside-toplevel

        plugin_devices = iter_entry_points("pennylane.plugins")
        dist_name = "project_name"
    else:  # pragma: no cover
        plugin_devices = metadata.entry_points(group="pennylane.plugins")
        dist_name = "name"

    try:
        dist = metadata.distribution("pennylane")
        meta = dist.metadata
        location = _pkg_location()

        lines = [
            f"Name: {meta.get('Name', 'PennyLane')}",
            f"Version: {meta.get('Version', '')}",
            f"Summary: {meta.get('Summary', '')}",
            f"Home-page: {meta.get('Home-page', '')}",
            f"Author: {meta.get('Author', '')}",
            f"License: {meta.get('License', '')}",
            f"Location: {location or '(unknown)'}",
        ]

        # PEP 610: detect editable with direct_url.json
        try:
            raw = dist.read_text("direct_url.json")
            if raw is None:
                raise FileNotFoundError
            direct = json.loads(raw)
            if direct.get("dir_info", {}).get("editable"):
                url = direct.get("url", "")
                if url.startswith("file://"):
                    url = url[7:]
                lines.append(f"Editable project location: {url}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        info = "\n".join(lines)

    except PackageNotFoundError:
        info = "PennyLane version info unavailable (no distribution metadata)"
    except OSError:
        info = "PennyLane version info unavailable (metadata read error)"

    print(info)
    print(f"Platform info:           {platform.platform(aliased=True)}")
    print(
        f"Python version:          {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version:           {numpy.__version__}")
    print(f"Scipy version:           {scipy.__version__}")
    print(f"JAX version:             {jax_version}")

    print("Installed devices:")

    for d in plugin_devices:
        print(f"- {d.name} ({getattr(d.dist, dist_name)}-{d.dist.version})")


if __name__ == "__main__":
    about()
