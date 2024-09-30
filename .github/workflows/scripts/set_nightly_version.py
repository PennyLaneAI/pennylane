# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module bumps the PennyLane development version by one unit.
"""

import os
import re

version_file_path = os.path.join(os.path.dirname(__file__), "../../../pennylane/_version.py")

assert os.path.isfile(version_file_path)

with open(version_file_path, "r+", encoding="UTF-8") as f:
    lines = f.readlines()

    version_line = lines[-1]
    assert "__version__ = " in version_line

    pattern = r"(\d+).(\d+).(\d+)-dev(\d+)"
    match = re.search(pattern, version_line)
    assert match

    major, minor, bug, dev = match.groups()

    replacement = f'__version__ = "{major}.{minor}.{bug}-dev{int(dev)+1}"\n'
    lines[-1] = replacement

    f.seek(0)
    f.writelines(lines)
    f.truncate()
