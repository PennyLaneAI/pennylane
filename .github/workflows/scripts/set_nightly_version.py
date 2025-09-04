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
import argparse

version_file_path = os.path.join(os.path.dirname(__file__), "../../../pennylane/_version.py")

assert os.path.isfile(version_file_path)

with open(version_file_path, "r+", encoding="UTF-8") as f:
    lines = f.readlines()

    parser = argparse.ArgumentParser(description="Bump the PennyLane development version.")
    parser.add_argument("--versiontype", type=str, required=True, help="The current version string to validate.")
    parser.add_argument("--pattern", type=str, default=r"(\d+).(\d+).(\d+)-dev(\d+)", help="The regex pattern to extract version components.")
    args = parser.parse_args()

    if "dev" in args.pattern:
        line_number = -2
        name = "dev"
    elif "rc" in args.pattern:
        line_number = -1
        name = "rc"
    
    version_line = lines[line_number]
    version = args.versiontype
    print("version: ",version)
    assert version in version_line 

    pattern = args.pattern
    match = re.search(pattern, version_line)
    assert match

    major, minor, bug, dev = match.groups()

    replacement = f'{version} "{major}.{minor}.{bug}-{name}{int(dev)+1}"\n'
    lines[line_number] = replacement

    f.seek(0)
    f.writelines(lines)
    f.truncate()
