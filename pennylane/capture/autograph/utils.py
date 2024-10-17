# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Utils for autograph implementation
"""


class AutoGraphError(Exception):
    """Errors related to Catalyst's AutoGraph module."""


class CompileError(Exception):
    """Error encountered in the compilation phase."""


class Patcher:
    """Patcher, a class to replace object attributes.

    Args:
        patch_data: List of triples. The first element in the triple corresponds to the object
        whose attribute is to be replaced. The second element is the attribute name. The third
        element is the new value assigned to the attribute.
    """

    def __init__(self, *patch_data):
        self.backup = {}
        self.patch_data = patch_data

        assert all(len(data) == 3 for data in patch_data)

    def __enter__(self):
        for obj, attr_name, fn in self.patch_data:
            self.backup[(obj, attr_name)] = getattr(obj, attr_name)
            setattr(obj, attr_name, fn)

    def __exit__(self, _type, _value, _traceback):
        for obj, attr_name, _ in self.patch_data:
            setattr(obj, attr_name, self.backup[(obj, attr_name)])
