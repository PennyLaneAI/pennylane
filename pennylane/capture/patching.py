# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module provides utilities for surgical, temporary patching of objects
at runtime using context managers. This approach is inspired by Catalyst's
patching system and allows for controlled, scoped modifications without
global side effects.
"""


class Patcher:
    """Context manager for temporarily patching object attributes.

    This class provides a clean way to temporarily replace attributes on objects
    within a specific scope. All changes are automatically reverted when exiting
    the context, ensuring no global side effects.

    Args:
        *patch_data: Variable number of tuples (obj, attr_name, new_value) where:
            - obj: The object to patch
            - attr_name: Name of the attribute to replace
            - new_value: The temporary value to use within the context

    Example:
        >>> import math
        >>> with Patcher((math, "pi", 3.0)):
        ...     print(math.pi)  # prints 3.0
        >>> print(math.pi)  # prints 3.141592653589793

    Example with multiple patches:
        >>> with Patcher(
        ...     (math, "pi", 3.0),
        ...     (math, "e", 2.0),
        ... ):
        ...     print(math.pi, math.e)  # prints 3.0 2.0
    """

    def __init__(self, *patch_data):
        """Initialize the patcher with patch specifications.

        Args:
            *patch_data: Tuples of (object, attribute_name, new_value)
        """
        self.backup = {}
        self.patch_data = patch_data

    def __enter__(self):
        """Apply all patches and backup original values."""
        for item in self.patch_data:
            if len(item) == 4:
                # Dictionary patch: (dict, '__dict_item__', key, value)
                obj, marker, key, fn = item
                if marker == "__dict_item__":
                    self.backup[(id(obj), "__dict_item__", repr(key))] = obj.get(
                        key, "__NOTFOUND__"
                    )
                    obj[key] = fn
            elif len(item) == 3:
                obj, attr_name, fn = item
                # Regular attribute patch
                self.backup[(obj, attr_name)] = getattr(obj, attr_name)
                setattr(obj, attr_name, fn)

    def __exit__(self, _type, _value, _traceback):
        """Restore all original values when exiting the context."""
        for item in self.patch_data:
            if len(item) == 4:
                # Dictionary patch
                obj, marker, key, _ = item
                if marker == "__dict_item__":
                    backup_val = self.backup[(id(obj), "__dict_item__", repr(key))]
                    if backup_val == "__NOTFOUND__":
                        obj.pop(key, None)
                    else:
                        obj[key] = backup_val
            elif len(item) == 3:
                # Regular attribute patch
                obj, attr_name, _ = item
                setattr(obj, attr_name, self.backup[(obj, attr_name)])
