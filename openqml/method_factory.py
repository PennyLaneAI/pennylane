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

class MethodFactory(type):
    """Metaclass that allows derived classes to dynamically instantiate
    new objects based on undefined methods. The dynamic methods pass their arguments
    directly to __init__ of the inheriting class."""
    def __getattr__(cls, name):
        """Get the attribute call via name"""
        def new_object(*args, **kwargs):
            """Return a new object of the same class, passing the attribute name
            as the first parameter, along with any additional parameters."""
            return cls(name, *args, **kwargs)
        return new_object
