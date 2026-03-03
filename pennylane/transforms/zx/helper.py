# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Helper functions for the ZX calculus module.
"""
from functools import wraps


def _needs_pyzx(func):
    """Private function to use as a ZX-based transforms decorator to raise the
    appropriate error when the pyzx external package is not installed."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # pylint: disable=import-outside-toplevel,unused-import
            import pyzx

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "The `pyzx` package is required. You can install it with `pip install pyzx`."
            ) from e

        return func(*args, **kwargs)

    return wrapper
