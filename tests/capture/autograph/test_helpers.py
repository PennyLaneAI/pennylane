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

"""PyTests for AutoGraph helper functions."""

from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES
from functools import wraps as ft_wraps

import pytest

from pennylane.capture.autograph import wraps as ag_wraps


def test_wrapper_assignments():
    """Test that our custom `wraps` function works the same as the functools version,
    except for the __module__ attribute."""

    # we use the custom wrapping function itself as a target
    @ag_wraps(ag_wraps)
    def ag_wrapped():
        """stub"""

    @ft_wraps(ag_wraps)
    def ft_wrapped():
        """stub"""

    assert "__module__" in WRAPPER_ASSIGNMENTS  # if this fails the custom function can be removed
    for attr in WRAPPER_ASSIGNMENTS:
        if attr == "__module__":
            assert getattr(ag_wrapped, attr) == "test_helpers"
            assert getattr(ft_wrapped, attr) == "pennylane.capture.autograph"
        else:
            assert getattr(ag_wrapped, attr) == getattr(ft_wrapped, attr)
    for attr in WRAPPER_UPDATES:
        assert getattr(ag_wrapped, attr) == getattr(ft_wrapped, attr)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
