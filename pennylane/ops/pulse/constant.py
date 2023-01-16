# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains the ``constant`` pulse convenience function."""

import jax


def constant(index, ti, tf):
    """Returns a function that evaluates to a constant when time ``t`` is between ``ti`` and ``tf``.

    Args:
        index (int): parameter index that we want to return
        ti (float): initial time
        tf (float): final time
    """

    def _constant(params, t):
        return jax.lax.cond(ti < t < tf, lambda: params[index], lambda: 0)

    return _constant
