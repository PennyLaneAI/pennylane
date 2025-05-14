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
"""Contains a function for getting the best MCM method for a given QNode."""

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qnode import QNode


def get_best_mcm_method(qnode: QNode):
    """Returns a string that represents the 'best' MCM method
    for a particular QNode.

    Args:
        qnode (QNode): the qnode to get the 'best' MCM method for.

    Returns:
        str: the MCM method.
    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        return

    return wrapper
