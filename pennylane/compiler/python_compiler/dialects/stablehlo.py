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
Extended StableHLO dialect that dynamically includes all upstream operations
plus custom operations for PennyLane's compiler infrastructure.

This module automatically imports all operations and attributes from the upstream
xdsl.dialects.stablehlo and adds custom ones without needing to hardcode
the upstream operation list.
"""

import xdsl.dialects.stablehlo as xstablehlo
from xdsl.ir import Dialect

# Operations to add to the dialect
OPERATIONS = []

# Operations from upstream that should be deleted/replaced in the local version
UPSTREAM_OPERATIONS_TO_DELETE = []


def filter_and_extend_upstream(upstream_list, to_delete, to_add):
    """Filter out operations/attributes from upstream list and add new ones.

    Args:
        upstream_list: List of operations/attributes to filter
        to_delete: List of operations/attributes to remove
        to_add: List of operations/attributes to add

    Returns:
        Modified list of operations/attributes
    """
    filtered_ops = list(upstream_list)

    # Remove operations that should be deleted
    for op_to_delete in to_delete:
        if op_to_delete in filtered_ops:
            filtered_ops.remove(op_to_delete)

    # Add new operations
    filtered_ops.extend(to_add)

    return filtered_ops


all_operations = filter_and_extend_upstream(
    xstablehlo.StableHLO.operations, UPSTREAM_OPERATIONS_TO_DELETE, OPERATIONS
)
all_attributes = list(xstablehlo.StableHLO.attributes)

# Create the extended StableHLO dialect by dynamically getting upstream components
StableHLO = Dialect(
    "stablehlo",
    all_operations,
    all_attributes,
)
