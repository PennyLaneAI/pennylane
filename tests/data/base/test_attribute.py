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
"""
Tests for :mod:`pennylane.data.base.attribute`.
"""

import numpy as np
import pytest

from pennylane.data.attributes import (
    DatasetArray,
    DatasetNone,
    DatasetOperator,
    DatasetScalar,
    DatasetSparseArray,
    DatasetString,
)
from pennylane.data.base.attribute import match_obj_type


@pytest.mark.parametrize(
    "type_or_obj, attribute_type",
    [
        (str, DatasetString),
        ("", DatasetString),
        ("abc", DatasetString),
        (0, DatasetScalar),
        (0.0, DatasetScalar),
        (np.int64(0), DatasetScalar),
        (complex(1, 2), DatasetScalar),
        (int, DatasetScalar),
        (complex, DatasetScalar),
        (np.array, DatasetArray),
        (np.array([0]), DatasetArray),
        (np.array([np.int64(0)]), DatasetArray),
        (np.array([complex(1, 2)]), DatasetArray),
        (np.zeros(shape=(5, 5, 7)), DatasetArray),
        (None, DatasetNone),
        (type(None), DatasetNone),
        *(
            (sp_cls, DatasetSparseArray)
            for sp_cls in sorted(DatasetSparseArray.consumes_types(), key=lambda t: str(t))
        ),
        *(
            (op, DatasetOperator)
            for op in sorted(DatasetOperator.consumes_types(), key=lambda t: str(t))
        ),
    ],
)
def test_match_obj_type(type_or_obj, attribute_type):
    """Test that ``match_obj_type`` returns the expected attribute
    type for each argument."""
    assert match_obj_type(type_or_obj) is attribute_type
