import numpy as np
import pytest

from pennylane.data.attributes import (
    DatasetArray,
    DatasetScalar,
    DatasetString,
    DatasetList,
    DatasetDict,
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
        ([], DatasetList),
        ([1, 2], DatasetList),
        ([np.int64(0)], DatasetList),
        ([{"a": 1}], DatasetList),
        ({}, DatasetDict),
        ({"a": [1, 2]}, DatasetDict),
    ],
)
def test_match_obj_type(type_or_obj, attribute_type):
    """Test that ``match_obj_type`` returns the expected attribute
    type for each argument."""
    assert match_obj_type(type_or_obj) is attribute_type
