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
Tests for the ``DatasetSparseArray`` attribute type.
"""


import pytest
import scipy.sparse

from pennylane.data.attributes.sparse_array import DatasetSparseArray

pytestmark = pytest.mark.data

SP_MATRIX = scipy.sparse.random(10, 15)


@pytest.mark.parametrize(
    "sp_in", [sp_type(SP_MATRIX) for sp_type in DatasetSparseArray.consumes_types()]
)
class TestDatasetsSparseArray:
    """Test bind and value initialization for a ``DatasetSparseArray``, with all
    possible sparse types."""

    def test_value_init(self, sp_in):
        """Test that a ``DatasetSparseArray`` can be value-initialized from
        any Scipy sparse array or matrix."""

        dset_sp = DatasetSparseArray(sp_in)

        assert dset_sp.info["type_id"] == "sparse_array"
        assert dset_sp.info["py_type"] == f"scipy.sparse.{type(sp_in).__qualname__}"
        assert dset_sp.sparse_array_class is type(sp_in)

        sp_out = dset_sp.get_value()
        assert isinstance(sp_out, type(sp_in))

        assert (sp_in.todense() == sp_out.todense()).all()

    def test_bind_init(self, sp_in):
        """Test that a ``DatasetSparseArray`` is correctly bind-initialized."""

        bind = DatasetSparseArray(sp_in).bind

        dset_sp = DatasetSparseArray(bind=bind)

        assert dset_sp.info["type_id"] == "sparse_array"
        assert dset_sp.info["py_type"] == f"scipy.sparse.{type(sp_in).__qualname__}"
        assert dset_sp.sparse_array_class is type(sp_in)

        sp_out = dset_sp.get_value()
        assert isinstance(sp_out, type(sp_in))

        assert (sp_in.todense() == sp_out.todense()).all()
