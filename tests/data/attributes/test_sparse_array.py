import pytest
import scipy.sparse
from pennylane.data.attributes.sparse_array import DatasetSparseArray, _ALL_SPARSE


SP_MATRIX = scipy.sparse.random(10, 15)


class TestDatasetsSparseArray:
    @pytest.mark.parametrize("sp_in", [sp_type(SP_MATRIX) for sp_type in _ALL_SPARSE])
    def test_value_init(self, sp_in):
        """Test that a ``DatasetSparseArray`` can be value-initialized from
        any Scipy sparse array."""

        dset_sm = DatasetSparseArray(sp_in)

        assert dset_sm.info["type_id"] == "sparse_array"
        assert dset_sm.info["py_type"] == f"scipy.sparse.{type(sp_in).__qualname__}"

        sp_out = dset_sm.get_value()
        assert isinstance(sp_out, type(sp_in))

        assert (sp_in.todense() == sp_out.todense()).all()
