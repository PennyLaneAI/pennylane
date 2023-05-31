import pytest
import scipy.sparse
from pennylane.data.attributes.sparse_array import DatasetSparseArray


SP_MATRIX = scipy.sparse.random(10, 15)


class TestDatasetsSparseArray:
    @pytest.mark.parametrize(
        "sp_in", [sp_type(SP_MATRIX) for sp_type in DatasetSparseArray.consumes_types()]
    )
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
