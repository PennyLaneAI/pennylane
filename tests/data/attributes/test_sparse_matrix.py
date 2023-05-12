import pytest
import scipy.sparse
from scipy.sparse import csr_matrix

from pennylane.data.attributes.sparse_matrix import DatasetSparseMatrix


class TestDatasetsSparseMatrix:
    @pytest.mark.parametrize("sp_in", [scipy.sparse.random(10, 7, format="csr")])
    def test_value_init_(self, sp_in):
        """Test that a ``Dataset"""

        dset_sm = DatasetSparseMatrix(sp_in)

        assert dset_sm.info["type_id"] == "sparse_matrix"
        assert dset_sm.info["py_type"] == "scipy.sparse.csr_matrix"

        sp_out = dset_sm.get_value()
        assert isinstance(sp_out, csr_matrix)

        assert (sp_out.data == sp_in.data).all()
        assert (sp_out.indices == sp_in.indices).all()
        assert (sp_out.indptr == sp_in.indptr).all()
        assert sp_out.shape == sp_in.shape
