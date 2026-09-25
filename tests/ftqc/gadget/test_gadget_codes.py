# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for codes and distance claims, ``pennylane.ftqc.gadget.codes``."""

import numpy as np
import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import rep_chain, steane_code


def _rep3(lx=((1, 1, 1),), lz=((1, 0, 0),)):
    hz, _, _ = rep_chain(3)
    return gadget.CSSCode(
        name="rep3",
        hx=np.zeros((0, 3), dtype=np.uint8),
        hz=hz,
        lx=np.array(lx, dtype=np.uint8),
        lz=np.array(lz, dtype=np.uint8),
    )


class TestDistanceClaim:
    """Tests for distance claims, which always carry a regime and a certified flag."""

    def test_defaults_are_uncertified_static(self):
        """Test that a bare claim is static and not certified."""
        claim = gadget.DistanceClaim(value=3)
        assert claim.regime == "static"
        assert not claim.certified
        assert str(claim) == "d=3 (static, UNCERTIFIED: asserted by author, not checked)"

    def test_certified_str(self):
        """Test that a certified claim prints as certified, with its regime and method."""
        claim = gadget.DistanceClaim(3, "phenomenological", True, "stim")
        assert str(claim) == "d=3 (phenomenological, certified: stim)"


class TestCSSCode:
    """Tests for CSS codes, which are validated at construction."""

    def test_steane_parameters(self):
        """Test the parameters of the Steane code."""
        code = steane_code()
        assert (code.n, code.k) == (7, 1)
        assert code.max_check_weight == 4
        assert code.max_qubit_degree == 6
        assert str(code).startswith("Steane [[7,1]], d=3 (static, certified:")

    def test_repetition_pair_parameters(self, rep_zz):
        """Test the parameters of the two-block repetition code used by the ZZ merge."""
        code, _, _ = rep_zz
        assert (code.n, code.k) == (6, 2)
        assert code.distance.value == 3

    def test_padded_chain_is_not_a_code(self):
        """Test that a repetition chain padded into a wider frame is rejected, because the
        padding qubits are free and rank accounting disagrees with the declared k."""
        hz, lx, lz = rep_chain(3, offset=0, n_frame=6)
        with pytest.raises(
            gadget.CodeError,
            match="rep3-padded: rank accounting gives k=4, but got 1 X logicals and 1 Z logicals",
        ):
            gadget.CSSCode(
                name="rep3-padded", hx=np.zeros((0, 6), dtype=np.uint8), hz=hz, lx=lx, lz=lz
            )

    def test_shape_mismatch(self):
        """Test that matrices over different frames are rejected."""
        with pytest.raises(gadget.CodeError, match=r"lz has shape \(1, 2\), expected \(\*, 3\)"):
            _rep3(lz=((1, 0),))

    def test_css_condition(self):
        """Test that anticommuting X and Z checks are rejected."""
        with pytest.raises(gadget.CodeError, match="CSS condition violated"):
            gadget.CSSCode(
                name="bad",
                hx=np.array([[1, 0]], dtype=np.uint8),
                hz=np.array([[1, 1]], dtype=np.uint8),
                lx=np.zeros((0, 2), dtype=np.uint8),
                lz=np.zeros((0, 2), dtype=np.uint8),
            )

    def test_x_logical_must_commute_with_z_checks(self):
        """Test that an X logical anticommuting with a Z check is rejected."""
        with pytest.raises(gadget.CodeError, match="some X logical does not commute"):
            _rep3(lx=((1, 0, 0),))

    def test_z_logical_must_commute_with_x_checks(self):
        """Test that a Z logical anticommuting with an X check is rejected."""
        hx, _, _ = rep_chain(3)
        with pytest.raises(gadget.CodeError, match="some Z logical does not commute"):
            gadget.CSSCode(
                name="xrep3",
                hx=hx,
                hz=np.zeros((0, 3), dtype=np.uint8),
                lx=np.array([[1, 0, 0]], dtype=np.uint8),
                lz=np.array([[1, 0, 0]], dtype=np.uint8),
            )

    def test_logicals_must_be_paired(self):
        """Test that X and Z logicals which do not anticommute pairwise are rejected."""
        with pytest.raises(gadget.CodeError, match="not symplectically paired"):
            _rep3(lz=((1, 1, 0),))

    def test_from_matrices_coerces(self):
        """Test that array-likes are coerced and the frame size is inferred."""
        code = gadget.CSSCode.from_matrices(
            "rep3", hx=[], hz=[[1, 1, 0], [0, 1, 1]], lx=[1, 1, 1], lz=[1, 0, 0]
        )
        assert code.hx.shape == (0, 3)
        assert code.lx.dtype == np.uint8
        assert code.k == 1

    def test_from_matrices_needs_a_frame(self):
        """Test that the frame size cannot be inferred from empty matrices alone."""
        with pytest.raises(gadget.CodeError, match="cannot infer the number of qubits"):
            gadget.CSSCode.from_matrices("empty", [], [], [], [])

    def test_fingerprint(self):
        """Test that the fingerprint is a stable content hash of the algebra."""
        assert _rep3().fingerprint() == _rep3().fingerprint()
        assert _rep3().fingerprint() != _rep3(lz=((0, 1, 0),)).fingerprint()
        assert len(_rep3().fingerprint()) == 16
