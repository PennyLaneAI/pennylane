# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the detailed error messages in qml.assert_equal for MeasurementProcess.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import ExpectationMP, MeasurementValue, MidMeasureMP, ProbabilityMP, SampleMP


class TestMeasurementAssertEqualErrors:
    """Tests for the detailed error messages provided by qml.assert_equal for MeasurementProcess."""

    def test_different_obs_type(self):
        """Test error when observables have different types."""
        m1 = qml.expval(qml.X(0))
        m2 = qml.expval(qml.Y(0))
        with pytest.raises(AssertionError, match="observables are not equal because op1 and op2 are of different types"):
            qml.assert_equal(m1, m2)

    def test_different_obs_wires(self):
        """Test error when observables have different wires."""
        m1 = qml.expval(qml.X(0))
        m2 = qml.expval(qml.X(1))
        with pytest.raises(AssertionError, match="observables are not equal because op1 and op2 have different wires"):
            qml.assert_equal(m1, m2)

    @pytest.mark.jax
    def test_different_obs_interface(self):
        """Test error when observables have different interfaces."""
        import jax
        M1 = np.eye(2)
        M2 = jax.numpy.eye(2)
        ob1 = qml.Hermitian(M1, 0)
        ob2 = qml.Hermitian(M2, 0)
        m1 = qml.expval(ob1)
        m2 = qml.expval(ob2)
        with pytest.raises(AssertionError, match="observables are not equal because Parameters have different interfaces"):
            qml.assert_equal(m1, m2, check_interface=True)
        qml.assert_equal(m1, m2, check_interface=False) # Should pass if interface check is off

    def test_obs_vs_no_obs(self):
        """Test error when one has obs and the other doesn't."""
        m1 = qml.expval(qml.X(0))
        m2 = ExpectationMP(wires=[0]) # Manually create MP without obs
        with pytest.raises(AssertionError, match="one has an observable while the other does not"):
            qml.assert_equal(m1, m2)

    def test_mv_vs_no_mv(self):
        """Test error when one has mv and the other doesn't."""
        mv = qml.measure(0)
        m1 = qml.expval(mv)
        m2 = qml.expval(qml.X(0))
        # Note: the observable check takes precedence here
        with pytest.raises(AssertionError, match="one has an observable while the other does not"):
            qml.assert_equal(m1, m2)

        # Create case where obs match (both None) but mv differs
        m3 = SampleMP(mv=mv)
        m4 = SampleMP(wires=[0]) # No mv
        with pytest.raises(AssertionError, match="one has a measurement value while the other does not"):
             qml.assert_equal(m3, m4)

    def test_different_wires_no_obs_no_mv(self):
        """Test error when wires differ (no obs/mv)."""
        m1 = ExpectationMP(wires=[0])
        m2 = ExpectationMP(wires=[1])
        with pytest.raises(AssertionError, match=r"their wires are different\. Got Wires\(\[0\]\) and Wires\(\[1\]\)"):
            qml.assert_equal(m1, m2)

    def test_different_eigvals_no_obs_no_mv(self):
        """Test error when eigvals differ (no obs/mv)."""
        m1 = ExpectationMP(wires=[0], eigvals=np.array([1, -1]))
        m2 = ExpectationMP(wires=[0], eigvals=np.array([1, 0]))
        with pytest.raises(AssertionError, match="their eigenvalues are different. Got"):
            qml.assert_equal(m1, m2)

    def test_eigvals_vs_no_eigvals_no_obs_no_mv(self):
        """Test error when one has eigvals and the other doesn't (no obs/mv)."""
        m1 = ExpectationMP(wires=[0], eigvals=np.array([1, -1]))
        m2 = ExpectationMP(wires=[0], eigvals=None)
        with pytest.raises(AssertionError, match="one has eigenvalues while the other does not"):
            qml.assert_equal(m1, m2)

    def test_different_mv_list_len(self):
        """Test error when lists of measurement values have different lengths."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        m1 = qml.probs(op=[mv1])
        m2 = qml.probs(op=[mv1, mv2])
        with pytest.raises(AssertionError, match="their measurement value lists have different lengths. Got 1 and 2"):
            qml.assert_equal(m1, m2)

    def test_different_mv_list_elements(self):
        """Test error when lists of measurement values differ at an element."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1)
        mv3 = qml.measure(2)
        mv3.measurements[0].id = mv2.measurements[0].id # Ensure id doesn't cause failure
        mv4 = qml.measure(3) # Different wire

        m1 = qml.probs(op=[mv1, mv2])
        m2 = qml.probs(op=[mv1, mv4]) # Different second element
        with pytest.raises(AssertionError, match="their measurement value lists differ at index 1 because underlying measurements are not equal because wires are different"):
            qml.assert_equal(m1, m2)

    def test_incompatible_mv_types(self):
        """Test error when measurement value types are incompatible (e.g., list vs single)."""
        mv1 = qml.measure(0)
        m1 = qml.probs(op=[mv1])
        m2 = qml.probs(op=mv1) # Single MV, not list
        with pytest.raises(AssertionError, match="their measurement values have incompatible types"):
            qml.assert_equal(m1, m2)

    @pytest.mark.jax
    def test_different_abstract_mv(self):
        """Test error when comparing different abstract measurement values."""
        import jax

        @jax.jit
        def check_abstract_mv():
            mv1 = qml.measure(0)
            mv2 = qml.measure(1)
            m1 = qml.expval(mv1)
            m2 = qml.expval(mv2)
            with pytest.raises(AssertionError, match="their measurement values are different abstract tracers"):
                qml.assert_equal(m1, m2)
            return True

        check_abstract_mv()

    def test_different_measurement_value_attributes(self):
        """Test error originating from MeasurementValue comparison."""
        mv1 = qml.measure(0)
        mv2 = qml.measure(1) # Different wire
        m1 = qml.expval(mv1)
        m2 = qml.expval(mv2)
        # The error bubbles up from the comparison of the underlying MidMeasureMP
        with pytest.raises(AssertionError, match="measurement values are not equal because underlying measurements are not equal because wires are different"):
            qml.assert_equal(m1,m2)


# Specific MeasurementProcess subclass tests

class TestSpecialMeasurementAssertEqualErrors:
    """Tests error messages for specific MeasurementProcess subclasses."""

    def test_mid_measure_different_reset(self):
        """Test MidMeasureMP reset flag difference."""
        mp1 = MidMeasureMP(0, reset=True)
        mp2 = MidMeasureMP(0, reset=False)
        with pytest.raises(AssertionError, match="reset flags are different"):
            qml.assert_equal(mp1, mp2)

    def test_mid_measure_different_id(self):
        """Test MidMeasureMP id difference."""
        mp1 = MidMeasureMP(0, id="a")
        mp2 = MidMeasureMP(0, id="b")
        with pytest.raises(AssertionError, match="ids are different"):
            qml.assert_equal(mp1, mp2)

    def test_mid_measure_different_postselect(self):
        """Test MidMeasureMP postselect difference."""
        mp1 = MidMeasureMP(0, postselect=0)
        mp2 = MidMeasureMP(0, postselect=1)
        with pytest.raises(AssertionError, match="postselect values are different"):
            qml.assert_equal(mp1, mp2)

    def test_vn_entropy_different_log_base(self):
        """Test VnEntropyMP log_base difference."""
        m1 = qml.vn_entropy(wires=[0], log_base=2)
        m2 = qml.vn_entropy(wires=[0], log_base=np.e)
        with pytest.raises(AssertionError, match="log bases are different"):
            qml.assert_equal(m1, m2)

    def test_mutual_info_different_log_base(self):
        """Test MutualInfoMP log_base difference."""
        m1 = qml.mutual_info(wires0=[0], wires1=[1], log_base=2)
        m2 = qml.mutual_info(wires0=[0], wires1=[1], log_base=np.e)
        with pytest.raises(AssertionError, match="log bases are different"):
            qml.assert_equal(m1, m2)

    def test_counts_different_all_outcomes(self):
        """Test CountsMP all_outcomes difference."""
        m1 = qml.counts(wires=0, all_outcomes=True)
        m2 = qml.counts(wires=0, all_outcomes=False)
        with pytest.raises(AssertionError, match="all_outcomes flags are different"):
            qml.assert_equal(m1, m2)

    def test_shadow_expval_different_k(self):
        """Test ShadowExpvalMP k difference."""
        H = qml.X(0)
        m1 = qml.shadow_expval(H=H, k=2)
        m2 = qml.shadow_expval(H=H, k=3)
        with pytest.raises(AssertionError, match="k values are different"):
            qml.assert_equal(m1, m2)

    def test_shadow_expval_different_H_type(self):
        """Test ShadowExpvalMP H type difference (op vs list)."""
        H = qml.X(0)
        m1 = qml.shadow_expval(H=H)
        m2 = qml.shadow_expval(H=[H])
        with pytest.raises(AssertionError, match="Hamiltonian types are incompatible"):
            qml.assert_equal(m1, m2)

    def test_shadow_expval_different_H_list_content(self):
        """Test ShadowExpvalMP H list content difference."""
        H1 = [qml.X(0)]
        H2 = [qml.Y(0)]
        m1 = qml.shadow_expval(H=H1, k=2)
        m2 = qml.shadow_expval(H=H2, k=2)
        with pytest.raises(AssertionError, match="Hamiltonian lists differ at index 0 because op1 and op2 are of different types"):
            qml.assert_equal(m1, m2)
