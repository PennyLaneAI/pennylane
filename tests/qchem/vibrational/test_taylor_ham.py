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
"""Unit Tests for the taylor hamiltonian construction functions."""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.bose import BoseWord
from pennylane.qchem import vibrational
from pennylane.qchem.vibrational.taylor_ham import (
    _fit_onebody,
    _fit_threebody,
    _fit_twobody,
    _remove_harmonic,
    _taylor_anharmonic,
    _taylor_harmonic,
    _taylor_kinetic,
    _threebody_degs,
    _twobody_degs,
    taylor_bosonic,
    taylor_coeffs,
    taylor_dipole_coeffs,
    taylor_hamiltonian,
)
from tests.qchem.vibrational.test_ref_files.pes_object import (
    expected_coeffs_x_arr,
    expected_coeffs_y_arr,
    expected_coeffs_z_arr,
    freqs,
    pes_object_3D,
    reference_taylor_bosonic_coeffs,
    reference_taylor_bosonic_ops,
    taylor_1D,
    taylor_2D,
    taylor_3D,
    uloc,
)

for i, ele in enumerate(reference_taylor_bosonic_ops):
    reference_taylor_bosonic_ops[i] = BoseWord(ele)


def test_import_sklearn(monkeypatch):
    """Test if an ImportError is raised by _import_sklearn function."""
    # pylint: disable=protected-access

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "sklearn", None)

        with pytest.raises(ImportError, match="This feature requires sklearn"):
            vibrational.taylor_ham._import_sklearn()


def test_taylor_anharmonic():
    """Test that taylor_anharmonic produces the correct anharmonic term of the hamiltonian"""
    expected_anh_ham = [
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+"}), -1.5818170215014748e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-"}), -4.7454510645044245e-05),
        (BoseWord({(0, 0): "+"}), 4.177931609625755e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-"}), -4.7454510645044245e-05),
        (BoseWord({(0, 0): "-"}), 4.177931609625755e-05),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "-"}), -1.5818170215014748e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+"}), -0.00011795343849999994),
        (BoseWord({(0, 0): "+", (1, 0): "-"}), -0.00023590687699999988),
        (BoseWord({}), 9.627115513999998e-05),
        (BoseWord({(0, 0): "-", (1, 0): "-"}), -0.00011795343849999994),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 0): "+"}), -2.7719982249999993e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 0): "-"}), -1.1087992899999997e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-", (3, 0): "-"}), -1.6631989349999996e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-", (3, 0): "-"}), -1.1087992899999997e-05),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "-", (3, 0): "-"}), -2.7719982249999993e-06),
        (BoseWord({(0, 1): "+", (1, 1): "+"}), 0.00018836178121499995),
        (BoseWord({(0, 1): "+", (1, 1): "-"}), 0.0003767235624299999),
        (BoseWord({(0, 1): "-", (1, 1): "-"}), 0.00018836178121499995),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+"}), -0.000491383188854377),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "-"}), -0.001474149566563131),
        (BoseWord({(0, 1): "+"}), -0.0013076515361278378),
        (BoseWord({(0, 1): "+", (1, 1): "-", (2, 1): "-"}), -0.001474149566563131),
        (BoseWord({(0, 1): "-"}), -0.0013076515361278378),
        (BoseWord({(0, 1): "-", (1, 1): "-", (2, 1): "-"}), -0.000491383188854377),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+", (3, 1): "+"}), 4.6117788749999985e-05),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+", (3, 1): "-"}), 0.00018447115499999994),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "-", (3, 1): "-"}), 0.0002767067324999999),
        (BoseWord({(0, 1): "+", (1, 1): "-", (2, 1): "-", (3, 1): "-"}), 0.00018447115499999994),
        (BoseWord({(0, 1): "-", (1, 1): "-", (2, 1): "-", (3, 1): "-"}), 4.6117788749999985e-05),
        (BoseWord({(0, 2): "+", (1, 2): "+"}), 0.00018836171996499998),
        (BoseWord({(0, 2): "+", (1, 2): "-"}), 0.00037672343992999996),
        (BoseWord({(0, 2): "-", (1, 2): "-"}), 0.00018836171996499998),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "+"}), -0.0004913831817833092),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "-"}), -0.0014741495453499277),
        (BoseWord({(0, 2): "+"}), -0.0013076514536084765),
        (BoseWord({(0, 2): "+", (1, 2): "-", (2, 2): "-"}), -0.0014741495453499277),
        (BoseWord({(0, 2): "-"}), -0.0013076514536084765),
        (BoseWord({(0, 2): "-", (1, 2): "-", (2, 2): "-"}), -0.0004913831817833092),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "+", (3, 2): "+"}), 4.611778774999999e-05),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "+", (3, 2): "-"}), 0.00018447115099999996),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "-", (3, 2): "-"}), 0.0002767067264999999),
        (BoseWord({(0, 2): "+", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), 0.00018447115099999996),
        (BoseWord({(0, 2): "-", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), 4.611778774999999e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+"}), -3.1160963125e-05),
        (BoseWord({(0, 1): "+", (1, 0): "-"}), -3.1160963125e-05),
        (BoseWord({(0, 0): "+", (1, 1): "-"}), -3.1160963125e-05),
        (BoseWord({(0, 0): "-", (1, 1): "-"}), -3.1160963125e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "+"}), 4.461690665313647e-05),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 0): "-"}), 4.461690665313647e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "-"}), 8.923381330627295e-05),
        (BoseWord({(0, 1): "+", (1, 0): "-", (2, 1): "-"}), 8.923381330627295e-05),
        (BoseWord({(0, 0): "+", (1, 1): "-", (2, 1): "-"}), 4.461690665313647e-05),
        (BoseWord({(0, 0): "-", (1, 1): "-", (2, 1): "-"}), 4.461690665313647e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "+"}), 0.0001624853534245611),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 0): "-"}), 0.0003249707068491222),
        (BoseWord({(0, 1): "+", (1, 0): "-", (2, 0): "-"}), 0.0001624853534245611),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "-"}), 0.0001624853534245611),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 1): "-"}), 0.0003249707068491222),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 1): "-"}), 0.0001624853534245611),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "+", (3, 1): "+"}), -7.978761674999999e-06),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+", (3, 0): "-"}), -7.978761674999999e-06),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "+", (3, 1): "-"}), -2.3936285024999997e-05),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 0): "-", (3, 1): "-"}), -2.3936285024999997e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "-", (3, 1): "-"}), -2.3936285024999997e-05),
        (BoseWord({(0, 1): "+", (1, 0): "-", (2, 1): "-", (3, 1): "-"}), -2.3936285024999997e-05),
        (BoseWord({(0, 0): "+", (1, 1): "-", (2, 1): "-", (3, 1): "-"}), -7.978761674999999e-06),
        (BoseWord({(0, 0): "-", (1, 1): "-", (2, 1): "-", (3, 1): "-"}), -7.978761674999999e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "+", (3, 1): "+"}), -5.318648924999997e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 1): "+", (3, 0): "-"}), -0.00010637297849999994),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 0): "-", (3, 0): "-"}), -5.318648924999997e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "+", (3, 1): "-"}), -0.00010637297849999994),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 0): "-", (3, 1): "-"}), -0.00021274595699999989),
        (BoseWord({(0, 1): "+", (1, 0): "-", (2, 0): "-", (3, 1): "-"}), -0.00010637297849999994),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "-", (3, 1): "-"}), -5.318648924999997e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 1): "-", (3, 1): "-"}), -0.00010637297849999994),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 1): "-", (3, 1): "-"}), -5.318648924999997e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 1): "+"}), 6.450183649999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 1): "+", (3, 0): "-"}), 1.9350550949999994e-05),
        (BoseWord({(0, 0): "+", (1, 1): "+", (2, 0): "-", (3, 0): "-"}), 1.9350550949999994e-05),
        (BoseWord({(0, 1): "+", (1, 0): "-", (2, 0): "-", (3, 0): "-"}), 6.450183649999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 1): "-"}), 6.450183649999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-", (3, 1): "-"}), 1.9350550949999994e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-", (3, 1): "-"}), 1.9350550949999994e-05),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "-", (3, 1): "-"}), 6.450183649999998e-06),
        (BoseWord({(0, 0): "+", (1, 2): "+"}), -3.1161011174999994e-05),
        (BoseWord({(0, 2): "+", (1, 0): "-"}), -3.1161011174999994e-05),
        (BoseWord({(0, 0): "+", (1, 2): "-"}), -3.1161011174999994e-05),
        (BoseWord({(0, 0): "-", (1, 2): "-"}), -3.1161011174999994e-05),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "+"}), 4.461692008816532e-05),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 0): "-"}), 4.461692008816532e-05),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "-"}), 8.923384017633064e-05),
        (BoseWord({(0, 2): "+", (1, 0): "-", (2, 2): "-"}), 8.923384017633064e-05),
        (BoseWord({(0, 0): "+", (1, 2): "-", (2, 2): "-"}), 4.461692008816532e-05),
        (BoseWord({(0, 0): "-", (1, 2): "-", (2, 2): "-"}), 4.461692008816532e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+"}), 0.00016248534776770686),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 0): "-"}), 0.0003249706955354137),
        (BoseWord({(0, 2): "+", (1, 0): "-", (2, 0): "-"}), 0.00016248534776770686),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "-"}), 0.00016248534776770686),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 2): "-"}), 0.0003249706955354137),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 2): "-"}), 0.00016248534776770686),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "+", (3, 2): "+"}), -7.978763299999997e-06),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "+", (3, 0): "-"}), -7.978763299999997e-06),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "+", (3, 2): "-"}), -2.393628989999999e-05),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 0): "-", (3, 2): "-"}), -2.393628989999999e-05),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "-", (3, 2): "-"}), -2.393628989999999e-05),
        (BoseWord({(0, 2): "+", (1, 0): "-", (2, 2): "-", (3, 2): "-"}), -2.393628989999999e-05),
        (BoseWord({(0, 0): "+", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), -7.978763299999997e-06),
        (BoseWord({(0, 0): "-", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), -7.978763299999997e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+", (3, 2): "+"}), -5.3186488749999974e-05),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 2): "+", (3, 0): "-"}), -0.00010637297749999995),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 0): "-", (3, 0): "-"}), -5.3186488749999974e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+", (3, 2): "-"}), -0.00010637297749999995),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 0): "-", (3, 2): "-"}), -0.0002127459549999999),
        (BoseWord({(0, 2): "+", (1, 0): "-", (2, 0): "-", (3, 2): "-"}), -0.00010637297749999995),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "-", (3, 2): "-"}), -5.3186488749999974e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 2): "-", (3, 2): "-"}), -0.00010637297749999995),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 2): "-", (3, 2): "-"}), -5.3186488749999974e-05),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 2): "+"}), 6.450184574999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+", (3, 0): "-"}), 1.935055372499999e-05),
        (BoseWord({(0, 0): "+", (1, 2): "+", (2, 0): "-", (3, 0): "-"}), 1.935055372499999e-05),
        (BoseWord({(0, 2): "+", (1, 0): "-", (2, 0): "-", (3, 0): "-"}), 6.450184574999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "+", (3, 2): "-"}), 6.450184574999998e-06),
        (BoseWord({(0, 0): "+", (1, 0): "+", (2, 0): "-", (3, 2): "-"}), 1.935055372499999e-05),
        (BoseWord({(0, 0): "+", (1, 0): "-", (2, 0): "-", (3, 2): "-"}), 1.935055372499999e-05),
        (BoseWord({(0, 0): "-", (1, 0): "-", (2, 0): "-", (3, 2): "-"}), 6.450184574999998e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+"}), -6.646252578249999e-05),
        (BoseWord({(0, 2): "+", (1, 1): "-"}), -6.646252578249999e-05),
        (BoseWord({(0, 1): "+", (1, 2): "-"}), -6.646252578249999e-05),
        (BoseWord({(0, 1): "-", (1, 2): "-"}), -6.646252578249999e-05),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "+"}), 4.012677010732053e-06),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 1): "-"}), 4.012677010732053e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "-"}), 8.025354021464107e-06),
        (BoseWord({(0, 2): "+", (1, 1): "-", (2, 2): "-"}), 8.025354021464107e-06),
        (BoseWord({(0, 1): "+", (1, 2): "-", (2, 2): "-"}), 4.012677010732053e-06),
        (BoseWord({(0, 1): "-", (1, 2): "-", (2, 2): "-"}), 4.012677010732053e-06),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "+"}), 4.012743973744231e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 1): "-"}), 8.025487947488462e-06),
        (BoseWord({(0, 2): "+", (1, 1): "-", (2, 1): "-"}), 4.012743973744231e-06),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "-"}), 4.012743973744231e-06),
        (BoseWord({(0, 1): "+", (1, 1): "-", (2, 2): "-"}), 8.025487947488462e-06),
        (BoseWord({(0, 1): "-", (1, 1): "-", (2, 2): "-"}), 4.012743973744231e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "+", (3, 2): "+"}), -4.712907074999999e-07),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 2): "+", (3, 1): "-"}), -4.712907074999999e-07),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "+", (3, 2): "-"}), -1.4138721224999998e-06),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 1): "-", (3, 2): "-"}), -1.4138721224999998e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "-", (3, 2): "-"}), -1.4138721224999998e-06),
        (BoseWord({(0, 2): "+", (1, 1): "-", (2, 2): "-", (3, 2): "-"}), -1.4138721224999998e-06),
        (BoseWord({(0, 1): "+", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), -4.712907074999999e-07),
        (BoseWord({(0, 1): "-", (1, 2): "-", (2, 2): "-", (3, 2): "-"}), -4.712907074999999e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "+", (3, 2): "+"}), 4.811507149999998e-07),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 2): "+", (3, 1): "-"}), 9.623014299999996e-07),
        (BoseWord({(0, 2): "+", (1, 2): "+", (2, 1): "-", (3, 1): "-"}), 4.811507149999998e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "+", (3, 2): "-"}), 9.623014299999996e-07),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 1): "-", (3, 2): "-"}), 1.924602859999999e-06),
        (BoseWord({(0, 2): "+", (1, 1): "-", (2, 1): "-", (3, 2): "-"}), 9.623014299999996e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "-", (3, 2): "-"}), 4.811507149999998e-07),
        (BoseWord({(0, 1): "+", (1, 1): "-", (2, 2): "-", (3, 2): "-"}), 9.623014299999996e-07),
        (BoseWord({(0, 1): "-", (1, 1): "-", (2, 2): "-", (3, 2): "-"}), 4.811507149999998e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+", (3, 2): "+"}), -4.7129871999999985e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 2): "+", (3, 1): "-"}), -1.4138961599999995e-06),
        (BoseWord({(0, 1): "+", (1, 2): "+", (2, 1): "-", (3, 1): "-"}), -1.4138961599999995e-06),
        (BoseWord({(0, 2): "+", (1, 1): "-", (2, 1): "-", (3, 1): "-"}), -4.7129871999999985e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "+", (3, 2): "-"}), -4.7129871999999985e-07),
        (BoseWord({(0, 1): "+", (1, 1): "+", (2, 1): "-", (3, 2): "-"}), -1.4138961599999995e-06),
        (BoseWord({(0, 1): "+", (1, 1): "-", (2, 1): "-", (3, 2): "-"}), -1.4138961599999995e-06),
        (BoseWord({(0, 1): "-", (1, 1): "-", (2, 1): "-", (3, 2): "-"}), -4.7129871999999985e-07),
    ]
    anh_ham = _taylor_anharmonic([taylor_1D, taylor_2D])

    assert expected_anh_ham == list(anh_ham.items())


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_taylor_harmonic():
    """Test that taylor_harmonic produces the correct harmonic term of the hamiltonian"""
    expected_taylor_harm = [
        (BoseWord({(0, 0): "+", (1, 0): "+"}), 0.0014742224999999996),
        (BoseWord({(0, 0): "+", (1, 0): "-"}), 0.002948444999999999),
        (BoseWord({}), 0.007636362499999998),
        (BoseWord({(0, 0): "-", (1, 0): "-"}), 0.0014742224999999996),
        (BoseWord({(0, 1): "+", (1, 1): "+"}), 0.003081069999999999),
        (BoseWord({(0, 1): "+", (1, 1): "-"}), 0.006162139999999998),
        (BoseWord({(0, 1): "-", (1, 1): "-"}), 0.003081069999999999),
        (BoseWord({(0, 2): "+", (1, 2): "+"}), 0.003081069999999999),
        (BoseWord({(0, 2): "+", (1, 2): "-"}), 0.006162139999999998),
        (BoseWord({(0, 2): "-", (1, 2): "-"}), 0.003081069999999999),
    ]
    taylor_harm = _taylor_harmonic([taylor_1D, taylor_2D], freqs)
    assert expected_taylor_harm == list(taylor_harm.items())


def test_taylor_kinetic():
    """Test that taylor_kinetic produces the correct kinetic term of the hamiltonian"""

    expected_taylor_kin = [
        (BoseWord({(0, 0): "+", (1, 0): "+"}), (-0.0014742224999999994 + 0j)),
        (BoseWord({(0, 0): "+", (1, 0): "-"}), (0.0029484449999999988 + 0j)),
        (BoseWord({}), (0.00763636247932089 + 0j)),
        (BoseWord({(0, 0): "-", (1, 0): "-"}), (-0.0014742224999999994 + 0j)),
        (BoseWord({(0, 0): "+", (1, 1): "+"}), 0j),
        (BoseWord({(0, 0): "+", (1, 1): "-"}), 0j),
        (BoseWord({(0, 1): "+", (1, 0): "-"}), 0j),
        (BoseWord({(0, 0): "-", (1, 1): "-"}), 0j),
        (BoseWord({(0, 0): "+", (1, 2): "+"}), 0j),
        (BoseWord({(0, 0): "+", (1, 2): "-"}), 0j),
        (BoseWord({(0, 2): "+", (1, 0): "-"}), 0j),
        (BoseWord({(0, 0): "-", (1, 2): "-"}), 0j),
        (BoseWord({(0, 1): "+", (1, 1): "+"}), (-0.0030810699896604453 + 0j)),
        (BoseWord({(0, 1): "+", (1, 1): "-"}), (0.0061621399793208905 + 0j)),
        (BoseWord({(0, 1): "-", (1, 1): "-"}), (-0.0030810699896604453 + 0j)),
        (BoseWord({(0, 1): "+", (1, 2): "+"}), 0j),
        (BoseWord({(0, 1): "+", (1, 2): "-"}), 0j),
        (BoseWord({(0, 2): "+", (1, 1): "-"}), 0j),
        (BoseWord({(0, 1): "-", (1, 2): "-"}), 0j),
        (BoseWord({(0, 2): "+", (1, 2): "+"}), (-0.0030810699896604453 + 0j)),
        (BoseWord({(0, 2): "+", (1, 2): "-"}), (0.0061621399793208905 + 0j)),
        (BoseWord({(0, 2): "-", (1, 2): "-"}), (-0.0030810699896604453 + 0j)),
    ]
    taylor_kin = _taylor_kinetic([taylor_1D, taylor_2D], freqs, uloc=uloc)
    assert expected_taylor_kin == list(taylor_kin.items())


def test_taylor_bosonic():
    """Test that taylor_bosonic produces the correct bosonic hamiltonian"""
    taylor_bos = taylor_bosonic([taylor_1D, taylor_2D], freqs, uloc=uloc)
    sorted_arr = sorted(taylor_bos.items(), key=lambda x: x[1].real)
    sorted_ops_arr, sorted_coeffs_arr = zip(*sorted_arr)

    assert np.allclose(
        abs(np.array(sorted_coeffs_arr)), abs(reference_taylor_bosonic_coeffs), atol=1e-10
    )
    assert len(sorted_ops_arr) == len(reference_taylor_bosonic_ops)
    assert list(sorted_ops_arr) == reference_taylor_bosonic_ops


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_taylor_hamiltonian():
    """Test that taylor_hamiltonian produces the correct taylor hamiltonian"""
    taylor_ham = taylor_hamiltonian(pes_object_3D, 4, 2)
    taylor_bos = taylor_bosonic([taylor_1D, taylor_2D, taylor_3D], freqs, uloc=uloc)

    assert len(taylor_ham) == len(taylor_bos)
    assert all(
        np.allclose(abs(taylor_bos.get(key)), abs(value), atol=1e-5)
        for key, value in taylor_ham.items()
    )


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_remove_harmonic():
    """Test that _remove_harmonic produces the correct anharmonic pes"""
    anh_pes, _ = _remove_harmonic(freqs=pes_object_3D.freqs, onemode_pes=pes_object_3D.pes_onemode)
    expected_anh_pes = np.array(
        [
            [
                -2.46470337e-03,
                -8.66444424e-04,
                -2.28151453e-04,
                -2.52213033e-05,
                -7.38964445e-13,
                1.98758167e-05,
                1.46472247e-04,
                3.82594976e-04,
                3.59486530e-04,
            ],
            [
                6.39651979e-02,
                1.97795952e-02,
                4.76215949e-03,
                5.06917611e-04,
                -7.38964445e-13,
                -4.18235839e-04,
                -3.21282001e-03,
                -1.07598839e-02,
                -2.70898724e-02,
            ],
            [
                -2.70898725e-02,
                -1.07598839e-02,
                -3.21282001e-03,
                -4.18235859e-04,
                -7.38964445e-13,
                5.06917581e-04,
                4.76215929e-03,
                1.97795947e-02,
                6.39651969e-02,
            ],
        ]
    )

    assert np.allclose(abs(anh_pes), abs(expected_anh_pes), atol=1e-5)


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_fit_onebody():
    """Test that the fitting for two-mode operators is accurate"""
    anh_pes = np.array(
        [
            [
                -2.46470337e-03,
                -8.66444424e-04,
                -2.28151453e-04,
                -2.52213033e-05,
                -7.38964445e-13,
                1.98758167e-05,
                1.46472247e-04,
                3.82594976e-04,
                3.59486530e-04,
            ],
            [
                6.39651979e-02,
                1.97795952e-02,
                4.76215949e-03,
                5.06917611e-04,
                -7.38964445e-13,
                -4.18235839e-04,
                -3.21282001e-03,
                -1.07598839e-02,
                -2.70898724e-02,
            ],
            [
                -2.70898725e-02,
                -1.07598839e-02,
                -3.21282001e-03,
                -4.18235859e-04,
                -7.38964445e-13,
                5.06917581e-04,
                4.76215929e-03,
                1.97795947e-02,
                6.39651969e-02,
            ],
        ]
    )
    coeff_1D, _ = _fit_onebody(anh_pes, 4, 2)

    assert np.allclose(abs(coeff_1D), abs(taylor_1D), atol=1e-6)


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_fit_twobody():
    """Test that the fitting for two-mode operators is accurate"""
    coeff_2D, _ = _fit_twobody(pes_object_3D.pes_twomode, 4, 2)
    assert np.allclose(abs(coeff_2D), abs(taylor_2D), atol=1e-10)


@pytest.mark.usefixtures("skip_if_no_sklearn_support")
def test_fit_threebody():
    """Test that the fitting for three-mode operators is accurate"""
    coeff_3D, _ = _fit_threebody(pes_object_3D.pes_threemode, 4, 2)
    assert np.allclose(abs(coeff_3D), abs(taylor_3D), atol=1e-10)


def test_twomode_degs():
    """Test that _twobody_degs produces the right combinations"""
    expected_degs = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3)]
    fit_degs = _twobody_degs(4, 2)
    assert fit_degs == expected_degs


def test_threemode_degs():
    """Test that _threebody_degs produces the right combinations"""
    expected_degs = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    fit_degs = _threebody_degs(4, 2)
    assert fit_degs == expected_degs


@pytest.mark.usefixtures(
    "skip_if_no_pyscf_support", "skip_if_no_geometric_support", "skip_if_no_sklearn_support"
)
def test_taylor_coeffs():
    """Test that the computer taylor coeffs for Hamiltonian are accurate"""
    taylor_coeffs_1D, taylor_coeffs_2D, _ = taylor_coeffs(pes_object_3D, 4, 2)
    assert np.allclose(abs(taylor_coeffs_1D), abs(taylor_1D), atol=1e-8)
    assert np.allclose(abs(taylor_coeffs_2D), abs(taylor_2D), atol=1e-8)


@pytest.mark.usefixtures(
    "skip_if_no_pyscf_support", "skip_if_no_geometric_support", "skip_if_no_sklearn_support"
)
def test_taylor_coeffs_dipole():
    """Test that the computed taylor coeffs for dipoles are accurate"""
    coeffs_x_arr, coeffs_y_arr, coeffs_z_arr = taylor_dipole_coeffs(pes_object_3D, 4, 1)
    assert np.allclose(coeffs_x_arr[0], expected_coeffs_x_arr[0], atol=1e-10)
    assert np.allclose(coeffs_x_arr[1], expected_coeffs_x_arr[1], atol=1e-10)
    assert np.allclose(abs(coeffs_y_arr[0]), abs(expected_coeffs_y_arr[0]), atol=1e-8)
    assert np.allclose(abs(coeffs_y_arr[1]), abs(expected_coeffs_y_arr[1]), atol=1e-8)
    assert np.allclose(abs(coeffs_z_arr[0]), abs(expected_coeffs_z_arr[0]), atol=1e-8)
    assert np.allclose(abs(coeffs_z_arr[1]), abs(expected_coeffs_z_arr[1]), atol=1e-8)
