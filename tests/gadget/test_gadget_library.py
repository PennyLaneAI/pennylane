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

"""Unit tests for the worked examples in ``pennylane.gadget.library``."""

import numpy as np
import pytest

from pennylane import gadget
from pennylane.gadget.library import rep_chain, rep_code_zz_merge, repetition_code, steane_code


def test_rep_chain_inside_frame():
    """Test that a chain placed at an offset has nearest-neighbour Z checks and logicals on
    its own qubits only."""
    hz, lx, lz = rep_chain(3, offset=2, n_frame=6)
    assert hz.tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0]]
    assert lx.tolist() == [[0, 0, 1, 1, 1, 0]]
    assert lz.tolist() == [[0, 0, 1, 0, 0, 0]]


@pytest.mark.parametrize("d", [3, 5])
def test_repetition_code(d):
    """Test the distance-d repetition code."""
    code = repetition_code(d)
    assert (code.n, code.k) == (d, 1)
    assert code.distance.value == d
    assert code.distance.certified


def test_steane_code_is_self_dual():
    """Test that the Steane code uses the same checks for X and Z."""
    code = steane_code()
    assert np.array_equal(code.hx, code.hz)
    assert code.distance.value == 3


def test_steane_memory_shape(steane_mem):
    """Test that the memory gadget is a single-phase idle gadget with one round window."""
    code, phases, defn = steane_mem
    assert isinstance(defn, gadget.TracedGadget)
    assert [p.name for p in phases] == ["steane"]
    assert str(defn.program.action) == "idle"
    assert defn.program.code is code
    assert defn.program.total_rounds == 3
    assert [r.name for r in defn.records] == ["mem"]


@pytest.mark.parametrize("d", [3, 5])
def test_rep_code_zz_merge_phases(d):
    """Test that the merged phase adds exactly one join check between the two blocks."""
    code, (base, merged), _ = rep_code_zz_merge(d=d)
    assert code.n == 2 * d
    assert merged.syndrome_width == base.syndrome_width + 1
    assert np.nonzero(merged.hz[-1])[0].tolist() == [d - 1, d]
    assert (base.k, merged.k) == (2, 1)


@pytest.mark.parametrize("merged_rounds, claimed", [(1, 1), (2, 2), (3, 3), (5, 3)])
def test_rep_code_zz_merge_claim(merged_rounds, claimed):
    """Test that the merge claims min(d, merged rounds), uncertified."""
    _, _, defn = rep_code_zz_merge(d=3, merged_rounds=merged_rounds)
    (claim,) = defn.program.claims
    assert claim.value == claimed
    assert claim.regime == "phenomenological"
    assert not claim.certified


def test_rep_code_zz_merge_round_windows():
    """Test that merged rounds default to d and pre rounds are configurable."""
    _, _, defn = rep_code_zz_merge(d=3, pre_rounds=2)
    rec = {r.name: r.rounds for r in defn.records}
    assert rec == {"pre": 2, "merged": 3, "post": 1}
