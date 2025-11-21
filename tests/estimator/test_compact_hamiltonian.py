# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Test the construction of the various compact Hamiltonian representation data classes.
"""
from collections import defaultdict

import pytest

import pennylane.estimator as qre
from pennylane.estimator.compact_hamiltonian import (
    _pauli_dist_from_commuting_groups,
    _sort_and_freeze,
    _validate_pauli_dist,
)

# pylint: too-many-arguments


class TestPauliHamiltonian:
    """Unit tests for the PauliHamiltonian class"""

    @pytest.mark.parametrize(
        "num_qubits, num_pauli_words, max_weight, one_norm",
        (
            (5, 10, 2, 0.01),
            (10, 1000, 5, None),
            (10, 50, None, 10.5),
        ),
    )
    def test_init_from_max_weight_and_num_pauli_words(
        self, num_qubits, num_pauli_words, max_weight, one_norm
    ):
        """Test that we can instantiate a PauliHamiltonian from max_weight and num_pauli_words"""
        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            num_pauli_words=num_pauli_words,
            max_weight=max_weight,
            one_norm=one_norm,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.num_pauli_words == num_pauli_words
        assert pauli_ham.one_norm == one_norm
        assert pauli_ham.pauli_dist is None
        assert pauli_ham.commuting_groups is None

        if max_weight:
            assert pauli_ham.max_weight == max_weight
        else:
            assert pauli_ham.max_weight == num_qubits

    @pytest.mark.parametrize(
        "num_qubits, pauli_dist, num_pauli_words, max_weight, one_norm",
        (
            (5, {"XX": 10, "YY": 10, "Z": 5}, 25, 2, 0.01),
            (10, {"X": 20, "YZY": 7, "ZXX": 3}, 30, 3, None),
            (50, {"XXXXX": 10, "Z": 5}, 15, 5, 10.5),
        ),
    )
    def test_init_from_pauli_dist(
        self, num_qubits, num_pauli_words, max_weight, pauli_dist, one_norm
    ):
        """Test that we can instantiate a PauliHamiltonian from pauli_dist"""
        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            pauli_dist=pauli_dist,
            one_norm=one_norm,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.pauli_dist == pauli_dist
        assert pauli_ham.num_pauli_words == num_pauli_words
        assert pauli_ham.max_weight == max_weight
        assert pauli_ham.one_norm == one_norm
        assert pauli_ham.commuting_groups is None

    @pytest.mark.parametrize(
        "num_qubits, pauli_dist, expected_num_pauli_words, expected_max_weight",
        (
            (5, {"XX": 10, "YY": 10, "Z": 5}, 25, 2),
            (10, {"X": 20, "YZY": 7, "ZXX": 3}, 30, 3),
            (50, {"XXXXX": 10, "Z": 5}, 15, 5),
        ),
    )
    def test_init_from_pauli_dist_overrides_max_weight_and_num_pauli_words(
        self, num_qubits, expected_num_pauli_words, expected_max_weight, pauli_dist
    ):
        """Test that instantiating a PauliHamiltonian from pauli_dist overrides the
        values passed for max_weight and num_pauli_words"""
        overriden_max_weight = 1  # arbitrary value
        overriden_num_pauli_words = 100000  # arbitrary value

        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            num_pauli_words=overriden_num_pauli_words,
            max_weight=overriden_max_weight,
            pauli_dist=pauli_dist,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.pauli_dist == pauli_dist
        assert pauli_ham.commuting_groups is None

        assert pauli_ham.max_weight != overriden_max_weight
        assert pauli_ham.num_pauli_words != overriden_num_pauli_words

        assert pauli_ham.max_weight == expected_max_weight
        assert pauli_ham.num_pauli_words == expected_num_pauli_words

    @pytest.mark.parametrize(
        "num_qubits, commuting_groups, pauli_dist, num_pauli_words, max_weight, one_norm",
        (
            (
                5,
                (
                    {"XX": 10, "X": 5},
                    {"ZZ": 10},
                ),
                {"XX": 10, "X": 5, "ZZ": 10},
                25,
                2,
                0.01,
            ),
            (
                10,
                (
                    {"XX": 15, "X": 5},
                    {"ZZ": 10},
                    {"YY": 5, "X": 5},
                ),
                {"XX": 15, "X": 10, "ZZ": 10, "YY": 5},
                40,
                2,
                None,
            ),
            (
                50,
                ({"XXXXX": 10, "X": 3},),
                {"XXXXX": 10, "X": 3},
                13,
                5,
                10.5,
            ),
        ),
    )
    def test_init_from_commuting_groups(
        self, num_qubits, num_pauli_words, max_weight, pauli_dist, commuting_groups, one_norm
    ):
        """Test that we can instantiate a PauliHamiltonian from commuting_groups"""
        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            commuting_groups=commuting_groups,
            one_norm=one_norm,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.commuting_groups == commuting_groups
        assert pauli_ham.num_pauli_words == num_pauli_words
        assert pauli_ham.max_weight == max_weight
        assert pauli_ham.pauli_dist == pauli_dist
        assert pauli_ham.one_norm == one_norm

    @pytest.mark.parametrize(
        "num_qubits, commuting_groups, expected_pauli_dist, expected_num_pauli_words, expected_max_weight",
        (
            (
                5,
                (
                    {"XX": 10, "X": 5},
                    {"ZZ": 10},
                ),
                {"XX": 10, "X": 5, "ZZ": 10},
                25,
                2,
            ),
            (
                10,
                (
                    {"XX": 15, "X": 5},
                    {"ZZ": 10},
                    {"YY": 5, "X": 5},
                ),
                {"XX": 15, "X": 10, "ZZ": 10, "YY": 5},
                40,
                2,
            ),
            (
                50,
                ({"XXXXX": 10, "X": 3},),
                {"XXXXX": 10, "X": 3},
                13,
                5,
            ),
        ),
    )
    def test_init_from_commuting_groups_overrides_all_other_args(
        self,
        num_qubits,
        expected_num_pauli_words,
        expected_max_weight,
        expected_pauli_dist,
        commuting_groups,
    ):
        """Test that we can instantiate a PauliHamiltonian from commuting_groups overrides the
        values passed for pauli_dist, max_weight and num_pauli_words"""
        overriden_max_weight = 1  # arbitrary value
        overriden_num_pauli_words = 100000  # arbitrary value
        overriden_pauli_dist = {"XXX": 1000000, "ZZZZZZZZ": 100000000}  # arbitrary value

        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            num_pauli_words=overriden_num_pauli_words,
            max_weight=overriden_max_weight,
            pauli_dist=overriden_pauli_dist,
            commuting_groups=commuting_groups,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.commuting_groups == commuting_groups

        assert pauli_ham.num_pauli_words != overriden_num_pauli_words
        assert pauli_ham.max_weight != overriden_max_weight
        assert pauli_ham.pauli_dist != overriden_pauli_dist

        assert pauli_ham.num_pauli_words == expected_num_pauli_words
        assert pauli_ham.max_weight == expected_max_weight
        assert pauli_ham.pauli_dist == expected_pauli_dist

    @pytest.mark.parametrize(
        "input_args, error_message",
        (
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": 100,
                    "max_weight": 30,
                    "pauli_dist": None,
                    "commuting_groups": None,
                },
                "`max_weight` represents the maximum number of qubits",
            ),
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": None,
                    "max_weight": None,
                    "pauli_dist": None,
                    "commuting_groups": None,
                },
                "One of the following sets of inputs must be provided",
            ),
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": 100,
                    "max_weight": 3,
                    "pauli_dist": None,
                    "commuting_groups": ({"XX": "Wrong"}, {"YY": False}, {"ZZZ": 1.5}),
                },
                "The values represent frequencies and should be positive integers",
            ),
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": 100,
                    "max_weight": 3,
                    "pauli_dist": {
                        "XX": "Wrong",
                        "YY": False,
                        "ZZZ": 1.5,
                    },
                    "commuting_groups": None,
                },
                "The values represent frequencies and should be positive integers",
            ),
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": 100,
                    "max_weight": 3,
                    "pauli_dist": {
                        "Wrong": 30,
                        False: 30,
                        5: 40,
                    },
                    "commuting_groups": None,
                },
                "The keys represent Pauli words and should be strings",
            ),
            (
                {
                    "num_qubits": 5,
                    "num_pauli_words": 100,
                    "max_weight": 3,
                    "pauli_dist": None,
                    "commuting_groups": ({"XX": 30}, {False: 30}, {"Wrong": 40}),
                },
                "The keys represent Pauli words and should be strings",
            ),
        ),
    )
    def test_init_raises_error_from_incompatible_inputs(self, input_args, error_message):
        """Test that passing incompatible input arguments will raise the appropriate error"""
        with pytest.raises(ValueError, match=error_message):
            qre.PauliHamiltonian(**input_args)

    @pytest.mark.parametrize(
        "pauli_ham, expected_repr",
        (
            (
                qre.PauliHamiltonian(5, 100, 3),
                "PauliHamiltonian(num_qubits=5, num_pauli_words=100, max_weight=3, one_norm=None)",
            ),
            (
                qre.PauliHamiltonian(5, 100, one_norm=0.01),
                "PauliHamiltonian(num_qubits=5, num_pauli_words=100, max_weight=5, one_norm=0.01)",
            ),
            (
                qre.PauliHamiltonian(10, pauli_dist={"X": 20, "YZY": 7, "ZXX": 3}),
                "PauliHamiltonian(num_qubits=10, num_pauli_words=30, max_weight=3, one_norm=None)",
            ),
            (
                qre.PauliHamiltonian(
                    15,
                    commuting_groups=(
                        {"XX": 15, "X": 5},
                        {"ZZ": 10},
                        {"YY": 5, "X": 5},
                    ),
                ),
                "PauliHamiltonian(num_qubits=15, num_pauli_words=40, max_weight=2, one_norm=None)",
            ),
        ),
    )
    def test_repr(self, pauli_ham, expected_repr):
        """Test the repr dundar method of the PauliHamiltonian"""
        assert repr(pauli_ham) == expected_repr

    @pytest.mark.parametrize(
        "input_args",
        (
            {
                "num_qubits": 10,
                "one_norm": 0.01,
                "num_pauli_words": 100,
                "max_weight": 3,
            },
            {
                "num_qubits": 10,
                "one_norm": None,
                "num_pauli_words": 100,
                "max_weight": 3,
            },
            {
                "num_qubits": 10,
                "one_norm": 10.5,
                "pauli_dist": {"XX": 5, "YY": 10, "Z": 1},
            },
            {
                "num_qubits": 10,
                "one_norm": 10.5,
                "commuting_groups": ({"XX": 5}, {"YY": 10}, {"Z": 1}),
            },
        ),
    )
    def test_hash_and_equality(self, input_args):
        """Test the hash method works as expected"""
        ph1 = qre.PauliHamiltonian(**input_args)
        ph2 = qre.PauliHamiltonian(**input_args)
        ph3 = qre.PauliHamiltonian(num_qubits=5, num_pauli_words=5)  # some other PauliHamiltonian

        assert ph1 == ph2
        assert ph1 != ph3
        assert hash(ph1) == hash(ph1)
        assert hash(ph1) == hash(ph2)

    def test_hash_and_equality_commuting_groups(self):
        """Test that hash and equality are dependant on
        the order of groups in the commuting_groups tuple"""
        ph1 = qre.PauliHamiltonian(
            num_qubits=10,
            commuting_groups=({"XX": 5, "X": 1}, {"YY": 10}, {"Z": 1}),
        )
        ph2 = qre.PauliHamiltonian(
            num_qubits=10,
            commuting_groups=({"X": 1, "XX": 5}, {"YY": 10}, {"Z": 1}),
        )
        ph3 = qre.PauliHamiltonian(
            num_qubits=10,
            commuting_groups=({"Z": 1}, {"XX": 5, "X": 1}, {"YY": 10}),
        )

        assert ph1 == ph2
        assert hash(ph1) == hash(ph2)
        assert ph1 != ph3
        assert hash(ph1) != hash(ph3)

    def test_hash_and_equality_pauli_dist(self):
        """Test that hash and equality are independant of
        the order of terms in the pauli_dist dictionary"""
        ph1 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_dist={"XX": 5, "YY": 10, "Z": 1},
        )
        ph2 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_dist={"YY": 10, "Z": 1, "XX": 5},
        )

        assert ph1 == ph2
        assert hash(ph1) == hash(ph2)


@pytest.mark.parametrize(
    "pauli_dist, error_message",
    (
        (
            {"XX": 10, "YY": 5, "Z": 5, "XZYX": 103},
            None,
        ),
        (
            {"XX": 10, "YY": 5, "Z": 5, "XZYX": -103},
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10, "YY": 5, "Z": True, "XZYX": 103},
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10, "YY": "Five", "Z": 5, "XZYX": 103},
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10.0, "YY": 5.5, "Z": 5.1, "XZYX": 103.2},
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"IXXI": 10, "Wrong": 5, "Characters": 5, "XZYX": 103},
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {"xx": 10, "yy": 5, "z": 5, "XzyX": 103},
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {False: 10, "YY": 5, "Z": 5, "XZYX": 103},
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {"XX": 10, 15: 5, "Z": 5, "XZYX": 103},
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
    ),
)
def test_validate_pauli_dist(pauli_dist, error_message):
    """Test the private _validate_pauli_dist function"""
    if error_message is None:
        # No error message --> valid pauli distribution format
        assert _validate_pauli_dist(pauli_dist) is None
    else:
        with pytest.raises(ValueError, match=error_message):
            _validate_pauli_dist(pauli_dist)


@pytest.mark.parametrize(
    "commuting_groups, expected_pauli_dist",
    (
        (
            (
                {"XX": 10, "X": 5},
                {"ZZ": 10},
            ),
            defaultdict(int, {"XX": 10, "X": 5, "ZZ": 10}),
        ),
        (
            (
                {"XX": 15, "X": 5},
                {"ZZ": 10},
                {"YY": 5, "X": 5},
            ),
            defaultdict(int, {"XX": 15, "X": 10, "ZZ": 10, "YY": 5}),
        ),
        (
            ({"XXXXX": 10, "X": 3},),
            defaultdict(int, {"XXXXX": 10, "X": 3}),
        ),
    ),
)
def test_pauli_dist_from_commuting_groups(commuting_groups, expected_pauli_dist):
    """Test the private _pauli_dist_from_commuting_groups function"""
    assert _pauli_dist_from_commuting_groups(commuting_groups) == expected_pauli_dist


def test_sort_and_freeze():
    """Test the private sort and freeze function behaves as expected"""
    pauli_dist = {
        "XX": 30,
        "ZZ": 76,
        "X": 10,
        "YXY": 105,
        "YY": 15,
    }

    expected_result = (
        ("X", 10),
        ("XX", 30),
        ("YXY", 105),
        ("YY", 15),
        ("ZZ", 76),
    )
    assert _sort_and_freeze(pauli_dist) == expected_result
