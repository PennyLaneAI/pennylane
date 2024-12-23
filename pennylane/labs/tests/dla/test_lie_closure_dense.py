# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/lie_closure_dense.py functionality"""
# pylint: disable=too-few-public-methods, protected-access, no-self-use
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.labs.dla import lie_closure_dense
from pennylane.pauli import PauliSentence, PauliVSpace, PauliWord

XX = PauliWord({0: "X", 1: "X"})
YY = PauliWord({0: "Y", 1: "Y"})
ops1 = [
    PauliSentence({XX: 1.0, YY: 1.0}),
    PauliSentence({XX: 1.0}),
    PauliSentence({YY: 2.0}),
]

ops2 = [
    PauliSentence({XX: 1.0, YY: 1.0}),
    PauliSentence({XX: 1.0}),
]

ops2plusY10 = ops2 + [PauliSentence({PauliWord({10: "Y"}): 1.0})]


dla11 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence({PauliWord({0: "Z"}): 1.0}),
    PauliSentence({PauliWord({1: "Z"}): 1.0}),
    PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
]


class TestLieClosureDense:
    """Tests for lie_closure_dense()"""

    def test_verbose_true(self, capsys):
        """Test the verbose output"""
        gen11 = dla11[:-1]
        _ = lie_closure_dense(gen11, verbose=True)
        captured = capsys.readouterr()
        assert "epoch 1 of lie_closure_dense, DLA size is 3" in captured.out
        assert "epoch 2 of lie_closure_dense, DLA size is 4" in captured.out
        assert "After 2 epochs, reached a DLA size of 4" in captured.out

    def test_verbose_false(self, capsys):
        """Test that there is no output when verbose is False"""
        gen11 = dla11[:-1]
        _ = lie_closure_dense(gen11, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_max_iterations(self, capsys):
        """Test that max_iterations truncates the lie closure iteration at the right point"""
        n = 3
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        with pytest.warns(UserWarning, match="reached the maximum number of iterations"):
            res = lie_closure_dense(generators, verbose=True, max_iterations=1)

        captured = capsys.readouterr()
        assert (
            captured.out
            == "epoch 1 of lie_closure_dense, DLA size is 4\nAfter 1 epochs, reached a DLA size of 8\n"
        )
        assert len(res) == 8

    def test_simple_lie_closure_dense(self):
        """Test simple lie_closure_dense example"""

        dla12 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -2.0, PauliWord({1: "Z"}): 2.0}),
        ]

        gen12 = dla12[:-1]
        res12 = lie_closure_dense(gen12)
        res12 = [qml.pauli_decompose(op) for op in res12]
        assert qml.pauli.PauliVSpace(res12) == qml.pauli.PauliVSpace(dla12)

    def test_lie_closure_dense_with_pl_ops(self):
        """Test that lie_closure_dense works properly with PennyLane ops instead of PauliSentences"""
        dla = [
            qml.sum(qml.prod(X(0), X(1)), qml.prod(Y(0), Y(1))),
            Z(0),
            Z(1),
            qml.sum(qml.prod(Y(0), X(1)), qml.s_prod(-1.0, qml.prod(X(0), Y(1)))),
        ]
        gen11 = dla[:-1]
        res11 = lie_closure_dense(gen11)

        res11 = [qml.pauli_decompose(op) for op in res11]  # back to pauli_rep for easier comparison
        assert PauliVSpace(res11) == PauliVSpace(dla11)

    def test_lie_closure_dense_with_ndarrays(self):
        """Test that lie_closure_dense works properly with ndarray inputs"""
        dla = [
            qml.sum(qml.prod(X(0), X(1)), qml.prod(Y(0), Y(1))),
            Z(0),
            Z(1),
            qml.sum(qml.prod(Y(0), X(1)), qml.s_prod(-1.0, qml.prod(X(0), Y(1)))),
        ]
        dla = [qml.matrix(op, wire_order=range(2)) for op in dla]
        gen11 = dla[:-1]
        res11 = lie_closure_dense(gen11)

        res11 = [qml.pauli_decompose(op) for op in res11]  # back to pauli_rep for easier comparison
        assert PauliVSpace(res11) == PauliVSpace(dla11)

    def test_lie_closure_dense_with_PauliWords(self):
        """Test that lie_closure_dense works properly with PauliWords"""
        gen = [
            PauliWord({0: "X", 1: "X"}),
            PauliWord({0: "Z"}),
            PauliWord({1: "Z"}),
        ]
        dla = gen + [
            PauliWord({0: "Y", 1: "X"}),
            PauliWord({0: "X", 1: "Y"}),
            PauliWord({0: "Y", 1: "Y"}),
        ]
        dla = [op.pauli_rep for op in dla]

        res = lie_closure_dense(gen)

        res = [qml.pauli_decompose(op) for op in res]  # convert to pauli_rep for easier comparison
        assert PauliVSpace(res) == PauliVSpace(dla)

    def test_lie_closure_dense_with_sentences(self):
        """Test that lie_closure_dense returns the correct results when using (true) PauliSentences, i.e. not just single PauliWords"""
        n = 3
        gen = [
            PauliSentence(
                {
                    PauliWord({i: "X", (i + 1) % n: "X"}): 1.0,
                    PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0,
                }
            )
            for i in range(n - 1)
        ]
        gen += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(n)]

        res = lie_closure_dense(gen)
        res = [qml.pauli_decompose(op) for op in res]
        true_res = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({1: "X", 2: "X"}): 1.0, PauliWord({1: "Y", 2: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({1: "Y", 2: "X"}): -1.0, PauliWord({1: "X", 2: "Y"}): 1.0}),
            PauliSentence(
                {
                    PauliWord({0: "X", 1: "Z", 2: "Y"}): 1.0,
                    PauliWord({0: "Y", 1: "Z", 2: "X"}): -1.0,
                }
            ),
            PauliSentence(
                {
                    PauliWord({0: "X", 1: "Z", 2: "X"}): -1.0,
                    PauliWord({0: "Y", 1: "Z", 2: "Y"}): -1.0,
                }
            ),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({1: "Z"}): 1.0}),
            PauliSentence({PauliWord({2: "Z"}): 1.0}),
        ]
        assert PauliVSpace(res) == PauliVSpace(true_res)

    @pytest.mark.parametrize("n", range(2, 5))
    def test_lie_closure_dense_transverse_field_ising_1D_open(self, n):
        """Test the lie closure works correctly for the transverse Field Ising model with open boundary conditions,
        a8 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = lie_closure_dense(generators)
        assert len(res) == (2 * n - 1) * (2 * n - 2) // 2

    def test_lie_closure_dense_transverse_field_ising_1D_cyclic(self):
        """Test the lie closure works correctly for the transverse Field Ising model with cyclic boundary conditions,
        a8 in theorem IV.2 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 3
        generators = [PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n)]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n)
        ]

        res = lie_closure_dense(generators)
        assert len(res) == 2 * n * (2 * n - 1)

    def test_lie_closure_dense_heisenberg_generators_odd(self):
        """Test the resulting DLA from Heisenberg generators with odd n,
        a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 3
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates su(2**(n-1)) for n odd            => dim should be (2**(n-1))**2 - 1
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = lie_closure_dense(generators)
        assert len(res) == (2 ** (n - 1)) ** 2 - 1

    def test_lie_closure_dense_heisenberg_generators_even(self):
        """Test the resulting DLA from Heisenberg generators with even n,
        a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 4
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates (su(2**(n-2)))**4 for n>=4 even   => dim should be 4*((2**(n-2))**2 - 1)
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = lie_closure_dense(generators)
        assert len(res) == 4 * ((2 ** (n - 2)) ** 2 - 1)

    @pytest.mark.parametrize("n, res", [(3, 4), (4, 12)])
    def test_lie_closure_dense_heisenberg_isotropic(self, n, res):
        """Test the resulting DLA from Heisenberg model with summed generators"""
        genXX = [X(i) @ X(i + 1) for i in range(n - 1)]
        genYY = [Y(i) @ Y(i + 1) for i in range(n - 1)]
        genZZ = [Z(i) @ Z(i + 1) for i in range(n - 1)]

        generators = [qml.sum(XX + YY + ZZ) for XX, YY, ZZ in zip(genXX, genYY, genZZ)]
        g = lie_closure_dense(generators)
        assert len(g) == res

    def test_universal_gate_set(self):
        """Test universal gate set"""
        n = 3

        generators = [Z(i) for i in range(n)]
        generators += [Y(i) for i in range(n)]
        generators += [
            (I(i) - Z(i)) @ (I(i + 1) - X(i + 1)) for i in range(n - 1)
        ]  # generator of CNOT gate

        vspace = lie_closure_dense(generators)

        assert len(vspace) == 4**3
