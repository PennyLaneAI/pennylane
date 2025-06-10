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
"""Tests for pennylane/dla/lie_closure.py functionality"""
# pylint: disable=too-few-public-methods, protected-access

import numpy as np
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z, lie_closure
from pennylane.pauli import PauliSentence, PauliVSpace, PauliWord

ops1 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
    PauliSentence(
        {
            PauliWord({0: "Y", 1: "Y"}): 2.0,
        }
    ),
]

ops2 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
]

ops2plusY10 = ops2 + [PauliSentence({PauliWord({10: "Y"}): 1.0})]

dla11 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence({PauliWord({0: "Z"}): 1.0}),
    PauliSentence({PauliWord({1: "Z"}): 1.0}),
    PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
]


class TestLieClosure:
    """Tests for qml.lie_closure()"""

    @pytest.mark.parametrize("matrix", [False, True])
    def test_verbose(self, capsys, matrix):
        """Test the verbose output"""
        gen11 = dla11[:-1]
        _ = lie_closure(gen11, verbose=True, matrix=matrix)
        captured = capsys.readouterr()
        assert "epoch 1 of lie_closure, DLA size is 3" in captured.out
        assert "epoch 2 of lie_closure, DLA size is 4" in captured.out
        assert "After 2 epochs, reached a DLA size of 4" in captured.out

    @pytest.mark.parametrize("matrix", [False, True])
    def test_verbose_false(self, capsys, matrix):
        """Test that there is no output when verbose is False"""
        gen11 = dla11[:-1]
        _ = lie_closure(gen11, verbose=False, matrix=matrix)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_pauli_true_wrong_inputs(self):
        """Test that an error with a meaningful error message is raised when inputting the wrong types while using pauli=True"""
        gens = [X(0), X(1), Y(0) @ Y(1)]

        with pytest.raises(TypeError, match="All generators need to be of type PauliSentence"):
            _ = lie_closure(gens, pauli=True)

    @pytest.mark.parametrize("matrix", [False, True])
    def test_max_iterations(self, capsys, matrix):
        """Test that max_iterations truncates the lie closure iteration at the right point"""
        n = 3
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        with pytest.warns(UserWarning, match="reached the maximum number of iterations"):
            res = lie_closure(generators, verbose=True, max_iterations=1, matrix=matrix)

        captured = capsys.readouterr()
        assert (
            captured.out
            == "epoch 1 of lie_closure, DLA size is 4\nAfter 1 epochs, reached a DLA size of 8\n"
        )
        assert len(res) == 8

    def test_simple_lie_closure(self):
        """Test simple lie_closure example"""
        gen11 = dla11[:-1]
        res11 = lie_closure(gen11, pauli=True)
        assert res11 == dla11

        dla12 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -2.0, PauliWord({1: "Z"}): 2.0}),
        ]

        gen12 = dla12[:-1]
        res12 = lie_closure(gen12, pauli=True)
        assert PauliVSpace(res12) == PauliVSpace(dla12)

    def test_lie_closure_with_pl_ops(self):
        """Test that lie_closure works properly with PennyLane ops instead of PauliSentences"""
        dla = [
            qml.sum(qml.prod(X(0), X(1)), qml.prod(Y(0), Y(1))),
            Z(0),
            Z(1),
            qml.sum(qml.prod(Y(0), X(1)), qml.s_prod(-1.0, qml.prod(X(0), Y(1)))),
        ]
        gen11 = dla[:-1]
        res11 = lie_closure(gen11)

        res11 = [op.pauli_rep for op in res11]  # back to pauli_rep for easier comparison
        assert PauliVSpace(res11) == PauliVSpace(dla11)

    @pytest.mark.parametrize("pauli", [True, False])
    def test_lie_closure_with_PauliWords(self, pauli):
        """Test that lie_closure works properly with PauliWords"""
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

        res = lie_closure(gen, pauli=pauli)

        res = [op.pauli_rep for op in res]  # convert to pauli_rep for easier comparison
        assert PauliVSpace(res) == PauliVSpace(dla)

    def test_lie_closure_with_sentences(self):
        """Test that lie_closure returns the correct results when using (true) PauliSentences, i.e. not just single PauliWords"""
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

        res = lie_closure(gen, pauli=True)
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

    @pytest.mark.parametrize("matrix", [False, True])
    @pytest.mark.parametrize("n", range(2, 5))
    def test_lie_closure_transverse_field_ising_1D_open(self, n, matrix):
        """Test the lie closure works correctly for the transverse Field Ising model with open boundary conditions, a8 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = lie_closure(generators, matrix=matrix)
        assert len(res) == (2 * n - 1) * (2 * n - 2) // 2

    @pytest.mark.parametrize("matrix", [False, True])
    @pytest.mark.parametrize("n", range(3, 5))
    def test_lie_closure_transverse_field_ising_1D_cyclic(self, n, matrix):
        """Test the lie closure works correctly for the transverse Field Ising model with cyclic boundary conditions, a8 in theorem IV.2 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n)]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n)
        ]

        res = lie_closure(generators, matrix=matrix)
        assert len(res) == 2 * n * (2 * n - 1)

    @pytest.mark.parametrize("matrix", [False, True])
    def test_lie_closure_heisenberg_generators_odd(self, matrix):
        """Test the resulting DLA from Heisenberg generators with odd n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
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

        res = lie_closure(generators, matrix=matrix)
        assert len(res) == (2 ** (n - 1)) ** 2 - 1

    @pytest.mark.parametrize("matrix", [False, True])
    def test_lie_closure_heisenberg_generators_even(self, matrix):
        """Test the resulting DLA from Heisenberg generators with even n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
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

        res = lie_closure(generators, matrix=matrix)
        assert len(res) == 4 * ((2 ** (n - 2)) ** 2 - 1)

    @pytest.mark.parametrize("matrix", [False, True])
    @pytest.mark.parametrize("n, res", [(3, 4), (4, 12)])
    def test_lie_closure_heisenberg(self, n, res, matrix):
        """Test the resulting DLA from Heisenberg model with summed generators"""
        genXX = [X(i) @ X(i + 1) for i in range(n - 1)]
        genYY = [Y(i) @ Y(i + 1) for i in range(n - 1)]
        genZZ = [Z(i) @ Z(i + 1) for i in range(n - 1)]

        generators = [qml.sum(XX + YY + ZZ) for XX, YY, ZZ in zip(genXX, genYY, genZZ)]
        g = qml.lie_closure(generators, matrix=matrix)
        assert len(g) == res

    @pytest.mark.parametrize("matrix", [False, True])
    def test_universal_gate_set(self, matrix):
        """Test universal gate set"""
        n = 3

        generators = [Z(i) for i in range(n)]
        generators += [Y(i) for i in range(n)]
        generators += [
            (I(i) - Z(i)) @ (I(i + 1) - X(i + 1)) for i in range(n - 1)
        ]  # generator of CNOT gate

        vspace = qml.lie_closure(generators, matrix=matrix)

        assert len(vspace) == 4**3


### Test matrix capabilities of lie_closure

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
    """Tests for lie_closure(*args, matrix=True)"""

    def test_simple_lie_closure_dense(self):
        """Test simple lie_closure_dense example"""

        dla12 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -2.0, PauliWord({1: "Z"}): 2.0}),
        ]

        gen12 = dla12[:-1]
        res12 = lie_closure(gen12, matrix=True)
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
        res11 = lie_closure(gen11, matrix=True)

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
        res11 = lie_closure(gen11, matrix=True)

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

        res = lie_closure(gen, matrix=True)

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

        res = lie_closure(gen, matrix=True)
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

    def test_non_hermitian_error(self):
        """Test that an error is raised for non-Hermitian input"""
        ops = [np.array([[0.0, 1.0], [0.0, 0.0]])]
        with pytest.raises(ValueError, match="At least one basis matrix"):
            _ = qml.lie_closure(ops, matrix=True)


X0 = qml.matrix(X(0))
Y0 = qml.matrix(Y(0))
Z0 = qml.matrix(Z(0))


class TestLieClosureInterfaces:
    """Test input for matrix inputs from AD interfaces"""

    @pytest.mark.jax
    def test_jax_lie_closure_matrix(self):
        """Test lie_closure can handle jax inputs in matrix mode"""
        import jax.numpy as jnp

        su2 = np.array([X0, Y0, -Z0])
        gens_list = [jnp.array(X0), jnp.array(Y0)]

        gens = jnp.array([X0, Y0])

        res = qml.lie_closure(gens, matrix=True)
        assert qml.math.allclose(res, su2)

        res_list = qml.lie_closure(gens_list, matrix=True)
        assert qml.math.allclose(res_list, su2)

    @pytest.mark.torch
    def test_torch_lie_closure_matrix(self):
        """Test lie_closure can handle torch inputs in matrix mode"""
        import torch

        su2 = torch.tensor(np.array([X0, Y0, -Z0]))
        gens_list = [torch.tensor(X0), torch.tensor(Y0)]

        gens = torch.tensor(np.array([X0, Y0]))

        res = qml.lie_closure(gens, matrix=True)
        assert qml.math.allclose(res, su2)

        res_list = qml.lie_closure(gens_list, matrix=True)
        assert qml.math.allclose(res_list, su2)

    @pytest.mark.tf
    def test_tf_lie_closure_matrix(self):
        """Test lie_closure can handle tf inputs in matrix mode"""
        import tensorflow as tf

        su2 = qml.math.stack([X0, Y0, -Z0], like="tensorflow")
        gens_list = [tf.constant(X0), tf.constant(Y0)]

        gens = qml.math.stack([tf.constant(X0), tf.constant(Y0)])

        res = qml.lie_closure(gens, matrix=True)
        assert qml.math.allclose(res, su2)

        res_list = qml.lie_closure(gens_list, matrix=True)
        assert qml.math.allclose(res_list, su2)

    @pytest.mark.autograd
    def test_autograd_lie_closure_matrix(self):
        """Test lie_closure can handle autograd inputs in matrix mode"""
        import pennylane.numpy as pnp

        su2 = pnp.array([X0, Y0, -Z0])
        gens_list = [pnp.array(X0), pnp.array(Y0)]

        gens = pnp.array([X0, Y0])

        res = qml.lie_closure(gens, matrix=True)
        assert qml.math.allclose(res, su2)

        res_list = qml.lie_closure(gens_list, matrix=True)
        assert qml.math.allclose(res_list, su2)
