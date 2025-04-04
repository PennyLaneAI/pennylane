# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.io.qualtran_io` module.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.io.qualtran_io import _get_to_pl_op
from pennylane.operation import DecompositionUndefinedError


@pytest.fixture
def skip_if_no_pl_qualtran_support():
    """Fixture to skip if qualtran is not available"""
    pytest.importorskip("qualtran")


@pytest.mark.external
@pytest.mark.usefixtures("skip_if_no_pl_qualtran_support")
class TestFromBloq:
    """Test that FromBloq accurately wraps around Bloqs."""

    def test_bloq_init(self):
        """Tests that FromBloq's __init__() functions as intended"""

        from qualtran.bloqs.basic_gates import XGate

        assert qml.FromBloq(XGate(), 1).__repr__() == "FromBloq(XGate, wires=Wires([1]))"
        with pytest.raises(TypeError, match="bloq must be an instance of"):
            qml.FromBloq("123", 1)

    def test_matrix_error(self):
        """Tests that FromBloq's matrix raises error as intended"""

        from qualtran.bloqs.phase_estimation import RectangularWindowState

        from pennylane.operation import MatrixUndefinedError

        with pytest.raises(MatrixUndefinedError):
            qml.FromBloq(RectangularWindowState(3), [0, 1, 2]).matrix()

    @pytest.mark.xfail(reason="Fails due to overly broad assertion")
    def test_assert_valid(self):
        """Tests that FromBloq passes the assert_valid check"""
        from qualtran import BloqBuilder, QUInt
        from qualtran.bloqs.arithmetic import Add, Product

        bb = BloqBuilder()

        w1 = bb.add_register("p1", 3)
        w2 = bb.add_register("p2", 3)
        w3 = bb.add_register("q1", 3)
        w4 = bb.add_register("q2", 3)

        w1, w2, res1 = bb.add(Product(3, 3), a=w1, b=w2)
        w3, w4, res2 = bb.add(Product(3, 3), a=w3, b=w4)
        p1p2, p1p2_plus_q1q2 = bb.add(Add(QUInt(bitsize=6), QUInt(bitsize=6)), a=res1, b=res2)

        cbloq = bb.finalize(p1=w1, p2=w2, q1=w3, q2=w4, p1p2=p1p2, p1p2_plus_q1q2=p1p2_plus_q1q2)

        op = qml.FromBloq(cbloq, wires=range(24))

        qml.ops.functions.assert_valid(op, True, True)

    def test_wrong_wires_error(self):
        """Tests that FromBloq validates the length of wires as intended"""

        from qualtran.bloqs.basic_gates import Hadamard

        with pytest.raises(ValueError, match="The length of wires must"):
            qml.FromBloq(Hadamard(), wires=[1, 2, 3]).decomposition()

    def test_allocated_and_freed_wires(self):
        """Tests that FromBloq properly handles bloqs that have allocate and free qubits"""

        from qualtran.bloqs.basic_gates import CZPowGate, ZPowGate

        assert qml.FromBloq(CZPowGate(), wires=range(2)).decomposition()[1] == qml.FromBloq(
            ZPowGate(), wires=["alloc_free_2"]
        )

    def test_partition_bloq(self):
        """Tests that FromBloq properly handles bloqs with partitions."""

        from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

        data1 = np.arange(5, dtype=int)
        data2 = np.arange(5, dtype=int) + 1
        qroam_clean_multi_data = QROAMClean.build_from_data(data1, data2, log_block_sizes=(1,))

        assert (
            len(
                qml.FromBloq(
                    qroam_clean_multi_data, wires=range(qroam_clean_multi_data.signature.n_qubits())
                ).decomposition()
            )
            == 3
        )

    def test_composite_bloq_advanced(self):
        """Tests that a composite bloq with higher level abstract bloqs has the correct
        decomposition after wrapped with `FromBloq`"""
        from qualtran import BloqBuilder, QUInt
        from qualtran.bloqs.arithmetic import Add, Product

        from pennylane.wires import Wires

        bb = BloqBuilder()

        w1 = bb.add_register("p1", 3)
        w2 = bb.add_register("p2", 3)
        w3 = bb.add_register("q1", 3)
        w4 = bb.add_register("q2", 3)

        w1, w2, res1 = bb.add(Product(3, 3), a=w1, b=w2)
        w3, w4, res2 = bb.add(Product(3, 3), a=w3, b=w4)
        p1p2, p1p2_plus_q1q2 = bb.add(Add(QUInt(bitsize=6), QUInt(bitsize=6)), a=res1, b=res2)

        cbloq = bb.finalize(p1=w1, p2=w2, q1=w3, q2=w4, p1p2=p1p2, p1p2_plus_q1q2=p1p2_plus_q1q2)

        expected = [
            qml.FromBloq(Product(3, 3), wires=Wires([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])),
            qml.FromBloq(Product(3, 3), wires=Wires([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23])),
            qml.FromBloq(
                Add(QUInt(bitsize=6), QUInt(bitsize=6)),
                wires=Wires([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            ),
        ]
        assert qml.FromBloq(cbloq, wires=range(24)).decomposition() == expected

    def test_composite_bloq(self):
        """Tests that a simple composite bloq has the correct decomposition after wrapped with `FromBloq`"""
        from qualtran import BloqBuilder
        from qualtran.bloqs.basic_gates import CNOT, Hadamard, Toffoli

        bb = BloqBuilder()  # bb is the circuit like object

        w1 = bb.add_register("wire1", 1)
        w2 = bb.add_register("wire2", 1)
        aux = bb.add_register("aux_wires", 2)

        aux_wires = bb.split(aux)

        w1 = bb.add(Hadamard(), q=w1)
        w2 = bb.add(Hadamard(), q=w2)

        w1, aux1 = bb.add(CNOT(), ctrl=w1, target=aux_wires[0])
        w2, aux2 = bb.add(CNOT(), ctrl=w2, target=aux_wires[1])

        ctrl_aux, w1 = bb.add(Toffoli(), ctrl=(aux1, aux2), target=w1)
        ctrl_aux, w2 = bb.add(Toffoli(), ctrl=ctrl_aux, target=w2)
        aux_wires = bb.join(ctrl_aux)

        circuit_bloq = bb.finalize(wire1=w1, wire2=w2, aux_wires=aux_wires)

        decomp = qml.FromBloq(circuit_bloq, wires=list(range(4))).decomposition()
        expected_decomp = [
            qml.H(0),
            qml.H(1),
            qml.CNOT([0, 2]),
            qml.CNOT([1, 3]),
            qml.Toffoli([2, 3, 0]),
            qml.Toffoli([2, 3, 1]),
        ]
        assert decomp == expected_decomp

        mapped_decomp = qml.FromBloq(circuit_bloq, wires=[3, 0, 1, 2]).decomposition()
        mapped_expected_decomp = [
            qml.H(3),
            qml.H(0),
            qml.CNOT([3, 1]),
            qml.CNOT([0, 2]),
            qml.Toffoli([1, 2, 3]),
            qml.Toffoli([1, 2, 0]),
        ]
        assert mapped_decomp == mapped_expected_decomp
        assert np.allclose(
            qml.FromBloq(circuit_bloq, wires=list(range(4))).matrix(),
            circuit_bloq.tensor_contract(),
        )

    def test_atomic_bloqs(self):
        """Tests that atomic bloqs have the correct PennyLane equivalent after wrapped with `FromBloq`"""
        from qualtran.bloqs.basic_gates import CNOT, Hadamard, Toffoli

        assert Hadamard().as_pl_op(0) == qml.Hadamard(0)
        assert CNOT().as_pl_op([0, 1]) == qml.CNOT([0, 1])
        assert Toffoli().as_pl_op([0, 1, 2]) == qml.Toffoli([0, 1, 2])

        assert qml.FromBloq(Hadamard(), 0).has_matrix is True
        with pytest.raises(DecompositionUndefinedError):
            qml.FromBloq(Hadamard(), wires=[1]).decomposition()

        assert np.allclose(qml.FromBloq(Hadamard(), 0).matrix(), qml.Hadamard(0).matrix())
        assert np.allclose(qml.FromBloq(CNOT(), [0, 1]).matrix(), qml.CNOT([0, 1]).matrix())
        assert np.allclose(
            qml.FromBloq(Toffoli(), [0, 1, 2]).matrix(), qml.Toffoli([0, 1, 2]).matrix()
        )

    def test_to_pl_op(self):  # Correctness is also validated in Qualtran's tests
        """Tests that _get_to_pl_op produces the correct PennyLane equivalent"""
        from qualtran.bloqs.basic_gates import (
            CZ,
            CYGate,
            GlobalPhase,
            Identity,
            Rx,
            Ry,
            Rz,
            SGate,
            TGate,
            TwoBitCSwap,
            XGate,
            YGate,
            ZGate,
        )

        to_pl = _get_to_pl_op()

        assert to_pl(GlobalPhase(exponent=1), 0) == qml.GlobalPhase(
            GlobalPhase(exponent=1).exponent * np.pi, 0
        )
        assert to_pl(Identity(), 0) == qml.Identity(0)
        assert to_pl(Ry(angle=np.pi / 2), 0) == qml.RY(np.pi / 2, 0)
        assert to_pl(Rx(angle=np.pi / 4), 0) == qml.RX(np.pi / 4, 0)
        assert to_pl(Rz(angle=np.pi / 3), 0) == qml.RZ(np.pi / 3, 0)
        assert to_pl(SGate(), 0) == qml.S(0)
        assert to_pl(TwoBitCSwap(), [0, 1, 2]) == qml.CSWAP([0, 1, 2])
        assert to_pl(TGate(), 0) == qml.T(0)
        assert to_pl(XGate(), 0) == qml.PauliX(0)
        assert to_pl(YGate(), 0) == qml.PauliY(0)
        assert to_pl(CYGate(), [0, 1]) == qml.CY([0, 1])
        assert to_pl(ZGate(), 0) == qml.PauliZ(0)
        assert to_pl(CZ(), [0, 1]) == qml.CZ([0, 1])

    def test_bloqs(self):
        """Tests that bloqs with decompositions have the correct PennyLane decompositions after
        being wrapped with `FromBloq`"""

        from qualtran.bloqs.basic_gates import Swap

        assert qml.FromBloq(Swap(3), wires=range(6)).decomposition() == [
            qml.SWAP(wires=[0, 3]),
            qml.SWAP(wires=[1, 4]),
            qml.SWAP(wires=[2, 5]),
        ]

        from qualtran.bloqs.basic_gates import ZPowGate
        from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE

        textbook_qpe_small = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FromBloq(textbook_qpe_small, wires=list(range(4)))
            return qml.state()

        # Expected value computed via the qualtran bloq's tensor_contract()
        assert np.allclose(
            circuit(),
            np.array(
                [
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 - 0.0j,
                    0.0 - 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
        )

    def test_bloq_registers(self):
        """Tests that bloq_registers returns the expected dictionary with the correct
        registers and wires."""

        from qualtran import BloqBuilder, QUInt
        from qualtran.bloqs.arithmetic import Add, Product

        from pennylane.wires import Wires

        with pytest.raises(TypeError, match="bloq must be an instance of"):
            qml.bloq_registers("123")

        bb = BloqBuilder()

        w1 = bb.add_register("p1", 3)
        w2 = bb.add_register("p2", 3)
        w3 = bb.add_register("q1", 3)
        w4 = bb.add_register("q2", 3)

        w1, w2, res1 = bb.add(Product(3, 3), a=w1, b=w2)
        w3, w4, res2 = bb.add(Product(3, 3), a=w3, b=w4)
        p1p2, p1p2_plus_q1q2 = bb.add(Add(QUInt(bitsize=6), QUInt(bitsize=6)), a=res1, b=res2)

        circuit_bloq = bb.finalize(
            p1=w1, p2=w2, q1=w3, q2=w4, p1p2=p1p2, p1p2_plus_q1q2=p1p2_plus_q1q2
        )

        expected = {
            "p1": Wires([0, 1, 2]),
            "p2": Wires([3, 4, 5]),
            "q1": Wires([6, 7, 8]),
            "q2": Wires([9, 10, 11]),
            "p1p2": Wires([12, 13, 14, 15, 16, 17]),
            "p1p2_plus_q1q2": Wires([18, 19, 20, 21, 22, 23]),
        }
        actual = qml.bloq_registers(circuit_bloq)

        assert actual == expected
