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
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.io.qualtran_io import _get_op_call_graph, _get_to_pl_op, _map_to_bloq, _QReg


@pytest.fixture
def skip_if_no_pl_qualtran_support():
    """Fixture to skip if qualtran is not available"""
    pytest.importorskip("qualtran")


def test_to_bloq_error():
    """Test import error message when ToBloq() is instantiated without qualtran installed"""
    try:
        import qualtran  # pylint: disable=unused-import
    except (ModuleNotFoundError, ImportError):
        with pytest.raises(ImportError, match="Optional dependency"):
            qml.io.ToBloq(qml.H(0))

        with pytest.raises(ImportError, match="The `to_bloq` function requires Qualtran "):
            qml.to_bloq(qml.H(0))


@pytest.mark.external
@pytest.mark.usefixtures("skip_if_no_pl_qualtran_support")
@pytest.fixture
def qubits():
    """Provides cirq.LineQubit instances for tests."""
    import cirq

    return cirq.LineQubit(0), cirq.LineQubit(1)


@pytest.mark.external
@pytest.mark.usefixtures("skip_if_no_pl_qualtran_support")
@pytest.fixture
def dtypes():
    """Provides qualtran QDType instances for tests."""
    from qualtran import QBit, QUInt

    # A single QBit and a single-bit QUInt are different types
    # but should be equivalent in a 1-qubit _QReg comparison.
    return QBit(), QUInt(bitsize=1)


@pytest.mark.external
@pytest.mark.usefixtures("skip_if_no_pl_qualtran_support")
class TestFromBloq:
    """Test that FromBloq accurately wraps around Bloqs."""

    def test_bloq_init(self):
        """Tests that FromBloq's __init__() functions as intended"""

        from qualtran.bloqs.basic_gates import XGate

        assert repr(qml.FromBloq(XGate(), 1)) == "FromBloq(XGate, wires=Wires([1]))"
        with pytest.raises(TypeError, match="bloq must be an instance of"):
            qml.FromBloq("123", 1)

    def test_matrix_error(self):
        """Tests that FromBloq's matrix raises error as intended"""

        from qualtran.bloqs.phase_estimation import RectangularWindowState

        from pennylane.exceptions import MatrixUndefinedError

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

        qml.ops.functions.assert_valid(op, skip_deepcopy=True, skip_pickle=True)

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


@pytest.mark.external
@pytest.mark.usefixtures("skip_if_no_pl_qualtran_support")
class TestToBloq:
    """Test that ToBloq and to_bloq accurately wraps or maps Bloqs."""

    def test_to_bloq_init(self):
        """Tests that ToBloq's __init__() functions as intended"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.H(0)

        assert repr(qml.io.ToBloq(qml.Hadamard(0))) == "ToBloq(Hadamard)"
        assert repr(qml.io.ToBloq(circuit)) == "ToBloq(QNode)"
        assert str(qml.io.ToBloq(qml.H(0))) == "PLHadamard"
        with pytest.raises(TypeError, match="Input must be either an instance of"):
            qml.io.ToBloq("123")

    def test_equivalence(self):
        """Tests that ToBloq's __eq__ functions as expected"""

        assert qml.io.ToBloq(qml.H(0)) == qml.io.ToBloq(qml.H(0))
        assert qml.io.ToBloq(qml.H(0)) != qml.io.ToBloq(qml.H(1))
        assert qml.io.ToBloq(qml.H(0)) != "Hadamard"

    def test_allocate_and_free(self):
        """Tests that ToBloq functions on a FromBloq that has ghost wires"""
        from qualtran._infra.data_types import QAny, QBit
        from qualtran.bloqs.basic_gates import CZPowGate
        from qualtran.bloqs.bookkeeping import Allocate, Free

        assert (
            qml.to_bloq(qml.FromBloq(CZPowGate(0.468, eps=1e-11), wires=[0, 1])).call_graph()[1][
                Allocate(QAny(bitsize=1))
            ]
            == 3
        )
        assert (
            qml.to_bloq(qml.FromBloq(CZPowGate(0.468, eps=1e-11), wires=[0, 1])).call_graph()[1][
                Free(QBit())
            ]
            == 3
        )

    def test_to_bloq(self):
        """Tests that to_bloq functions as intended for simple circuits and gates"""

        from qualtran.bloqs.basic_gates import Hadamard

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            return (
                qml.expval(qml.Y(0)),
                qml.probs(op=qml.X(0)),
                qml.state(),
                qml.sample(qml.X(0)),
                qml.var(qml.X(0)),
                qml.counts(qml.X(0)),
            )

        def qfunc():
            qml.H(0)

        assert qml.to_bloq(qml.Hadamard(0)) == Hadamard()
        assert repr(qml.to_bloq(circuit)) == "ToBloq(QNode)"
        assert repr(qml.to_bloq(qfunc)) == "ToBloq(Qfunc)"
        assert str(qml.to_bloq(qfunc)) == "PLQfunc"
        assert repr(qml.to_bloq(qml.Hadamard(0), map_ops=False)) == "Hadamard()"
        assert qml.to_bloq(circuit).call_graph()[1] == {Hadamard(): 1}
        assert qml.to_bloq(qfunc).call_graph()[1] == {Hadamard(): 1}

        with pytest.raises(
            ValueError, match="Custom mappings are not possible for basic operations"
        ):
            qml.to_bloq(qml.X(0), custom_mapping={qml.X(0): qml.Y(0)})

    def test_to_bloq_circuits(self):
        """Tests that to_bloq functions as intended for complex circuits"""

        from qualtran.bloqs.basic_gates import CNOT, Hadamard

        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.QuantumPhaseEstimation(unitary=qml.RX(0.1, wires=5), estimation_wires=range(5))

        mapped_circuit = qml.to_bloq(circuit)
        mapped_circuit_cg = mapped_circuit.call_graph()[1]
        custom_mapped_circuit = qml.to_bloq(
            circuit,
            custom_mapping={
                qml.QuantumPhaseEstimation(
                    unitary=qml.RX(0.1, wires=5), estimation_wires=range(5)
                ): Hadamard()
            },
        )
        custom_mapped_circuit_cg = custom_mapped_circuit.call_graph()[1]
        wrapped_circuit = qml.to_bloq(circuit, map_ops=False)
        wrapped_circuit_cg = wrapped_circuit.call_graph()[1]

        assert mapped_circuit_cg[Hadamard()] == 11
        assert wrapped_circuit_cg[Hadamard()] == 11
        assert custom_mapped_circuit_cg[Hadamard()] == 2
        assert CNOT() not in mapped_circuit_cg
        assert wrapped_circuit_cg[CNOT()] == 20

    def test_from_bloq_to_bloq(self):
        """Tests that FromBloq and to_bloq functions as intended"""

        qpe_op = qml.QuantumPhaseEstimation(
            unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
        )
        qpe_bloq = qml.to_bloq(qpe_op, map_ops=False)

        decomp_ops = qml.FromBloq(qpe_bloq, wires=range(5)).decomposition()
        expected_decomp_ops = qpe_op.decomposition()
        assert decomp_ops == [
            qml.H(1),
            qml.H(2),
            qml.H(3),
            qml.H(4),
            qml.FromBloq(_map_to_bloq(expected_decomp_ops[4]), wires=[1, 0]),
            qml.FromBloq(_map_to_bloq(expected_decomp_ops[5]), wires=[2, 0]),
            qml.FromBloq(_map_to_bloq(expected_decomp_ops[6]), wires=[3, 0]),
            qml.FromBloq(_map_to_bloq(expected_decomp_ops[7]), wires=[4, 0]),
            qml.FromBloq(_map_to_bloq(expected_decomp_ops[8], map_ops=False), wires=range(1, 5)),
        ]

    def test_circuit_to_bloq_kwargs(self):
        """Tests that to_bloq functions as intended for circuits with kwargs"""

        from qualtran.bloqs.basic_gates import GlobalPhase, Rx

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angle):
            qml.RX(phi=angle, wires=[0])
            qml.GlobalPhase(angle)

        assert qml.to_bloq(circuit, angle=0).call_graph()[1] == {
            Rx(angle=0.0, eps=1e-11): 1,
            GlobalPhase(exponent=0): 1,
        }
        with pytest.raises(TypeError):
            qml.to_bloq(circuit).call_graph()

        assert qml.to_bloq(circuit, map_ops=False, angle=0).call_graph()[1] == {
            Rx(angle=0.0, eps=1e-11): 1,
            GlobalPhase(exponent=0): 1,
        }

    def test_decomposition_undefined_error(self):
        """Tests that DecomposeTypeError is raised when the input op has no decomposition"""
        import qualtran as qt

        with pytest.raises(qt.DecomposeTypeError):
            qml.to_bloq(qml.RZ(phi=0.3, wires=[0]), map_ops=False).decompose_bloq()

    def test_call_graph(self):
        """Tests that build_call_graph calls build_call_graph as expected"""
        from qualtran.resource_counting import SympySymbolAllocator as ssa

        cg = qml.to_bloq(
            qml.QuantumPhaseEstimation(unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)),
            False,
        ).build_call_graph(ssa=ssa())

        assert cg == {
            qml.to_bloq(qml.Hadamard(0), True): 4,
            qml.to_bloq(qml.ctrl(qml.RX(0.1, wires=0), control=[1]), True): 15,
            qml.to_bloq(qml.adjoint(qml.QFT(wires=range(1, 5))), False): 1,
        }

    def test_map_to_bloq(self):
        """Tests that _map_to_bloq produces the correct Qualtran equivalent"""
        from qualtran.bloqs.basic_gates import (
            CNOT,
            CZ,
            CYGate,
            GlobalPhase,
            Identity,
            Rx,
            Ry,
            Rz,
            SGate,
            TGate,
            Toffoli,
            TwoBitCSwap,
            TwoBitSwap,
            XGate,
            YGate,
            ZGate,
        )

        assert GlobalPhase(exponent=1) == _map_to_bloq(
            qml.GlobalPhase(GlobalPhase(exponent=1).exponent * np.pi, 0)
        )
        assert Identity() == _map_to_bloq(qml.Identity(0))
        assert Ry(angle=np.pi / 2) == _map_to_bloq(qml.RY(np.pi / 2, 0))
        assert Rx(angle=np.pi / 4) == _map_to_bloq(qml.RX(np.pi / 4, 0))
        assert Rz(angle=np.pi / 3) == _map_to_bloq(qml.RZ(np.pi / 3, 0))
        assert SGate() == _map_to_bloq(qml.S(0))
        assert TwoBitSwap() == _map_to_bloq(qml.SWAP([0, 1]))
        assert TwoBitCSwap() == _map_to_bloq(qml.CSWAP([0, 1, 2]))
        assert TGate() == _map_to_bloq(qml.T(0))
        assert XGate() == _map_to_bloq(qml.PauliX(0))
        assert YGate() == _map_to_bloq(qml.PauliY(0))
        assert CYGate() == _map_to_bloq(qml.CY([0, 1]))
        assert ZGate() == _map_to_bloq(qml.PauliZ(0))
        assert CZ() == _map_to_bloq(qml.CZ([0, 1]))
        assert CNOT() == _map_to_bloq(qml.CNOT([0, 1]))
        assert Toffoli() == _map_to_bloq(qml.Toffoli([0, 1, 2]))

    @pytest.mark.parametrize(
        (
            "op",
            "qml_call_graph",  # Computed by resources from labs or decompositions
        ),
        [
            (
                qml.QuantumPhaseEstimation(
                    unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
                ),
                # ResourceQPE
                {
                    (qml.Hadamard(0), True): 4,
                    (qml.ctrl(qml.RX(0.1, wires=0), control=[1]), True): 15,
                    (qml.adjoint(qml.QFT(wires=range(1, 5))), False): 1,
                },
            ),
            (
                qml.Superposition(
                    coeffs=np.sqrt(np.array([1 / 3, 1 / 3, 1 / 3])),
                    bases=np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
                    wires=[0, 1, 2],
                    work_wire=3,
                ),
                # Inspired by Resource Superposition
                {
                    (
                        qml.StatePrep(
                            np.array([0.57735027, 0.57735027, 0.57735027]), wires=[2, 3], pad_with=0
                        ),
                        False,
                    ): 1,
                    (qml.CNOT([0, 1]), True): 2,
                    (qml.MultiControlledX(wires=range(4), control_values=[1, 0, 0]), True): 4,
                },
            ),
            (qml.BasisState(np.array([1, 1]), wires=[0, 1]), {(qml.X(0), True): 2}),
            (
                qml.QFT(wires=range(5)),
                # From ResourceQFT
                {
                    (qml.H(0), True): 5,
                    (qml.ControlledPhaseShift(1, [0, 1]), True): 10,
                    (qml.SWAP([0, 1]), True): 2,
                },
            ),
            (
                qml.QROMStatePreparation(
                    np.sqrt(np.array([0.5, 0.0, 0.25, 0.25])), [4, 5], [1, 2, 3], [0]
                ),
                {
                    (
                        qml.QROM(
                            bitstrings=["001"],
                            control_wires=[],
                            target_wires=[1, 2, 3],
                            work_wires=[0],
                            clean=False,
                        ),
                        True,
                    ): 1,
                    (
                        qml.adjoint(
                            qml.QROM(
                                bitstrings=["001"],
                                control_wires=[],
                                target_wires=[1, 2, 3],
                                work_wires=[0],
                                clean=False,
                            )
                        ),
                        True,
                    ): 1,
                    (
                        qml.QROM(
                            bitstrings=["000", "001"],
                            control_wires=[4],
                            target_wires=[1, 2, 3],
                            work_wires=[0],
                            clean=False,
                        ),
                        True,
                    ): 1,
                    (
                        qml.adjoint(
                            qml.QROM(
                                bitstrings=["000", "001"],
                                control_wires=[4],
                                target_wires=[1, 2, 3],
                                work_wires=[0],
                                clean=False,
                            )
                        ),
                        True,
                    ): 1,
                    (qml.CRY(0.0, wires=[0, 1]), True): 6,
                },
            ),
            (
                qml.QROM(
                    bitstrings=["000", "001"],
                    control_wires=[4],
                    target_wires=[1, 2, 3],
                    work_wires=[0],
                    clean=False,
                ),
                # From ResourceQROM
                {
                    (qml.CNOT([0, 1]), True): 1,
                    (
                        qml.MultiControlledX(
                            wires=[0, 1], control_values=[True], work_wires=range(2, 3)
                        ),
                        True,
                    ): 4,
                    (qml.X(0), True): 4,
                    (qml.CSWAP([0, 1, 2]), True): 0.0,
                },
            ),
            (
                qml.QROM(
                    bitstrings=["001"],
                    control_wires=[],
                    target_wires=[1, 2, 3],
                    work_wires=[0],
                    clean=False,
                ),
                # From ResourceQROM
                {
                    (qml.X(0), True): 1,
                },
            ),
            (
                qml.QROM(
                    bitstrings=["000", "001"],
                    control_wires=[4],
                    target_wires=[1, 2, 3],
                    work_wires=[0],
                    clean=True,
                ),
                # From ResourceQROM
                {
                    (qml.Hadamard(0), True): 6,
                    (qml.CNOT([0, 1]), True): 1,
                    (
                        qml.MultiControlledX(
                            wires=[0, 1], control_values=[True], work_wires=range(2, 3)
                        ),
                        True,
                    ): 8,
                    (qml.X(0), True): 8,
                    (qml.CSWAP([0, 1, 2]), True): 0.0,
                },
            ),
            (
                qml.QROMStatePreparation(np.array([0.5, -0.5, 0.5, 0.5]), [4, 5], [1, 2, 3], [0]),
                {
                    (
                        qml.QROM(
                            bitstrings=["001"],
                            control_wires=[],
                            target_wires=[1, 2, 3],
                            work_wires=[0],
                            clean=False,
                        ),
                        True,
                    ): 1,
                    (
                        qml.adjoint(
                            qml.QROM(
                                bitstrings=["001"],
                                control_wires=[],
                                target_wires=[1, 2, 3],
                                work_wires=[0],
                                clean=False,
                            )
                        ),
                        True,
                    ): 1,
                    (
                        qml.QROM(
                            bitstrings=["000", "001"],
                            control_wires=[4],
                            target_wires=[1, 2, 3],
                            work_wires=[0],
                            clean=False,
                        ),
                        True,
                    ): 1,
                    (
                        qml.adjoint(
                            qml.QROM(
                                bitstrings=["000", "001"],
                                control_wires=[4],
                                target_wires=[1, 2, 3],
                                work_wires=[0],
                                clean=False,
                            )
                        ),
                        True,
                    ): 1,
                    (
                        qml.QROM(
                            bitstrings=["000", "000", "001", "001"],
                            control_wires=[4, 5],
                            target_wires=[1, 2, 3],
                            work_wires=[0],
                            clean=False,
                        ),
                        True,
                    ): 1,
                    (
                        qml.adjoint(
                            qml.QROM(
                                bitstrings=["000", "000", "001", "001"],
                                control_wires=[4, 5],
                                target_wires=[1, 2, 3],
                                work_wires=[0],
                                clean=False,
                            )
                        ),
                        True,
                    ): 1,
                    (qml.CRY(0.0, wires=[0, 1]), True): 6,
                    (
                        qml.ctrl(
                            qml.GlobalPhase((2 * np.pi), wires=[1]),
                            control=0,
                        ),
                        True,
                    ): 3,
                },
            ),
            (
                qml.ModExp(
                    x_wires=[0, 1],
                    output_wires=[2, 3, 4],
                    base=2,
                    mod=7,
                    work_wires=[5, 6, 7, 8, 9],
                ),
                {
                    (qml.ctrl(qml.adjoint(qml.QFT(range(4))), control=[4]), False): 1,
                    (qml.ctrl(qml.QFT(range(4)), control=[4]), False): 1,
                    (qml.Toffoli([0, 1, 2]), True): 6,
                },
            ),
            (
                qml.ModExp(
                    x_wires=[0, 1, 2],
                    output_wires=[3, 4, 5],
                    base=3,
                    mod=8,
                    work_wires=[6, 7, 8, 9, 10],
                ),
                {
                    (qml.ctrl(qml.QFT(range(3)), control=[4]), False): 1,
                    (qml.ctrl(qml.adjoint(qml.QFT(range(3))), control=[4]), False): 1,
                    (qml.Toffoli([0, 1, 2]), True): 21,
                },
            ),
            (
                qml.QSVT(
                    UA=qml.H(0),
                    projectors=[qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, -0.3)],
                ),
                {
                    (qml.RZ(phi=-2.46, wires=0), True): 1,
                    (qml.RZ(phi=1.0, wires=0), True): 1,
                    (qml.Hadamard(0), True): 2,
                    (qml.RZ(phi=0.6, wires=0), True): 1,
                },
            ),
            (
                qml.TrotterizedQfunc(
                    0.1,
                    *(0.12, -3.45),
                    qfunc=lambda time, theta, phi, wires, flip: (
                        qml.RX(time * theta, wires[0]),
                        qml.RY(time * phi, wires[1]),
                        qml.CNOT(wires=wires[:2]) if flip else None,
                    ),
                    n=1,
                    order=2,
                    wires=["a", "b"],
                    flip=True,
                ),
                {
                    (qml.RX(phi=0.012, wires=[0]), True): 2,
                    (qml.RY(phi=-0.34500000000000003, wires=[0]), True): 2,
                    (qml.CNOT(wires=[0, 1]), True): 2,
                },
            ),
            (
                qml.TrotterizedQfunc(
                    0.1,
                    *(0.12, -3.45),
                    qfunc=lambda time, theta, phi, wires, flip: (
                        qml.RX(time * theta, wires[0]),
                        qml.RY(time * phi, wires[1]),
                        qml.CNOT(wires=wires[:2]) if flip else None,
                    ),
                    n=1,
                    order=1,
                    wires=["a", "b"],
                    flip=True,
                ),
                {
                    (qml.RX(phi=0.012, wires=[0]), True): 1,
                    (qml.RY(phi=-0.34500000000000003, wires=[0]), True): 1,
                    (qml.CNOT(wires=[0, 1]), True): 1,
                },
            ),
            (
                qml.Select(ops=[qml.X(2), qml.QFT(wires=[2, 3, 4])], control=[0, 1]),
                {
                    (qml.X(wires=[2]), True): 2,
                    (qml.ctrl(qml.X(2), control=[0]), True): 1,
                    (qml.ctrl(qml.QFT(wires=[2, 3, 4]), control=[0]), True): 1,
                },
            ),
            (
                qml.StatePrep(state=[0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], wires=range(3)),
                {(qml.RZ(0, wires=[0]), True): 27, (qml.CNOT([0, 1]), True): 16},
            ),
        ],
    )
    def test_build_call_graph(self, op, qml_call_graph):
        """ "Tests that the defined call_graphs match the expected decompostions"""
        bloq_call_graph = {}

        for k, v in qml_call_graph.items():  # k is a tuple of (op, bool)
            bloq_call_graph[qml.to_bloq(k[0], map_ops=k[1])] = v

        call_graph = _get_op_call_graph(op)
        assert dict(call_graph) == bloq_call_graph

    @pytest.mark.parametrize(
        (
            "op",
            "qt_bloq",
        ),
        [
            (
                qml.QuantumPhaseEstimation(
                    unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
                ),
                "qpe_bloq",
            ),
            (qml.QFT(wires=range(4)), "qft_bloq"),
            (
                qml.ModExp(
                    x_wires=[0, 1],
                    output_wires=[2, 3, 4],
                    base=2,
                    mod=7,
                    work_wires=[5, 6, 7, 8, 9],
                ),
                "modexp_bloq",
            ),
            (
                qml.QROM(
                    bitstrings=["010", "111", "110", "000"],
                    control_wires=[0, 1],
                    target_wires=[2, 3, 4],
                    work_wires=[5, 6, 7],
                ),
                "qrom_bloq_clean",
            ),
            (
                qml.QROM(
                    bitstrings=["010", "111", "110", "000"],
                    control_wires=[0, 1],
                    target_wires=[2, 3, 4],
                    work_wires=[5, 6, 7],
                    clean=False,
                ),
                "qrom_bloq_dirty",
            ),
        ],
    )
    def test_default_mapping(self, op, qt_bloq):
        """Tests that the defined default maps match the expected qualtran bloq"""

        def _build_expected_qualtran_bloq(qt_bloq):
            """Factory function inside for parametrization of test cases"""
            from qualtran.bloqs.cryptography.rsa import ModExp
            from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
            from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
            from qualtran.bloqs.phase_estimation import RectangularWindowState
            from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE
            from qualtran.bloqs.qft import QFTTextBook

            qualtran_bloqs = {
                "qpe_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=RectangularWindowState(4),
                ),
                "qft_bloq": QFTTextBook(4),
                "modexp_bloq": ModExp(base=2, mod=7, exp_bitsize=2, x_bitsize=3),
                "qrom_bloq_clean": QROAMClean.build_from_data([2, 7, 6, 0]),
                "qrom_bloq_dirty": SelectSwapQROM.build_from_data([2, 7, 6, 0]),
            }

            return qualtran_bloqs[qt_bloq]

        qt_qpe = qml.to_bloq(op, map_ops=True)
        assert qt_qpe == _build_expected_qualtran_bloq(qt_bloq)

    @pytest.mark.parametrize(
        (
            "op",
            "custom_map",
            "qt_bloq",
        ),
        [
            (
                qml.QuantumPhaseEstimation(
                    unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
                ),
                "qpe_custom_mapping",
                "qpe_custom_bloq",
            ),
            # Tests the behaviour of using custom mapping for ops without default mappings
            (
                qml.QSVT(qml.H(0), [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]),
                "qsvt_custom_mapping",
                "qsvt_custom_bloq",
            ),
            (qml.QFT(wires=range(4)), "qft_custom_mapping", "qft_custom_bloq"),
            (
                qml.ModExp(
                    x_wires=[0, 1],
                    output_wires=[2, 3, 4],
                    base=2,
                    mod=7,
                    work_wires=[5, 6, 7, 8, 9],
                ),
                "modexp_custom_mapping",
                "modexp_custom_bloq",
            ),
            (
                qml.QROM(
                    bitstrings=["010", "111", "110", "000"],
                    control_wires=[0, 1],
                    target_wires=[2, 3, 4],
                    work_wires=[5, 6, 7],
                ),
                "qrom_custom_mapping",
                "qrom_custom_bloq",
            ),
        ],
    )
    def test_custom_mapping(self, op, custom_map, qt_bloq):
        """Tests that custom mapping maps the expected qualtran bloq"""

        def _build_expected_qualtran_bloq(qt_bloq):
            """Factory function to build expected Qualtran bloq inside for parametrization of test cases"""
            from qualtran.bloqs.phase_estimation import LPResourceState
            from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

            qualtran_bloqs = {
                "qpe_custom_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=LPResourceState(4),
                ),
                "qsvt_custom_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=LPResourceState(4),
                ),
                "qft_custom_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=LPResourceState(4),
                ),
                "modexp_custom_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=LPResourceState(4),
                ),
                "qrom_custom_bloq": TextbookQPE(
                    unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                    ctrl_state_prep=LPResourceState(4),
                ),
            }

            return qualtran_bloqs[qt_bloq]

        def _build_custom_map(custom_map):
            """Factory function to build custom maps for parametrization of test cases"""
            from qualtran.bloqs.phase_estimation import LPResourceState
            from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

            custom_mapping = {
                "qpe_custom_mapping": {
                    qml.QuantumPhaseEstimation(
                        unitary=qml.RX(0.1, wires=0), estimation_wires=range(1, 5)
                    ): TextbookQPE(
                        unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                        ctrl_state_prep=LPResourceState(4),
                    )
                },
                "qsvt_custom_mapping": {
                    qml.QSVT(
                        qml.H(0), [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]
                    ): TextbookQPE(
                        unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                        ctrl_state_prep=LPResourceState(4),
                    )
                },
                "qft_custom_mapping": {
                    qml.QFT(wires=range(4)): TextbookQPE(
                        unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                        ctrl_state_prep=LPResourceState(4),
                    )
                },
                "modexp_custom_mapping": {
                    qml.ModExp(
                        x_wires=[0, 1],
                        output_wires=[2, 3, 4],
                        base=2,
                        mod=7,
                        work_wires=[5, 6, 7, 8, 9],
                    ): TextbookQPE(
                        unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                        ctrl_state_prep=LPResourceState(4),
                    )
                },
                "qrom_custom_mapping": {
                    qml.QROM(
                        bitstrings=["010", "111", "110", "000"],
                        control_wires=[0, 1],
                        target_wires=[2, 3, 4],
                        work_wires=[5, 6, 7],
                    ): TextbookQPE(
                        unitary=qml.to_bloq(qml.RX(0.1, wires=0)),
                        ctrl_state_prep=LPResourceState(4),
                    )
                },
            }

            return custom_mapping[custom_map]

        qt_qpe = qml.to_bloq(op, map_ops=True, custom_mapping=_build_custom_map(custom_map))
        assert qt_qpe == _build_expected_qualtran_bloq(qt_bloq)

    # pylint: disable=redefined-outer-name, protected-access
    def test_initialization_with_tuple(self, qubits, dtypes):
        """Tests standard initialization with a tuple of qubits."""
        q0, q1 = qubits
        _, dtype_uint = dtypes
        qubits_tuple = (q0, q1)

        qreg = _QReg(qubits=qubits_tuple, dtype=dtype_uint)

        assert isinstance(qreg.qubits, tuple)

        assert qreg.qubits == qubits_tuple
        assert qreg.dtype == dtype_uint
        assert qreg._initialized is True

    # pylint: disable=redefined-outer-name
    def test_initialization_with_single_qubit(self, qubits, dtypes):
        """Tests that a single qubit is correctly wrapped in a tuple."""
        q0, _ = qubits
        dtype_bit, _ = dtypes

        qreg = _QReg(qubits=q0, dtype=dtype_bit)

        assert repr(qreg) == "_QReg(qubits=(cirq.LineQubit(0),), dtype=QBit())"
        assert isinstance(qreg.qubits, tuple)
        assert qreg.qubits == (q0,)
        assert qreg.dtype == dtype_bit

    # pylint: disable=redefined-outer-name
    def test_immutability_raises_error_on_attribute_change(self, qubits, dtypes):
        """Tests that changing an attribute after initialization raises an AttributeError."""
        q0, q1 = qubits
        dtype_bit, dtype_uint = dtypes
        qreg = _QReg(qubits=(q0,), dtype=dtype_bit)

        with pytest.raises(AttributeError, match="Cannot set attribute 'qubits'"):
            qreg.qubits = (q1,)

        with pytest.raises(AttributeError, match="Cannot set attribute 'dtype'"):
            qreg.dtype = dtype_uint

        with pytest.raises(AttributeError, match="Cannot set attribute 'new_attr'"):
            qreg.new_attr = "some_value"

    # pylint: disable=redefined-outer-name
    def test_equality_ignores_dtype(self, qubits, dtypes):
        """Tests the core feature: equality should only depend on qubits, not dtype."""
        q0, _ = qubits
        dtype_bit, dtype_uint = dtypes

        qreg1 = _QReg(qubits=(q0,), dtype=dtype_bit)
        qreg2 = _QReg(qubits=(q0,), dtype=dtype_uint)

        assert qreg1 == qreg2

    # pylint: disable=redefined-outer-name
    def test_hash_ignores_dtype(self, qubits, dtypes):
        """Tests that the hash also only depends on the qubits."""
        q0, _ = qubits
        dtype_bit, dtype_uint = dtypes

        qreg1 = _QReg(qubits=(q0,), dtype=dtype_bit)
        qreg2 = _QReg(qubits=(q0,), dtype=dtype_uint)

        assert hash(qreg1) == hash(qreg2)

    # pylint: disable=redefined-outer-name
    def test_inequality_for_different_qubits(self, qubits, dtypes):
        """Tests that instances with different qubits are not equal."""
        q0, q1 = qubits
        dtype_bit, _ = dtypes

        qreg1 = _QReg(qubits=(q0,), dtype=dtype_bit)
        qreg2 = _QReg(qubits=(q1,), dtype=dtype_bit)
        qreg3 = _QReg(qubits=(q0, q1), dtype=dtype_bit)

        assert qreg1 != qreg2
        assert qreg1 != qreg3

    # pylint: disable=redefined-outer-name
    def test_inequality_for_different_types(self, qubits, dtypes):
        """Tests that comparison with other types returns NotImplemented/False."""
        q0, _ = qubits
        dtype_bit, _ = dtypes
        qreg = _QReg(qubits=(q0,), dtype=dtype_bit)

        assert qreg != "not_a_qreg"
        assert qreg != (q0,)
        assert qreg is not None
