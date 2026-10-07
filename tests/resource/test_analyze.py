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
"""Unit tests for the analyze transform"""

from functools import partial

import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


@pytest.mark.usefixtures("enable_and_disable_capture")
class TestAnalyze:
    """Test qp.analyze() with capture enabled and disabled."""

    @pytest.fixture
    def circuit(self):
        """Fixture for a qjit'd circuit with two transforms and a marker between them."""

        @qp.qjit(capture="global")
        @qp.transforms.merge_rotations
        @qp.marker("cancelled")
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.X(0)
            qp.X(0)
            qp.CNOT(wires=[0, 1])
            return qp.probs()

        return circuit

    # Gate counts of the ``circuit`` fixture at each level
    TOP_COUNTS = {"RX": 2, "PauliX": 2, "CNOT": 1}
    CANCELLED_COUNTS = {"RX": 2, "CNOT": 1}
    USER_COUNTS = {"RX": 1, "CNOT": 1}

    @pytest.mark.parametrize("level", ["all", range(3)])
    def test_resources_at_each_level(self, circuit, level):
        """Test that analyze counts the resources left after each transform."""

        specs = qp.analyze(circuit, level=level)(0.1)

        assert specs == CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "Before MLIR Passes", 1: "cancelled", 2: "merge-rotations"},
            resources={
                "Before MLIR Passes": SpecsResources(
                    counts=self.TOP_COUNTS,
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "cancelled": SpecsResources(
                    counts=self.CANCELLED_COUNTS,
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
                "merge-rotations": SpecsResources(
                    counts=self.USER_COUNTS,
                    measurement_processes={"probs(all wires)": 1},
                    num_wires=2,
                ),
            },
        )

    @pytest.mark.parametrize(
        "level, expected_counts",
        [
            (0, TOP_COUNTS),
            ("top", TOP_COUNTS),
            (1, CANCELLED_COUNTS),
            ("cancelled", CANCELLED_COUNTS),
            (2, USER_COUNTS),
            ("user", USER_COUNTS),
        ],
    )
    def test_single_level(self, circuit, level, expected_counts):
        """Test that analyze counts the resources at a single level."""

        specs = qp.analyze(circuit, level=level)(0.1)

        assert specs.resources.counts == expected_counts

    def test_default_level_is_user(self, circuit):
        """Test that analyze defaults to the resources after all user transforms."""

        assert qp.analyze(circuit)(0.1) == qp.analyze(circuit, level="user")(0.1)

    @pytest.mark.parametrize(
        "level, expected_counts",
        [
            ([0, "cancelled"], {"Before MLIR Passes": TOP_COUNTS, "cancelled": CANCELLED_COUNTS}),
            ((0, 2), {"Before MLIR Passes": TOP_COUNTS, "merge-rotations": USER_COUNTS}),
        ],
    )
    def test_multiple_levels(self, circuit, level, expected_counts):
        """Test that analyze counts the resources at each of the given levels only."""

        specs = qp.analyze(circuit, level=level)(0.1)

        assert {name: res.counts for name, res in specs.resources.items()} == expected_counts

    def test_symbolic_resources(self):
        """Test that analyze counts the resources inside a loop with a number of iterations that
        is not known at compile time symbolically."""

        @qp.qjit(autograph=True, capture="global")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(n):
            qp.Hadamard(0)
            for _ in range(n):
                qp.PauliX(0)
            return qp.expval(qp.Z(0))

        resources = qp.analyze(circuit, level=0)(5).resources

        assert resources.is_symbolic
        assert len(resources.vars) == 1

        concrete = resources.subs({var: 5 for var in resources.vars})
        assert not concrete.is_symbolic
        assert concrete.counts == {"Hadamard": 1, "PauliX": 5}

    @pytest.mark.parametrize(
        "level", [0, "top", 2, "user", "cancelled", [0, "cancelled"], (0, 2), range(3), "all"]
    )
    def test_matches_specs(self, circuit, level):
        """Test that analyze returns the same specs as qp.specs for the same level."""

        assert qp.analyze(circuit, level=level)(0.1) == qp.specs(circuit, level=level)(0.1)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_with_catalyst_passes(self):
        """Test that analyze counts the resources after each pass of a pipeline of Catalyst passes,
        with program capture enabled and disabled."""

        pipeline = qp.CompilePipeline(
            catalyst.passes.cancel_inverses, catalyst.passes.merge_rotations
        )

        @qp.qjit(capture="global")
        @pipeline
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        specs = qp.analyze(circuit, level="all")(0.5)

        assert specs.level == {0: "Before MLIR Passes", 1: "cancel-inverses", 2: "merge-rotations"}
        assert [resources.counts for resources in specs.resources.values()] == [
            {"Hadamard": 2, "RX": 2, "CNOT": 1},
            {"RX": 2, "CNOT": 1},
            {"RX": 1, "CNOT": 1},
        ]

    def test_partial(self, circuit):
        """Test analyze for a partial-wrapped Catalyst jitted QNode."""

        specs = qp.analyze(partial(circuit, 0.1), level="user")()

        assert specs.resources.counts == self.USER_COUNTS

    @pytest.mark.xfail(
        raises=NotImplementedError, strict=True, reason="level='device' is not supported yet."
    )
    def test_device_level(self, circuit):
        """Test that analyze counts the resources after device preprocessing."""

        specs = qp.analyze(circuit, level="device")(0.1)

        assert specs.resources.counts == self.USER_COUNTS

    @pytest.mark.parametrize("level", [None, 1.5])
    def test_unsupported_level(self, circuit, level):
        """Test that a helpful error message is raised for levels of an unsupported type."""

        with pytest.raises(ValueError, match="Invalid level"):
            qp.analyze(circuit, level=level)(0.1)

    def test_error_with_non_qjit(self):
        """Test that a helpful error message is raised if the input is not QJIT'd."""

        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match="qp.analyze can only be applied to a qjit'd QNode"):
            qp.analyze(circuit, level=0)()


@pytest.mark.catalyst
class TestSpecsHintIntegration:
    """Test integration of hints with qp.specs."""

    def test_for_loop_hint_outside(self):
        """Test for_loop hints."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c(n):

            @qp.hint({"num-iters": 20})
            @qp.for_loop(n)
            def loop(i):
                qp.X(i)

            loop()

            return qp.probs(wires=0)

        r = qp.analyze(c)(2)
        assert r.resources.quantum_operations["PauliX"] == 20

    def test_for_loop_hint_inside(self):
        """Test for_loop hints where the loop itself is decorated."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c(n):

            @qp.for_loop(n)
            @qp.hint({"num-iters": 4})
            def loop(i):
                qp.Y(i)

            loop()

            return qp.probs(wires=0)

        r = qp.analyze(c)(2)
        assert r.resources.quantum_operations["PauliY"] == 4

    def test_while_loop_hint_outside(self):
        """Test while_loop hints."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c(n):

            @qp.hint({"num-iters": 20})
            @qp.while_loop(lambda i: i < n)
            def loop(i):
                qp.X(i)
                return i + 1

            loop(0)

            return qp.probs(wires=0)

        r = qp.analyze(c)(2)
        assert r.resources.quantum_operations["PauliX"] == 20

    def test_while_loop_hint_inside(self):
        """Test while_loop hints where the loop itself is decorated."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c(n):

            @qp.while_loop(lambda i: i < n)
            @qp.hint({"num-iters": 4})
            def loop(i):
                qp.Y(i)
                return i + 1

            loop(0)

            return qp.probs(wires=0)

        r = qp.analyze(c)(2)
        assert r.resources.quantum_operations["PauliY"] == 4

    def test_cond_branch_prob_weighted_resources(self):
        """``branch-prob`` should weight resources instead of taking a max over branches."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(m1, m2):

            @qp.hint({"branch-prob": 0.4})
            def true_fn():
                for _ in range(10):
                    qp.X(0)

            @qp.hint({"branch-prob": 0.4})
            def false_fn():
                for _ in range(10):
                    qp.Y(0)

            def elif_fn():
                for _ in range(10):
                    qp.X(0)
                    qp.Z(0)

            qp.cond(m1, true_fn, false_fn, elifs=(m2, elif_fn))()
            return qp.expval(qp.Z(0))

        r = qp.analyze(circuit)(True, True)
        assert r.resources.quantum_operations["PauliX"] == 6
        assert r.resources.quantum_operations["PauliY"] == 4
        assert r.resources.quantum_operations["PauliZ"] == 2
        assert r.resources.total_quantum_operations == 12

    def test_cond_multiple_unhinted_branches(self):
        """Multiple unhinted branches should share remaining probability in specs."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(m1, m2, m3):

            @qp.hint({"branch-prob": 0.4})
            def true_fn():
                for _ in range(10):
                    qp.X(0)

            def elif_fn1():
                for _ in range(10):
                    qp.Y(0)

            def elif_fn2():
                for _ in range(10):
                    qp.Z(0)

            def false_fn():
                for _ in range(10):
                    qp.H(0)

            qp.cond(m1, true_fn, false_fn, elifs=((m2, elif_fn1), (m3, elif_fn2)))()
            return qp.expval(qp.Z(0))

        r = qp.analyze(circuit)(True, False, False)
        # remaining 0.6 split three ways -> 0.2 each: 4 X, 2 Y, 2 Z, 2 H
        assert r.resources.quantum_operations["PauliX"] == 4
        assert r.resources.quantum_operations["PauliY"] == 2
        assert r.resources.quantum_operations["PauliZ"] == 2
        assert r.resources.quantum_operations["Hadamard"] == 2
        assert r.resources.total_quantum_operations == 10

    def test_cond_branch_probs_sum_greater_than_one(self):
        """When hinted probs sum above 1, later branches are truncated in specs."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(m1, m2):

            @qp.hint({"branch-prob": 0.7})
            def true_fn():
                for _ in range(10):
                    qp.X(0)

            @qp.hint({"branch-prob": 0.7})
            def false_fn():
                for _ in range(10):
                    qp.Y(0)

            def elif_fn():
                for _ in range(10):
                    qp.Z(0)

            qp.cond(m1, true_fn, false_fn, elifs=(m2, elif_fn))()
            return qp.expval(qp.Z(0))

        r = qp.analyze(circuit)(True, False)
        # pennylane fills unhinted with 0 -> (0.7, 0.0, 0.7); catalyst clamps total weight to 1
        assert r.resources.quantum_operations["PauliX"] == 7
        assert r.resources.quantum_operations["PauliZ"] == 0
        assert r.resources.quantum_operations["PauliY"] == 3
        assert r.resources.total_quantum_operations == 10

    def test_cond_low_branch_prob_rounds_resources_to_zero(self):
        """Low probability times low gate count should floor to zero resources."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(m):

            @qp.hint({"branch-prob": 0.4})
            def true_fn():
                qp.X(0)

            @qp.hint({"branch-prob": 0.6})
            def false_fn():
                qp.Y(0)

            qp.cond(m, true_fn, false_fn)()
            return qp.expval(qp.Z(0))

        r = qp.analyze(circuit)(True)
        # 0.4 * 1 floors to 0; 0.6 * 1 rounds to 1
        assert r.resources.quantum_operations["PauliX"] == 0
        assert r.resources.quantum_operations["PauliY"] == 1
        assert r.resources.total_quantum_operations == 1
