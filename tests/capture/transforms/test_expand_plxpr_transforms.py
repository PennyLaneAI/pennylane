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
"""Unit tests for ``expand_plxpr_transforms``"""

# pylint:disable=wrong-import-position,protected-access
from functools import partial

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture import expand_plxpr_transforms
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter

pytestmark = [pytest.mark.jax, pytest.mark.capture]


@qml.transform
def dummy_tape_only_transform(tape):
    return [tape], lambda res: res[0]


def _dummy_plxpr_transform(jaxpr, consts, targs, tkwargs, *args):  # pylint: disable=unused-argument
    """Dummy function to transform plxpr."""

    def wrapper(*inner_args):
        return jax.core.eval_jaxpr(jaxpr, consts, *inner_args)

    return jax.make_jaxpr(wrapper)(*args)


@partial(qml.transform, plxpr_transform=_dummy_plxpr_transform)
def dummy_tape_and_plxpr_transform(tape):
    return [tape], lambda res: res[0]


class TestExpandTransformsInterpreter:
    """Unit tests for ExpandTransformsInterpreter"""

    def test_expand_transforms_interpreter_registration(self):
        """Test that the primitives of PennyLane transforms are automatically registered with the
        ExpandTransformsInterpreter."""

        assert (
            dummy_tape_only_transform._primitive
            in ExpandTransformsInterpreter._primitive_registrations
        )
        assert (
            dummy_tape_and_plxpr_transform._primitive
            in ExpandTransformsInterpreter._primitive_registrations
        )

    def test_expand_transforms_interpreter_plxpr_transform(self):
        """Test that transforms that have a valid ``plxpr_transform`` are handled
        correctly."""

        custom_handler = ExpandTransformsInterpreter._primitive_registrations[
            dummy_tape_and_plxpr_transform._primitive
        ]
        assert dummy_tape_and_plxpr_transform.plxpr_transform is not None

        def f(x):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            return qml.expval(qml.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)

        def wrapper(*inner_args):
            interpreter = ExpandTransformsInterpreter()
            invals = [*inner_args, *jaxpr.consts]
            params = {
                "inner_jaxpr": jaxpr.jaxpr,
                "args_slice": slice(0, len(inner_args)),
                "consts_slice": slice(len(inner_args), len(jaxpr.consts) + len(inner_args)),
                "targs_slice": slice(len(jaxpr.consts) + len(inner_args), None),
                "tkwargs": {},
            }
            return custom_handler(interpreter, *invals, **params)

        new_jaxpr = jax.make_jaxpr(wrapper)(*args)
        assert all(
            orig_eqn.primitive == new_eqn.primitive
            for orig_eqn, new_eqn in zip(jaxpr.eqns, new_jaxpr.eqns, strict=True)
        )
        assert all(
            orig_eqn.params == new_eqn.params
            for orig_eqn, new_eqn in zip(jaxpr.eqns, new_jaxpr.eqns, strict=True)
        )


class TestExpandPlxprTransforms:
    """Unit tests for ``expand_plxpr_transforms``."""

    def test_expand_single_transform_full_transform(self):
        """Test that a function that has a single transform is expanded correctly when the entire
        function is transformed."""

        @qml.transforms.cancel_inverses
        def f():
            qml.X(0)
            qml.S(1)
            qml.X(0)
            qml.adjoint(qml.S(1))
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == qml.transforms.cancel_inverses._primitive
        assert jaxpr.jaxpr.outvars == jaxpr.eqns[0].outvars

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()
        assert len(transformed_jaxpr.eqns) == 2
        assert transformed_jaxpr.eqns[0].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.jaxpr.outvars == transformed_jaxpr.eqns[1].outvars

    def test_expand_single_transform_partial_transform(self):
        """Test that a function that has a single transform is expanded correctly when only
        a part of the original function is transformed."""

        def f(x):
            qml.RX(x, 0)

            @qml.transforms.cancel_inverses
            def g():
                qml.X(0)
                qml.S(1)
                qml.X(0)
                qml.adjoint(qml.S(1))
                return qml.expval(qml.Z(0))

            return g()

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.transforms.cancel_inverses._primitive
        assert jaxpr.jaxpr.outvars == jaxpr.eqns[1].outvars

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 3
        assert transformed_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.jaxpr.outvars == transformed_jaxpr.eqns[2].outvars

    def test_expand_multiple_transforms_non_nested(self):
        """Test that a function that has multiple transforms that are not nested is
        expanded correctly."""

        def f(w, x, y, z):
            qml.RX(w, 0)

            @qml.transforms.cancel_inverses
            def g():
                qml.X(0)
                qml.S(1)
                qml.X(0)
                qml.adjoint(qml.S(1))
                return qml.expval(qml.Z(0))

            m1 = g()
            qml.RX(x, 0)

            @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
            def h(m, n, o):
                qml.Rot(m, n, o, 0)
                return qml.probs(wires=[0, 1])

            m2 = h(x, y, z)

            return m1, m2

        args = (1.2, 3.4, 5.6, 7.8)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.transforms.cancel_inverses._primitive
        assert jaxpr.eqns[2].primitive == qml.RX._primitive
        assert jaxpr.eqns[3].primitive == qml.transforms.decompose._primitive
        assert jaxpr.jaxpr.outvars == [jaxpr.eqns[1].outvars[0], jaxpr.eqns[3].outvars[0]]

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 8
        assert transformed_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.RX._primitive
        assert transformed_jaxpr.eqns[4].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[5].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[6].primitive == qml.RZ._primitive
        assert (
            transformed_jaxpr.eqns[7].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )
        assert transformed_jaxpr.jaxpr.outvars == [
            transformed_jaxpr.eqns[2].outvars[0],
            transformed_jaxpr.eqns[7].outvars[0],
        ]

    def test_expand_multiple_transforms_nested(self):
        """Test that a function that has multiple transforms that are nested is
        expanded correctly."""

        def f(w, x, y, z):
            qml.RX(w, 0)

            @qml.transforms.cancel_inverses
            def g(a, b, c):
                qml.X(0)
                qml.S(1)

                @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
                def h(m, n, o):
                    qml.Rot(m, n, o, 0)
                    return qml.probs(wires=[0, 1])

                qml.X(0)
                qml.adjoint(qml.S(1))
                return qml.expval(qml.Z(0)), h(a, b, c)

            return g(x, y, z)

        args = (1.2, 3.4, 5.6, 7.8)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.transforms.cancel_inverses._primitive
        inner_jaxpr = jaxpr.eqns[1].params["inner_jaxpr"]
        assert len(inner_jaxpr.eqns) == 8
        assert inner_jaxpr.eqns[-2].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert inner_jaxpr.eqns[-1].primitive == qml.transforms.decompose._primitive
        assert inner_jaxpr.outvars == [
            inner_jaxpr.eqns[-2].outvars[0],
            inner_jaxpr.eqns[-1].outvars[0],
        ]

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 7
        assert transformed_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.RZ._primitive
        assert transformed_jaxpr.eqns[4].primitive == qml.RY._primitive
        assert transformed_jaxpr.eqns[5].primitive == qml.RZ._primitive
        assert (
            transformed_jaxpr.eqns[6].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )
        assert transformed_jaxpr.jaxpr.outvars == [
            transformed_jaxpr.eqns[2].outvars[0],
            transformed_jaxpr.eqns[6].outvars[0],
        ]

    def test_expand_function_with_no_transforms(self):
        """Test that using expand_plxpr_transforms on a function with no transforms does
        not affect it."""

        def f(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 1)
            return qml.expval(qml.Z(1))

        args = (1.5, 2.6)
        jaxpr = jax.make_jaxpr(f)(*args)
        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(jaxpr.eqns) == len(transformed_jaxpr.eqns)
        assert all(
            eqn1.primitive == eqn2.primitive
            for eqn1, eqn2 in zip(jaxpr.eqns, transformed_jaxpr.eqns, strict=True)
        )
        assert all(
            eqn1.params == eqn2.params
            for eqn1, eqn2 in zip(jaxpr.eqns, transformed_jaxpr.eqns, strict=True)
        )
        assert jaxpr.jaxpr.outvars == jaxpr.eqns[-1].outvars
        assert transformed_jaxpr.jaxpr.outvars == transformed_jaxpr.eqns[-1].outvars
