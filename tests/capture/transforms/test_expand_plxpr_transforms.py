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

import pennylane as qp

jax = pytest.importorskip("jax")

from pennylane.capture import expand_plxpr_transforms
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
from pennylane.capture.primitives import transform_prim

pytestmark = [pytest.mark.jax, pytest.mark.capture]


@qp.transform
def dummy_tape_only_transform(tape):
    return [tape], lambda res: res[0]


def _dummy_plxpr_transform(jaxpr, consts, targs, tkwargs, *args):  # pylint: disable=unused-argument
    """Dummy function to transform plxpr."""

    def wrapper(*inner_args):
        return jax.core.eval_jaxpr(jaxpr, consts, *inner_args)

    return jax.make_jaxpr(wrapper)(*args)


@partial(qp.transform, plxpr_transform=_dummy_plxpr_transform)
def dummy_tape_and_plxpr_transform(tape):
    return [tape], lambda res: res[0]


class TestExpandTransformsInterpreter:
    """Unit tests for ExpandTransformsInterpreter"""

    def test_expand_transforms_interpreter_registration(self):
        """Test that the primitives of PennyLane transforms are automatically registered with the
        ExpandTransformsInterpreter."""

        assert transform_prim in ExpandTransformsInterpreter._primitive_registrations

    def test_expand_transforms_interpreter_plxpr_transform(self):
        """Test that transforms that have a valid ``plxpr_transform`` are handled
        correctly."""

        custom_handler = ExpandTransformsInterpreter._primitive_registrations[transform_prim]
        assert dummy_tape_and_plxpr_transform.plxpr_transform is not None

        def f(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(0))

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)

        def wrapper(*inner_args):
            interpreter = ExpandTransformsInterpreter()
            invals = [*inner_args, *jaxpr.consts]
            params = {
                "inner_jaxpr": jaxpr.jaxpr,
                "args_slice": (0, len(inner_args), None),
                "consts_slice": (len(inner_args), len(jaxpr.consts) + len(inner_args), None),
                "targs_slice": (len(jaxpr.consts) + len(inner_args), None, None),
                "tkwargs": {},
                "transform": dummy_tape_and_plxpr_transform,
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

        @qp.transforms.cancel_inverses
        def f():
            qp.X(0)
            qp.S(1)
            qp.X(0)
            qp.adjoint(qp.S(1))
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == transform_prim
        assert jaxpr.eqns[0].params["transform"] == qp.transforms.cancel_inverses
        assert jaxpr.jaxpr.outvars == jaxpr.eqns[0].outvars

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()
        assert len(transformed_jaxpr.eqns) == 2
        assert transformed_jaxpr.eqns[0].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[1].primitive == qp.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.jaxpr.outvars == transformed_jaxpr.eqns[1].outvars

    def test_expand_single_transform_partial_transform(self):
        """Test that a function that has a single transform is expanded correctly when only
        a part of the original function is transformed."""

        def f(x):
            qp.RX(x, 0)

            @qp.transforms.cancel_inverses
            def g():
                qp.X(0)
                qp.S(1)
                qp.X(0)
                qp.adjoint(qp.S(1))
                return qp.expval(qp.Z(0))

            return g()

        args = (1.5,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == transform_prim
        assert jaxpr.eqns[1].params["transform"] == qp.transforms.cancel_inverses
        assert jaxpr.jaxpr.outvars == jaxpr.eqns[1].outvars

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 3
        assert transformed_jaxpr.eqns[0].primitive == qp.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.jaxpr.outvars == transformed_jaxpr.eqns[2].outvars

    def test_expand_multiple_transforms_non_nested(self):
        """Test that a function that has multiple transforms that are not nested is
        expanded correctly."""

        def f(w, x, y, z):
            qp.RX(w, 0)

            @qp.transforms.cancel_inverses
            def g():
                qp.X(0)
                qp.S(1)
                qp.X(0)
                qp.adjoint(qp.S(1))
                return qp.expval(qp.Z(0))

            m1 = g()
            qp.RX(x, 0)

            @qp.transforms.decompose(gate_set=[qp.RX, qp.RY, qp.RZ])
            def h(m, n, o):
                qp.Rot(m, n, o, 0)
                return qp.probs(wires=[0, 1])

            m2 = h(x, y, z)

            return m1, m2

        args = (1.2, 3.4, 5.6, 7.8)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == transform_prim
        assert jaxpr.eqns[1].params["transform"] == qp.transforms.cancel_inverses
        assert jaxpr.eqns[2].primitive == qp.RX._primitive
        assert jaxpr.eqns[3].primitive == transform_prim
        assert jaxpr.eqns[3].params["transform"] == qp.transforms.decompose
        assert jaxpr.jaxpr.outvars == [jaxpr.eqns[1].outvars[0], jaxpr.eqns[3].outvars[0]]

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 8
        assert transformed_jaxpr.eqns[0].primitive == qp.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.eqns[3].primitive == qp.RX._primitive
        assert transformed_jaxpr.eqns[4].primitive == qp.RZ._primitive
        assert transformed_jaxpr.eqns[5].primitive == qp.RY._primitive
        assert transformed_jaxpr.eqns[6].primitive == qp.RZ._primitive
        assert (
            transformed_jaxpr.eqns[7].primitive == qp.measurements.ProbabilityMP._wires_primitive
        )
        assert transformed_jaxpr.jaxpr.outvars == [
            transformed_jaxpr.eqns[2].outvars[0],
            transformed_jaxpr.eqns[7].outvars[0],
        ]

    def test_expand_multiple_transforms_nested(self):
        """Test that a function that has multiple transforms that are nested is
        expanded correctly."""

        def f(w, x, y, z):
            qp.RX(w, 0)

            @qp.transforms.cancel_inverses
            def g(a, b, c):
                qp.X(0)
                qp.S(1)

                @qp.transforms.decompose(gate_set=[qp.RX, qp.RY, qp.RZ])
                def h(m, n, o):
                    qp.Rot(m, n, o, 0)
                    return qp.probs(wires=[0, 1])

                qp.X(0)
                qp.adjoint(qp.S(1))
                return qp.expval(qp.Z(0)), h(a, b, c)

            return g(x, y, z)

        args = (1.2, 3.4, 5.6, 7.8)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == transform_prim
        assert jaxpr.eqns[1].params["transform"] == qp.transforms.cancel_inverses
        inner_jaxpr = jaxpr.eqns[1].params["inner_jaxpr"]
        assert len(inner_jaxpr.eqns) == 8
        assert inner_jaxpr.eqns[-2].primitive == qp.measurements.ExpectationMP._obs_primitive
        assert inner_jaxpr.eqns[-1].primitive == transform_prim
        assert inner_jaxpr.eqns[-1].params["transform"] == qp.transforms.decompose
        assert inner_jaxpr.outvars == [
            inner_jaxpr.eqns[-2].outvars[0],
            inner_jaxpr.eqns[-1].outvars[0],
        ]

        transformed_f = expand_plxpr_transforms(f)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)(*args)
        assert len(transformed_jaxpr.eqns) == 7
        assert transformed_jaxpr.eqns[0].primitive == qp.RX._primitive
        assert transformed_jaxpr.eqns[1].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[2].primitive == qp.measurements.ExpectationMP._obs_primitive
        assert transformed_jaxpr.eqns[3].primitive == qp.RZ._primitive
        assert transformed_jaxpr.eqns[4].primitive == qp.RY._primitive
        assert transformed_jaxpr.eqns[5].primitive == qp.RZ._primitive
        assert (
            transformed_jaxpr.eqns[6].primitive == qp.measurements.ProbabilityMP._wires_primitive
        )
        assert transformed_jaxpr.jaxpr.outvars == [
            transformed_jaxpr.eqns[2].outvars[0],
            transformed_jaxpr.eqns[6].outvars[0],
        ]

    def test_expand_function_with_no_transforms(self):
        """Test that using expand_plxpr_transforms on a function with no transforms does
        not affect it."""

        def f(x, y):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            qp.RY(y, 1)
            return qp.expval(qp.Z(1))

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
