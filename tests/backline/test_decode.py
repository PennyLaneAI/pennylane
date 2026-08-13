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

# pylint: disable=redefined-outer-name

"""Tests for :mod:`pennylane.backline.decode`."""

from importlib import import_module

import numpy as np
import pytest

import pennylane as qp

decode_mod = import_module("pennylane.backline.decode")


@pytest.fixture
def x64():
    """Run a test with 64-bit values available, as Catalyst configures JAX."""
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        yield jax


class TestDecodeBitpack:
    """Checks for decode(..., bitpack=True)."""

    @staticmethod
    def _nodes():
        controller = qp.Controller()
        coprocessor = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1")
        return controller, coprocessor

    def test_bitpack_decode_returns_64_bits(self, x64):
        """It should unpack the collected 8-byte reply into a 64-bit vector."""
        controller, coprocessor = self._nodes()
        jaxpr = x64.make_jaxpr(
            lambda a, b: decode_mod.decode(
                (a, b), controller=controller, coprocessor=coprocessor, bitpack=True
            )
        )(np.uint8(1), np.uint8(0))

        avals = [v.aval for v in jaxpr.jaxpr.outvars]
        assert [tuple(a.shape) for a in avals] == [(64,)]
        assert [a.dtype for a in avals] == [np.dtype(bool)]

        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert calls[-1].params["out_bytes"] == (8,)

    def test_bitpack_decode_rejects_non_vector_input(self):
        """It should require a 1D syndrome bit vector in packed mode."""
        controller, coprocessor = self._nodes()

        with pytest.raises(ValueError, match="1D bit vector"):
            decode_mod.decode(
                np.uint64(1),
                controller=controller,
                coprocessor=coprocessor,
                bitpack=True,
            )

    def test_bitpack_decode_rejects_vectors_longer_than_u64(self):
        """It should cap packed syndromes at 64 bits."""
        controller, coprocessor = self._nodes()

        with pytest.raises(ValueError, match="at most 64 bits"):
            decode_mod.decode(
                np.ones(65, dtype=np.uint8),
                controller=controller,
                coprocessor=coprocessor,
                bitpack=True,
            )

    @pytest.mark.parametrize(
        ("name", "kwargs"), [("in_bytes", {"in_bytes": 4}), ("out_bytes", {"out_bytes": 4})]
    )
    def test_bitpack_decode_requires_u64_sized_buffers(self, name, kwargs):
        """It should refuse packed transport sizes other than one u64."""
        controller, coprocessor = self._nodes()

        with pytest.raises(ValueError, match=rf"{name}=8"):
            decode_mod.decode(
                np.array([1], dtype=np.uint8),
                controller=controller,
                coprocessor=coprocessor,
                bitpack=True,
                **kwargs,
            )
