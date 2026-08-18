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

import numpy as np
import pytest

import pennylane as qp
from pennylane.backline.decode import (
    ROLE_CONTROLLER,
    _byte_count,
    _resolve_nodes,
    _resolve_out_bytes,
    _session_key,
    decode,
)
from pennylane.backline.runtime import operands

TRANSPORT_CALLS = [
    "__catalyst__transport__get_session__call",
    "__catalyst__transport__stage_payload__call",
    "__catalyst__transport__post__call",
    "__catalyst__transport__collect__call",
]


@pytest.fixture(name="x64")
def x64_fixture():
    """Run a test with 64-bit values available, as Catalyst configures JAX."""
    jax = pytest.importorskip("jax")
    with jax.experimental.enable_x64():
        yield jax


class TestDecodeBitpack:
    """Checks for decode(..., bitpack=True)."""

    @staticmethod
    def _nodes():
        controller = qp.Controller()
        coprocessor = qp.Coprocessor(
            coprocessor_fn="decoder", endpoint=qp.Endpoint("127.0.0.1", 7760)
        )
        return controller, coprocessor

    def test_bitpack_decode_returns_64_bits(self, x64):
        """It should unpack the collected 8-byte reply into a 64-bit vector."""
        controller, coprocessor = self._nodes()
        jaxpr = x64.make_jaxpr(
            lambda a, b: decode(
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
            decode(
                np.uint64(1),
                controller=controller,
                coprocessor=coprocessor,
                bitpack=True,
            )

    def test_bitpack_decode_rejects_vectors_longer_than_u64(self):
        """It should cap packed syndromes at 64 bits."""
        controller, coprocessor = self._nodes()

        with pytest.raises(ValueError, match="at most 64 bits"):
            decode(
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
            decode(
                np.array([1], dtype=np.uint8),
                controller=controller,
                coprocessor=coprocessor,
                bitpack=True,
                **kwargs,
            )


def a_coprocessor(name=None):
    """A coprocessor"""
    return qp.Coprocessor(
        coprocessor_fn="decoder", name=name, endpoint=qp.Endpoint("127.0.0.1", 7760)
    )


def a_device(coprocessors=()):
    """A backline device for decode rounds."""
    controller = qp.Controller(device=qp.device("null.qubit", wires=2))
    return qp.Backline(controller=controller, coprocessors=coprocessors, transport="rdma")


def a_round(jax, dev, syndrome=None, **kwargs):
    """Trace a round on ``dev`` and return its jaxpr."""
    syndrome = np.zeros(4, dtype=np.uint8) if syndrome is None else syndrome
    kwargs.setdefault("bitpack", False)
    with qp.capture.tracing_device(dev):
        return jax.make_jaxpr(lambda s: decode(s, **kwargs))(syndrome)


def calls_of(jaxpr):
    """The transport calls."""
    return [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]


def scalars_of(jaxpr, call):
    """The compile-time scalars a call was given."""
    literals = {
        eqn.outvars[0]: eqn.invars[0].val
        for eqn in jaxpr.eqns
        if str(eqn.primitive) == "reshape" and hasattr(eqn.invars[0], "val")
    }
    return [literals.get(var) for var in call.invars]


def session_key_of(jaxpr):
    """The session key."""
    fields = [
        np.asarray(const)
        for const in jaxpr.consts
        if np.asarray(const).shape == (operands.STR_OPERAND_BYTES,)
    ]
    assert len(fields) == 1, "expected exactly one str operand"
    return bytes(fields[0]).rstrip(b"\x00").decode()


class TestSessionKey:
    """The session key."""

    def test_a_placement_with_no_coprocessor_uses_the_controller_key(self):
        """The only session is the controller's own."""
        assert _session_key(None) == "controller"

    def test_a_named_coprocessor_names_the_session(self):
        """One session per coprocessor, by that coprocessor's name."""
        assert _session_key(a_coprocessor(name="decoder-0")) == "decoder-0"

    @pytest.mark.parametrize("name", [None, ""])
    def test_an_unnamed_coprocessor_falls_back(self, name):
        """Without a name there is one conventional key."""
        coprocessor = a_coprocessor(name=name)
        assert _session_key(coprocessor) == "coprocessor.0"
        assert _session_key(coprocessor, [coprocessor]) == "coprocessor.0"

    def test_an_unnamed_coprocessor_among_several_is_refused(self):
        """Unnamed coprocessors among several cannot be routed."""
        coprocessors = [a_coprocessor(), a_coprocessor()]
        with pytest.raises(ValueError, match="no name, and the placement has 2"):
            _session_key(coprocessors[1], coprocessors)

    def test_a_name_is_enough_to_route_several(self):
        """A name is enough to route several coprocessors."""
        coprocessors = [a_coprocessor(name="decoder-0"), a_coprocessor(name="decoder-1")]
        assert _session_key(coprocessors[1], coprocessors) == "decoder-1"

    def test_a_controller_only_placement_is_unaffected(self):
        """A controller-only placement has no key to get wrong."""
        assert _session_key(None, [a_coprocessor(), a_coprocessor()]) == "controller"


class TestByteCount:
    """The byte count."""

    @pytest.mark.parametrize(
        "array, expected",
        [
            (np.zeros(4, dtype=np.uint8), 4),
            (np.zeros((2, 3), dtype=np.float64), 48),
            (np.zeros((), dtype=np.uint32), 4),
            ([1.0, 2.0, 3.0], 24),
        ],
    )
    def test_the_count_comes_from_shape_and_dtype(self, array, expected):
        """The count comes from shape and dtype."""
        assert _byte_count(array) == expected

    def test_a_traced_syndrome_is_measured_by_its_aval(self, x64):
        """The count is known at trace time."""
        counted = []
        x64.make_jaxpr(lambda s: counted.append(_byte_count(s)) or s)(
            np.zeros((2, 8), dtype=np.uint16)
        )
        assert counted == [32]


class TestOutBytes:
    """The correction size."""

    def test_an_explicit_size_wins(self):
        """The call site's override wins."""
        controller = qp.Controller()
        assert _resolve_out_bytes(controller, 16) == 16

    def test_the_committed_size_is_the_default(self):
        """The committed size is the default."""
        assert _resolve_out_bytes(qp.Controller(), None) == 8

    @pytest.mark.parametrize("controller", [qp.Controller(), object()])
    def test_the_transport_default_is_eight_bytes(self, controller):
        """An unconfigured reply uses the transport's default message size."""
        assert _resolve_out_bytes(controller, None) == 8


class TestNodeResolution:
    """Which nodes a round runs between."""

    def test_explicit_nodes_need_no_placement(self):
        """Both nodes given are self-contained, so no device has to be traced."""
        controller, coprocessor = qp.Controller(), a_coprocessor()
        assert _resolve_nodes(controller, coprocessor) == (controller, coprocessor)

    def test_the_nodes_come_from_the_traced_device(self):
        """The placement supplies both nodes."""
        dev = a_device(coprocessors=[a_coprocessor(name="decoder-0")])
        with qp.capture.tracing_device(dev):
            controller, coprocessor = _resolve_nodes(None, None)

        assert controller is dev.placement.controller
        assert coprocessor is dev.placement.coprocessors[0]

    def test_an_explicit_coprocessor_selects_the_node(self):
        """An explicit coprocessor picks the transport session."""
        coprocs = [a_coprocessor(name="decoder-0"), a_coprocessor(name="decoder-1")]
        with qp.capture.tracing_device(a_device(coprocessors=coprocs)):
            _, coprocessor = _resolve_nodes(None, coprocs[1])

        assert coprocessor is coprocs[1]

    def test_multiple_coprocessors_need_an_explicit_node(self):
        """A decoder ID does not select among transport sessions."""
        coprocs = [a_coprocessor(name="decoder-0"), a_coprocessor(name="decoder-1")]
        with qp.capture.tracing_device(a_device(coprocessors=coprocs)):
            with pytest.raises(ValueError, match="with multiple coprocessors"):
                _resolve_nodes(None, None)

    def test_a_placement_without_coprocessors_is_refused(self):
        """A decoding round requires a coprocessor."""
        dev = a_device()
        with qp.capture.tracing_device(dev):
            with pytest.raises(ValueError, match="with multiple coprocessors"):
                _resolve_nodes(None, None)

    @pytest.mark.parametrize("device", [None, qp.device("null.qubit", wires=1)])
    def test_a_trace_with_no_placement_is_refused(self, device):
        """A trace without a placement cannot resolve nodes."""
        with qp.capture.tracing_device(device):
            with pytest.raises(ValueError, match="this trace has none"):
                _resolve_nodes(None, None)


class TestRecordedRound:
    """The four transport calls a round is recorded as."""

    def test_a_round_is_four_calls_in_order(self, x64):
        """The calls are in order."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor(name="decoder-0")]))
        assert [call.params["symbol"] for call in calls_of(jaxpr)] == TRANSPORT_CALLS

    def test_every_call_is_local(self, x64):
        """The round is driven from the controller's own process."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor()]))
        assert all(call.params["dispatch"] is None for call in calls_of(jaxpr))

    def test_the_session_is_claimed_as_the_controller(self, x64):
        """The controller is the data initiator."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor(name="decoder-0")]))
        role, _key = scalars_of(jaxpr, calls_of(jaxpr)[0])

        assert role == ROLE_CONTROLLER
        assert session_key_of(jaxpr) == "decoder-0"

    @pytest.mark.parametrize(
        "coprocessors, key",
        [
            ([a_coprocessor(name="decoder-1")], "decoder-1"),
            ([a_coprocessor()], "coprocessor.0"),
        ],
    )
    def test_the_session_key_follows_the_placement(self, x64, coprocessors, key):
        """The key follows the placement."""
        jaxpr = a_round(x64, a_device(coprocessors=coprocessors))
        assert session_key_of(jaxpr) == key

    def test_a_second_decoder_needs_a_name(self, x64):
        """A second decoder needs a name to choose it."""
        dev = a_device(coprocessors=[a_coprocessor(), a_coprocessor()])
        with pytest.raises(ValueError, match="no name, and the placement has 2"):
            a_round(x64, dev, coprocessor=dev.coprocessors[1], decoder_id=1)

    def test_the_staged_payload_carries_its_length_and_decoder(self, x64):
        """The length and decoder are stamped alongside the payload."""
        coprocs = [a_coprocessor(name="decoder-0"), a_coprocessor(name="decoder-1")]
        dev = a_device(coprocessors=coprocs)
        jaxpr = a_round(
            x64,
            dev,
            syndrome=np.zeros(6, dtype=np.uint8),
            coprocessor=coprocs[1],
            decoder_id=1,
        )
        _session, _src, nbytes, decoder_id = scalars_of(jaxpr, calls_of(jaxpr)[1])

        assert nbytes == 6
        assert decoder_id == 1

    def test_in_bytes_sends_only_part_of_the_syndrome(self, x64):
        """The committed bytes are sent."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor()]), in_bytes=2)
        assert scalars_of(jaxpr, calls_of(jaxpr)[1])[2] == 2

    def test_the_posted_work_item_idx_is_the_one_asked_for(self, x64):
        """The caller's choice, defaulting to the first."""
        dev = a_device(coprocessors=[a_coprocessor()])

        default = a_round(x64, dev)
        assert scalars_of(default, calls_of(default)[2])[1] == 0

        asked = a_round(x64, dev, work_item_idx=3)
        assert scalars_of(asked, calls_of(asked)[2])[1] == 3

    def test_the_correction_is_the_committed_size(self, x64):
        """The correction is the committed size."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor()]))
        collect = calls_of(jaxpr)[3]

        assert collect.params["out_bytes"] == (8,)
        (correction,) = jaxpr.jaxpr.outvars
        assert correction.aval.shape == (8,)
        assert str(correction.aval.dtype) == "uint8"

    def test_an_unconfigured_correction_uses_the_transport_default(self, x64):
        """The Python return shape agrees with the transport dialect's 8-byte default."""
        dev = qp.Backline(
            controller=qp.Controller(device=qp.device("null.qubit", wires=2)),
            coprocessors=[a_coprocessor()],
            transport="rdma",
        )
        jaxpr = a_round(x64, dev)

        assert calls_of(jaxpr)[3].params["out_bytes"] == (8,)

    def test_an_explicit_size_overrides_the_committed_one(self, x64):
        """The call site's override wins."""
        jaxpr = a_round(x64, a_device(coprocessors=[a_coprocessor()]), out_bytes=32)
        assert calls_of(jaxpr)[3].params["out_bytes"] == (32,)

    def test_the_library_is_recorded_on_every_call(self, x64):
        """The library is recorded once per call."""
        dev = a_device(coprocessors=[a_coprocessor()])
        jaxpr = a_round(x64, dev, library="/opt/librt_transport.so")

        libraries = {call.params["library"] for call in calls_of(jaxpr)}
        assert libraries == {"/opt/librt_transport.so"}

    def test_a_round_outside_a_program_is_refused(self):
        """A recorded call has nowhere to go without a trace."""
        pytest.importorskip("jax")
        with pytest.raises(RuntimeError, match="outside a compiled program"):
            decode(
                np.zeros(4, dtype=np.uint8),
                controller=qp.Controller(),
                coprocessor=a_coprocessor(),
            )
