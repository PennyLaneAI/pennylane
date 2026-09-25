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
"""
.. currentmodule:: pennylane.gadget

This module contains functionality for writing fault-tolerant gadgets on CSS codes in
Python, checking them, and emitting them into Catalyst's IR.

A fault-tolerant gadget, such as a lattice-surgery merge, a logical measurement or a
memory, is a schedule of stabilizer measurements. Running it correctly also requires the
detectors a decoder uses, and the logical operation the surrounding program relies on. A
gate-level description of a gadget holds only the schedule, and a gadget with the wrong
number of rounds or an outcome read from the wrong parity looks no different from a
correct one. This module keeps the three together, derives the detectors and outcome
parities from the schedule, and checks them against the declared logical operation.

.. warning::

    This module is experimental. Its API may change without notice between releases.

.. note::

    The module requires only NumPy. :func:`verify` also runs a noise simulation when
    `Stim <https://github.com/quantumlib/Stim>`__ is installed, and
    :mod:`pennylane.gadget.lowering` requires xDSL and Catalyst's dialect definitions.

For example, the library gadget :func:`~.library.rep_code_zz_merge` measures logical
:math:`Z \\otimes Z` on two distance-3 repetition codes by merging them. Verifying it
derives 22 detectors, reports the one record that has no detector, and passes every
algebraic check:

>>> from pennylane import gadget
>>> from pennylane.gadget.library import rep_code_zz_merge
>>> code, phases, measure_zz = rep_code_zz_merge(d=3)
>>> receipt, layout = gadget.verify(measure_zz.program, simulate=False)
>>> receipt.ok, layout.n_detectors
(True, 22)
>>> [name for name, _ in layout.undetermined]
['merged[r0,c4]']

Defining gadgets
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~define
    ~TracedGadget
    ~Action
    ~rounds
    ~deform
    ~detach
    ~observe
    ~frame
    ~unroll
    ~Outcome

A gadget is written as a Python function decorated with :func:`define`. The decorator is
given the code of the encoded qubits, the :class:`Action` the gadget performs, and every
:class:`Phase` it may measure. The function receives a :class:`Handle` on the encoded
qubits and uses five operations:

- :func:`rounds` measures the current phase's checks for a fixed number of rounds and
  returns a :class:`RecordBlock` of outcomes.
- :func:`deform` switches to another phase, for example one with extra checks that merge
  two blocks. Qubits it activates must be given a preparation basis.
- :func:`detach` switches to another phase and reads out the qubits it deactivates.
- :func:`observe` declares a parity of outcomes as one of the gadget's measurement
  outcomes.
- :func:`frame` applies one of the Pauli frame updates declared in :func:`define`,
  conditioned on a parity.

.. code-block:: python

    from pennylane import gadget

    code, (base, merged), _ = rep_code_zz_merge(d=3)

    @gadget.define(
        action=gadget.Action.measure(("z", (0, 1))),
        code=code,
        phases=(base, merged),
        claims=(gadget.DistanceClaim(3, "phenomenological"),),
    )
    def measure_zz(handle):
        handle, _ = gadget.rounds(handle, 1, record="pre")
        handle = gadget.deform(handle, to="merged")
        handle, checks = gadget.rounds(handle, 3, record="merged")
        outcome = gadget.observe(checks.product((4,)), index=0)
        handle = gadget.deform(handle, to="base")
        handle, _ = gadget.rounds(handle, 1, record="post")
        return gadget.frame(handle, outcome), outcome

The body is traced once, when it is decorated. Two rules make the traced program
checkable:

- **Phases are declared, not built.** A body can only choose which declared phase to
  measure, for how many rounds, and how to combine outcomes. It cannot construct a check,
  so every later analysis works on a fixed set of matrices. Round counts must be Python
  integers for the same reason.
- **Handles are single-use.** Every operation consumes the handle it is given and returns
  a new one. Using a consumed handle raises :class:`OwnershipError`, so a body cannot act
  twice on the same encoded state.

A :class:`TracedGadget` can be called inside another gadget's body, which copies its
operations there with record names prefixed by ``name#n`` for the ``n``-th call. The
called gadget's outcomes are returned with ``index=None``; they become outcomes of the
enclosing gadget only when passed to :func:`observe`, which lets the caller choose which
outcomes to declare and in which order. Frame updates declared by the called gadget are
carried into the caller's ``frame_update`` matrix. :func:`unroll` repeats a body a fixed
number of times.

>>> @gadget.define(
...     action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases
... )
... def measure_twice_keep_last(handle):
...     handle, _ = measure_zz(handle)
...     handle, (second,) = measure_zz(handle)
...     return handle, gadget.observe(second, index=0)
>>> [r.name for r in measure_twice_keep_last.records][3:]
['measure_zz#1/pre', 'measure_zz#1/merged', 'measure_zz#1/post']

Codes and phases
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~CSSCode
    ~Phase
    ~DistanceClaim
    ~CodeError

A :class:`CSSCode` holds X and Z check matrices and logical operators, and is checked for
consistency when it is created. Logical qubit ``i`` is row ``i`` of ``lx`` and ``lz``, and
gadgets refer to logical qubits by this index.

A :class:`Phase` is a stabilizer group over a qubit frame shared by all phases of a gadget:
the data qubits first, then any auxiliary qubits. Qubits outside the phase's ``active``
mask must not be touched by its checks.

A :class:`DistanceClaim` records a distance together with its regime (``"static"``,
``"phenomenological"`` or ``"circuit"``) and whether it was certified. The distance of a
code says nothing on its own about the fault distance of a gadget built from it.

Traced programs
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~GadgetProgram
    ~Handle
    ~RecordBlock
    ~RecordExpr
    ~entry_syndrome
    ~GadgetError
    ~OwnershipError

A :class:`GadgetProgram` is the traced form of a gadget, consumed by every later step. It
contains no Python callables, and :meth:`~.GadgetProgram.fingerprint` hashes its code,
phases and operations so that results derived from it can be checked against it later.

Each :func:`rounds` or :func:`detach` produces a :class:`RecordBlock` named in the body.
Outcomes are addressed by block, round and check rather than by Python variable, and
:class:`RecordExpr` parities are combined with ``^``. :func:`entry_syndrome` refers to the
check outcomes the encoded qubits carried when the gadget started.

Detectors and outcomes
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~derive_detectors
    ~DetectorLayout
    ~Detector
    ~LogicalObservable

:func:`derive_detectors` walks the program, keeping for every operator whose value is known
the parity that last determined it:

- The first round of the entry phase is compared with the entry syndrome, since the gadget
  is only deterministic relative to the state it receives. A caller that composes gadgets
  supplies these bits from its own records.
- A check measured for the first time whose operator is not known, such as the join check
  of a merge, has a random first outcome. It gets no first-round detector and is listed in
  :attr:`~.DetectorLayout.undetermined`, so a decoder is not told to expect determinism
  there.
- Each declared outcome is completed. If its parity differs from the declared logical
  operator by operators whose values are known, their parities are added, so the observable
  measures the declared operator exactly and any representative of the right coset can be
  declared. A parity whose difference is not yet known is rejected.

In the merge above, the declared outcome is the join check alone. Completion adds the two
base checks of the first block, so the observable measures :math:`Z_0 Z_3`:

>>> (obs,) = layout.observables
>>> obs.expr.describe()
'merged[r2,c0] ^ merged[r2,c1] ^ merged[r2,c4]'

Verification
~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~verify
    ~Receipt
    ~Check

:func:`verify` returns a :class:`Receipt` of named checks, each passed, warned, skipped or
failed with a stated reason. It checks that the detector layout matches the program, that
the entry phase accepts the code, that each deformation's prepared qubits are checked, that
a merged phase removes one logical degree of freedom per outcome, and that each completed
outcome measures exactly the declared logical operator. A check that cannot run is
reported as skipped, not passed.

With Stim installed, :func:`verify` also simulates the gadget under phenomenological
noise (see :mod:`pennylane.gadget.simulate`). It confirms that every derived detector is
deterministic without noise, and searches for the smallest undetectable error that flips an
outcome. The distance found is added to the receipt as a certified claim, and a larger
phenomenological claim by the author fails. For the merge above, the certified distance is
``min(merged_rounds, d)``: because the join check has no first-round detector, flipping its
outcome in every merged round is undetectable until the number of rounds reaches ``d``.

Scheduling
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~schedule_phase
    ~schedule_gadget
    ~Schedule
    ~GadgetSchedule

:func:`schedule_phase` splits one round of check measurements into layers in which no
check and no qubit interacts twice. The number of layers equals the lower bound
``max(check weight, qubit degree)``.

Toolchain support
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~check_support
    ~Toolchain
    ~SupportReport
    ~Gap

:func:`check_support` reports what a :class:`Toolchain` is missing to compile a gadget, as
blocking or degrading :class:`Gap` objects. :data:`~.support.CATALYST_CURRENT` describes
the current Catalyst QEC pipeline and Backline real-time path, with each limit backed by an
entry of :data:`~.support.CATALYST_EVIDENCE`; :data:`~.support.CATALYST_PROPOSED` describes
the pipeline with those limits lifted. The report never changes the gadget to fit.

Emission into Catalyst IR
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~lowering.emit
    ~lowering.lower_gadget_to_qecl
    ~lowering.LoweringGap

:func:`~.lowering.emit` writes a program into the proposed ``gadget`` xDSL dialect over
Catalyst's ``!qecl.codeblock<k>`` type:

- ``gadget.phase`` declares each phase once, at module scope.
- ``gadget.rounds``, ``gadget.deform`` and ``gadget.detach`` act on the codeblock;
  ``gadget.rounds`` and ``gadget.detach`` return ``!gadget.records<rounds x width>``
  values, and round counts are attributes.
- ``gadget.observable`` and ``gadget.frame_update`` refer to records by
  ``[operand, round, check]`` triples. Outcomes are emitted with their completed parities,
  and a frame update names a row of the declared matrix rather than a Pauli operator.
- ``gadget.detectors`` holds the detector and observable matrices over the flat record
  index, and the indices of undetermined records.

:func:`~.lowering.lower_gadget_to_qecl` lowers single-phase gadgets on ``k=1`` codeblocks
into ``qecl.qec`` cycles. Anything else raises :class:`~.lowering.LoweringGap` naming the
missing capability, rather than being approximated.

Example gadgets
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~library.rep_code_zz_merge
    ~library.steane_memory
    ~library.steane_code
    ~library.repetition_code
    ~library.rep_chain

Limitations
~~~~~~~~~~~

- Stabilizer groups cannot depend on runtime values. A program can choose between
  gadgets with declared phases, but cannot construct a new phase while it runs.
- :func:`unroll` expands every iteration, so a long memory produces one operation and one record block
  per iteration.
- The ``!qecl.codeblock`` type does not record the current phase, so the IR does not
  prevent a transversal gate during a merged phase.
- The simulation is phenomenological, not circuit-level, and includes only the declared
  outcomes as observables. It starts from the all-zero state, so a detector that is
  deterministic only for that state (for example a wrongly added first-round detector on a
  Z-type join check) is not caught.
- ``k.gauged`` assumes all outcomes come from one merge. It fails for a gadget that
  measures the same operator in sequence, and it does not flag a merged phase whose ``k``
  equals the entry ``k``.
- Emitted outcome and frame-update parities must come from a single record block.
- Only single-phase ``k=1`` gadgets whose records are not used later lower to ``qecl``.
  Lowering more needs ``qecl.qec`` to return its syndrome and a ``qecl`` operation that
  changes the measured stabilizer group.
"""

from . import codes, detectors, ir, library, schedule, support
from .authoring import (
    Outcome,
    TracedGadget,
    define,
    deform,
    detach,
    frame,
    observe,
    unroll,
    rounds,
)
from .codes import CodeError, CSSCode, DistanceClaim
from .detectors import Detector, DetectorLayout, LogicalObservable, derive_detectors
from .ir import (
    Action,
    GadgetError,
    GadgetProgram,
    Handle,
    OwnershipError,
    Phase,
    RecordBlock,
    RecordExpr,
    entry_syndrome,
)
from .schedule import GadgetSchedule, Schedule, schedule_gadget, schedule_phase
from .support import (
    CATALYST_CURRENT,
    CATALYST_PROPOSED,
    Gap,
    SupportReport,
    Toolchain,
    check_support,
)
from .verify import Check, Receipt, verify

__all__ = [
    "CSSCode",
    "CodeError",
    "DistanceClaim",
    "Phase",
    "Handle",
    "RecordExpr",
    "RecordBlock",
    "GadgetProgram",
    "GadgetError",
    "OwnershipError",
    "Action",
    "entry_syndrome",
    "TracedGadget",
    "Outcome",
    "define",
    "rounds",
    "deform",
    "detach",
    "observe",
    "frame",
    "unroll",
    "Detector",
    "DetectorLayout",
    "LogicalObservable",
    "derive_detectors",
    "Check",
    "Receipt",
    "verify",
    "Schedule",
    "GadgetSchedule",
    "schedule_phase",
    "schedule_gadget",
    "Gap",
    "Toolchain",
    "SupportReport",
    "check_support",
    "CATALYST_CURRENT",
    "CATALYST_PROPOSED",
    "codes",
    "ir",
    "detectors",
    "schedule",
    "support",
    "library",
]
