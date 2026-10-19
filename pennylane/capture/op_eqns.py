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
This submodule removes captured operator equations from the active JAX tracing frames.
"""

from collections.abc import Iterable


def pop_op_eqns(ops: Iterable):
    """Delete the jaxpr equations for operators that have been used as data.

    When quantum operators are passed or manipulated as data structures (e.g., in a pytree),
    their operations must not execute as side-effecting quantum instructions during tracing.
    This function removes their corresponding equations from the active JAX tracing frames.

    All 'ops' are removed in one backward pass per frame, so the cost of removal is set
    by the deepest target and does not depend on the order of 'ops'.
    """
    old_eqns = []

    # Maps frame ID -> (frame_obj, set_of_target_equation_ids)
    # This allows us to sweep each frame's equation list only once.
    targets_by_frame = {}

    # Phase 1: Collect equation targets and group them by their frame
    for op in ops:
        if op.tracer is not None:
            # pylint: disable=protected-access
            frame = op.tracer._trace.frame
            assert frame.auto_dce is False  # eqns are stored differently if this is enabled

            # NOTE: JAX frame equation lists wrap equations in zero argument callables
            eqn = op.tracer.parent
            old_eqns.append(eqn)

            # Store ID of equation in the target set
            targets_by_frame.setdefault(id(frame), (frame, set()))[1].add(id(eqn))

            # Detach tracer from op to make a pure data object
            op.tracer = None

    # Phase 2: Sweep each frame's equation list backwards and perform batch deletion
    for frame, targets in targets_by_frame.values():
        eqns = frame.tracing_eqns

        # NOTE: Iterate backwards, target equations are almost always near the tail
        for i in reversed(range(len(eqns))):
            key = id(eqns[i]())

            if key in targets:
                targets.discard(key)
                del eqns[i]

                # All targets for this frame have been pruned, can terminate early
                if not targets:
                    break

    return old_eqns
