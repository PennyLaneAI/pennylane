# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains wrapper transforms to provide an API to bind primitives for
Catalyst passes when using capture and the unified compiler. This is a temporary fix
to manually add passes relevant for ongoing MBQC work. It can be removed one a more
general solution for all Catalyst passes is in place."""

from catalyst.from_plxpr import register_transform

from ..transforms.core import transform


@transform
def to_ppr(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError("The to_ppm pass is only implemented when using capture and QJIT.")


register_transform(to_ppr, "to-ppr", False)


@transform
def commute_ppr(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The commute_ppr pass is only implemented when using capture and QJIT."
    )


register_transform(commute_ppr, "commute-ppr", False)


@transform
def merge_ppr_ppm(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The merge_ppr_ppm pass is only implemented when using capture and QJIT."
    )


register_transform(merge_ppr_ppm, "merge-ppr-ppm", False)


@transform
def ppr_to_mbqc(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The ppm_to_mbqc pass is only implemented when using capture and QJIT."
    )


register_transform(ppr_to_mbqc, "ppr-to-mbqc", False)


@transform
def reduce_t_depth(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The reduce_t_depth pass is only implemented when using capture and QJIT."
    )


register_transform(reduce_t_depth, "reduce-t-depth", False)


@transform
def decompose_non_clifford_ppr(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The decompose_non_clifford_ppr pass is only implemented when using capture and QJIT."
    )


register_transform(decompose_non_clifford_ppr, "decompose-non-clifford-ppr", False)


@transform
def decompose_clifford_ppr(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The decompose_clifford_ppr pass is only implemented when using capture and QJIT."
    )


register_transform(decompose_clifford_ppr, "decompose-clifford-ppr", False)


@transform
def ppr_to_ppm(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The ppr_to_ppm pass is only implemented when using capture and QJIT."
    )


register_transform(ppr_to_ppm, "ppr-to-ppm", False)


@transform
def ppm_compilation(tape):
    """A wrapper that allows us to register a primitive that represents the transform during capture.
    The transform itself is only implemented in Catalyst. This is just to enable capture."""
    raise NotImplementedError(
        "The ppm_compilation pass is only implemented when using capture and QJIT."
    )


register_transform(ppm_compilation, "ppm-compilation", False)
