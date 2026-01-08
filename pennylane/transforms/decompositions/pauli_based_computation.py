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
"""Aliases for pauli-based computation passes from Catalyst's passes module."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="to-ppr")
def to_ppr(tape):
    r"""A quantum compilation pass that converts Clifford+T gates into Pauli Product Rotation (PPR)
    gates.
    """

    raise NotImplementedError(
        "The to_ppr compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="commute-ppr")
def commute_ppr(tape, *, max_pauli_size=0):
    r"""A quantum compilation pass that commutes Clifford Pauli product rotation (PPR) gates,
    :math:`\exp(-{iP\tfrac{\pi}{4}})`, past non-Clifford PPRs gates,
    :math:`\exp(-{iP\tfrac{\pi}{8}})`, where :math:`P` is a Pauli word.
    """
    raise NotImplementedError(
        "The commute_ppr compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="merge-ppr-ppm")
def merge_ppr_ppm(tape=None, *, max_pauli_size=0):
    r"""
    A quantum compilation pass that absorbs Clifford Pauli product rotation (PPR) operations,
    :math:`\exp{-iP\tfrac{\pi}{4}}`, into the final Pauli product measurements (PPMs).
    """
    raise NotImplementedError(
        "The merge_ppr_ppm compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="ppr-to-ppm")
def ppr_to_ppm(tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False):
    r"""
    A quantum compilation pass that decomposes Pauli product rotations (PPRs),
    :math:`P(\theta) = \exp(-iP\theta)`, into Pauli product measurements (PPMs).
    """
    raise NotImplementedError(
        "The ppr_to_ppm compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="ppm-compilation")
def ppm_compilation(
    tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    r"""
    A quantum compilation pass that transforms Clifford+T gates into Pauli product measurements
    (PPMs).
    """
    raise NotImplementedError(
        "The ppm_compilation compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="reduce-t-depth")
def reduce_t_depth(qnode):
    r"""
    A quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
    rotation (PPR, :math:`P(\theta) = \exp(-iP\theta)`) operators (e.g., ``T`` gates) by commuting
    PPRs in adjacent layers and merging compatible ones (a layer comprises PPRs that mutually
    commute). For more details, see Figure 6 of
    `A Game of Surface Codes <https://arXiv:1808.02892v3>`_.
    """
    raise NotImplementedError(
        "The reduce_t_depth compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )
