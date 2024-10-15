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
"""Cartan involutions"""
# pylint: disable=missing-function-docstring

import numpy as np

# see https://arxiv.org/pdf/2406.04418 appendix C

# matrices


def J(n):
    return np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])


def Ipq(p, q):
    IIm = np.block([[-np.eye(p), np.zeros((p, p))], [np.zeros((q, q)), np.eye(q)]])
    return IIm


def Kpq(p, q):
    KKm = np.block(
        [
            [-np.eye(p), np.zeros((p, p)), np.zeros((p, p)), np.zeros((p, p))],
            [np.zeros((q, q)), np.eye(q), np.zeros((q, q)), np.zeros((q, q))],
            [np.zeros((p, p)), np.zeros((p, p)), -np.eye(p), np.zeros((p, p))],
            [np.zeros((q, q)), np.zeros((q, q)), np.zeros((q, q)), np.eye(q)],
        ]
    )
    return KKm


# involution


def AI(op):
    return np.allclose(op, op.conj())


def AII(op):
    JJ = J(op.shape[-1] // 2)
    return np.allclose(op, JJ @ op.conj() @ JJ.T)


def AIII(op, p=None, q=None):
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(AIII, p=p, q=q)"
        )
    IIm = Ipq(p, q)
    return np.allclose(op, IIm @ op @ IIm)


def BDI(op, p=None, q=None):
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(BDI, p=p, q=q)"
        )
    IIm = Ipq(p, q)
    return np.allclose(op, IIm @ op @ IIm)


def CI(op):
    return np.allclose(op, op.conj())


def CII(op, p=None, q=None):
    if p is None or q is None:
        raise ValueError(
            "please specify p and q for the involution via functools.partial(CII, p=p, q=q)"
        )
    KKm = Kpq(p, q)
    return np.allclose(op, KKm @ op @ KKm)


def DIII(op):
    JJ = J(op.shape[-1] // 2)
    return np.allclose(op, JJ @ op @ JJ.T)
