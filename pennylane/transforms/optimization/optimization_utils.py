# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for circuit optimization."""
# pylint: disable=too-many-return-statements,import-outside-toplevel
from pennylane.math import arccos, arctan2, asarray, cos, moveaxis, sin, sqrt, stack
from pennylane.ops.identity import GlobalPhase
from pennylane.wires import Wires


def find_next_gate(wires, op_list):
    """Given a list of operations, finds the next operation that acts on at least one of
    the same set of wires, if present.

    Args:
        wires (Wires): A set of wires acted on by a quantum operation.
        op_list (list[Operation]): A list of operations that are implemented after the
            operation that acts on ``wires``.

    Returns:
        int or None: The index, in ``op_list``, of the earliest gate that uses one or more
        of the same wires, or ``None`` if no such gate is present.
    """
    next_gate_idx = None

    for op_idx, op in enumerate(op_list):
        if len(Wires.shared_wires([wires, op.wires])) > 0:
            next_gate_idx = op_idx
            break

    return next_gate_idx


def fuse_rot_angles(angles1, angles2):
    r"""Compute the set of rotation angles that is equivalent to performing
    two successive ``qml.Rot`` operations.

    The ``qml.Rot`` operation represents the most general single-qubit operation.
    Two such operations can be fused into a new operation, however the angular dependence
    is non-trivial.

    Args:
        angles_1 (tensor_like): A set of three angles for the first ``qml.Rot`` operation.
        angles_2 (tensor_like): A set of three angles for the second ``qml.Rot`` operation.

    Returns:
        tensor_like: Rotation angles for a single ``qml.Rot`` operation that
        implements the same operation as the two sets of input angles.

    This function supports broadcasting/batching as long as the two inputs are standard
    broadcast-compatible.

    .. note::

        The output angles are not always defined uniquely because Euler angles are not
        defined uniquely for some rotations. ``fuse_rot_angles`` makes a particular
        choice in this case.

    .. warning::

        This function is not differentiable everywhere. It has singularities for specific
        input values where the derivative will be NaN.

    .. warning::

        This function is numerically unstable at singular points. It is recommended to use
        it with 64-bit floating point precision.

    .. details::
        :title: Derivation
        :href: derivation

        The matrices for the two individual rotations are given by

        .. math::

            R(\phi_j,\theta_j,\omega_j)
            &= \begin{bmatrix}
            e^{-i(\phi_j+\omega_j)/2}\cos(\theta_j/2) & -e^{i(\phi_j-\omega_j)/2}\sin(\theta_j/2)\\
            e^{-i(\phi_j-\omega_j)/2}\sin(\theta_j/2) & e^{i(\phi_j+\omega_j)/2}\cos(\theta_j/2)
            \end{bmatrix}\\
            &= \begin{bmatrix}
            e^{-i\alpha_j}c_j & -e^{i\beta_j}s_j \\
            e^{-i\beta_j}s_j & e^{i\alpha_j}c_j
            \end{bmatrix},

        where we introduced abbreviations :math:`\alpha_j`, :math:`\beta_j`,
        :math:`c_j=\cos(\theta_j / 2)` and :math:`s_j=\sin(\theta_j / 2)` for notational brevity.
        The upper left entry of the matrix product
        :math:`R(\phi_2,\theta_2,\omega_2)R(\phi_1,\theta_1,\omega_1)` reads

        .. math::

            x = e^{-i(\alpha_2+\alpha_1)} c_2 c_1 - e^{i(\beta_2-\beta_1)} s_2 s_1

        and should equal :math:`e^{-i(\alpha_f)/2}c_f` for the fused rotation angles.
        This means that we can obtain :math:`\theta_f` from the magnitude of the matrix product
        entry above, choosing :math:`c_f=\cos(\theta_f / 2)` to be non-negative:

        .. math::

            c_f = |x| &=
            \left|
            e^{-i(\alpha_2+\alpha_1)} c_2 c_1
            -e^{i(\beta_2-\beta_1)} s_2 s_1
            \right| \\
            &= \sqrt{c_1^2 c_2^2 + s_1^2 s_2^2 - 2 c_1 c_2 s_1 s_2 \cos(\omega_1 + \phi_2)}.

        Now we again make a choice and pick :math:`\theta_f` to be non-negative:

        .. math::

            \theta_f = 2\arccos(|x|).

        We can extract the angle combination :math:`\alpha_f` from :math:`x` above as well via
        :math:`\operatorname{arg}(x)`, which can be readily computed with :math:`arctan`:

        .. math::

            \alpha_f = -\arctan\left(
            \frac{-c_1c_2\sin(\alpha_1+\alpha_2)-s_1s_2\sin(\beta_2-\beta_1)}
            {c_1c_2\cos(\alpha_1+\alpha_2)-s_1s_2\cos(\beta_2-\beta_1)}
            \right).

        We can use the standard numerical function :math:`\operatorname{arctan2}`, which
        computes :math:`\arctan(x_1/x_2)` from :math:`x_1` and :math:`x_2` while handling
        special points suitably to obtain the argument of the underlying complex number
        :math:`x_2 + x_1 i`.

        Finally, to obtain :math:`\beta_f`, we need a second element of the matrix product from
        above. We compute the lower-left entry to be

        .. math::

            y = e^{-i(\beta_2+\alpha_1)} s_2 c_1 + e^{i(\alpha_2-\beta_1)} c_2 s_1,

        which should equal :math:`e^{-i \beta_f}s_f`. From this, we can compute

        .. math::

            \beta_f = -\arctan\left(
            \frac{-c_1s_2\sin(\alpha_1+\beta_2)+s_1c_2\sin(\alpha_2-\beta_1)}
            {c_1s_2\cos(\alpha_1+\beta_2)+s_1c_2\cos(\alpha_2-\beta_1)}
            \right).

        From this, we may extract

        .. math::

            \phi_f = \alpha_f + \beta_f
            \omega_f = \alpha_f - \beta_f

        and are done.

        **Mathematical properties:**

        All functions above are well-defined on the domain we are using them on,
        if we handle :math:`\arctan` via standard numerical implementations such as
        ``np.arctan2``.
        Based on the choices we made in the derivation above, the fused angles will lie in
        the intervals

        .. math::

            \phi_f, \omega_f \in [-\pi, \pi],\quad \theta_f \in [0, \pi].

        Close to the boundaries of these intervals, ``fuse_rot_angles`` exhibits
        discontinuities, depending on the combination of input angles.
        These discontinuities also lead to singular (non-differentiable) points as discussed below.

        **Differentiability:**

        The function derived above is differentiable almost everywhere.
        In particular, there are two problematic scenarios at which the derivative is not defined:
        First, the square root is not differentiable at :math:`0`, making all input angles with
        :math:`|x|=0` singular. Second, :math:`\arccos` is not differentiable at :math:`1`, making
        all input angles with :math:`|x|=1` singular.
    """

    # moveaxis required for batched inputs
    phi1, theta1, omega1 = moveaxis(asarray(angles1), -1, 0)
    phi2, theta2, omega2 = moveaxis(asarray(angles2), -1, 0)
    c1, c2, s1, s2 = cos(theta1 / 2), cos(theta2 / 2), sin(theta1 / 2), sin(theta2 / 2)

    mag = sqrt(c1**2 * c2**2 + s1**2 * s2**2 - 2 * c1 * c2 * s1 * s2 * cos(omega1 + phi2))
    theta_f = 2 * arccos(mag)

    alpha1, beta1 = (phi1 + omega1) / 2, (phi1 - omega1) / 2
    alpha2, beta2 = (phi2 + omega2) / 2, (phi2 - omega2) / 2

    alpha_arg1 = -c1 * c2 * sin(alpha1 + alpha2) - s1 * s2 * sin(beta2 - beta1)
    alpha_arg2 = c1 * c2 * cos(alpha1 + alpha2) - s1 * s2 * cos(beta2 - beta1)
    alpha_f = -1 * arctan2(alpha_arg1, alpha_arg2)

    beta_arg1 = -c1 * s2 * sin(alpha1 + beta2) + s1 * c2 * sin(alpha2 - beta1)
    beta_arg2 = c1 * s2 * cos(alpha1 + beta2) + s1 * c2 * cos(alpha2 - beta1)
    beta_f = -1 * arctan2(beta_arg1, beta_arg2)

    return stack([alpha_f + beta_f, theta_f, alpha_f - beta_f], axis=-1)


def _fuse_global_phases(operations):
    """Fuse all the global phase operations into single one.

    Args:
        operations (list[Operation]): list of operations to be iterated over

    Returns:
        transformed list with a single :func:`~.pennylane.GlobalPhase` operation.
    """

    fused_ops, global_ops = [], []
    for op in operations:
        if isinstance(op, GlobalPhase):
            global_ops.append(op)
        else:
            fused_ops.append(op)

    fused_ops.append(GlobalPhase(sum(op.data[0] for op in global_ops)))
    return fused_ops
