# pylint:disable=missing-module-docstring
from .rings import ZRootTwo, Matrix, ZOmega, DOmega
from .newsynth import gridsynth
from .conversion import denomexp

ROOT_TWO = ZRootTwo(0, 1)
ZERO = ZOmega(0, 0, 0, 0)
H_OMEGA = Matrix.array([[1, 1], [1, -1]]) / ROOT_TWO
TINV_OMEGA = Matrix.array([[ZOmega(0, 0, 0, 1), ZERO], [ZERO, ZOmega(-1, 0, 0, 0)]])


def rz_to_clifford_plus_t(theta, epsilon):
    """
    Convert an RZ rotation by theta to a sequence of Clifford+T gates.

    Args:
        theta (float): the RZ rotation angle to approximate
        epsilon (float): the error bound

    Returns:
        List[str]: The list of operators that approximate the RZ rotation.
    """
    mat, _, _ = gridsynth(epsilon, theta)
    res = ""
    z: DOmega = mat[0][0]
    sde = denomexp(z.abs_squared())
    print("initial sde:", sde)
    while sde > 2:
        mat_t = mat
        mat_ = H_OMEGA @ mat_t
        for k in range(4):
            z_: DOmega = mat_[0][0]
            if (curr_sde := denomexp(z_.abs_squared())) == sde - 1:
                res += "T" * k + "H"
                sde = curr_sde
                mat = mat_
                break
            mat_t = TINV_OMEGA @ mat_t
            mat_ = H_OMEGA @ mat_t
    return res, H_OMEGA @ mat_t
