import shutil
import subprocess
import pytest
import pennylane as qml


def cmd_exists(cmd):
    """Returns True if a binary exists on
    the system path"""
    return shutil.which(cmd) is not None


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return {"rtol": 0, "atol": 1e-8}


@pytest.fixture(scope="module")
def psi4_support():
    """Boolean fixture for Psi4 support"""
    if not cmd_exists("psi4"):
        return False

    res = subprocess.call(["psi4", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res == 1:
        return False

    try:
        import psi4

        support = True
    except ImportError as e:
        support = False

    return support


@pytest.fixture(scope="module")
def babel_support():
    """Boolean fixture for Babel support"""
    if cmd_exists("obabel"):
        return True

    return False


@pytest.fixture()
def requires_babel(babel_support):
    if not babel_support:
        pytest.skip("Skipped, no Babel support")


@pytest.fixture(
    scope="module",
    params=[
        None,
        qml.wires.Wires(
            list("ab") + [-3, 42] + ["xyz", "23", "wireX"] + ["w{}".format(i) for i in range(20)]
        ),
        list(range(100, 120)),
        {13 - i: "abcdefghijklmn"[i] for i in range(14)},
    ],
)
def custom_wires(request):
    """Custom wire mapping for Pennylane<->OpenFermion conversion"""
    return request.param
