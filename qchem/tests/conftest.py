import shutil
import subprocess
import pytest


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
