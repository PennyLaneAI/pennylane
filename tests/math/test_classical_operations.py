import pytest
import numpy as np
import pennylane as qml


class Test_gcd:
    def test_gcd_result(self, a, b):
        expected = np.gcd(a, b)
        res = qml.math.gcd(a, b)
        assert res == expected

class Test_modular_multiplicative_inverse:
    def test_co_primality(self):
        with pytest.raises(
            Exception, match="a and N should be co-prime, i.e. gcd(a,N) should be 1"
        ):
            qml.math.modular_multiplicative_inverse(3,3)