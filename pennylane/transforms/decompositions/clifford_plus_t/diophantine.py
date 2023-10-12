# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Solve the Diophantine equation :math:`t^{\dagger}t = \xi`"""
# pylint:disable=missing-function-docstring,unused-argument

from typing import Union, List, Tuple
import numpy as np

from .conversion import denomexp
from .rings import ZRootTwo, DRootTwo, ZOmega, DOmega, Dyadic

ZRing = Union[ZRootTwo, ZOmega]
Factorable = Union[int, ZRootTwo]
zero = ZRootTwo(0, 0)
EFFORT = 1000


class DiophantineError(Exception):
    """A generic error meaning we failed to solve the Diophantine equation."""


def diophantine_dyadic(xi: DRootTwo, effort: int) -> DOmega:
    r"""Given a :math:`\xi` value, solve the Diophantine equation or fail."""
    k = denomexp(xi)
    k_, k__ = divmod(k, 2)
    xi_ = ZRootTwo(2, 1) ** k__ * 2**k_ * xi
    if not isinstance(xi_, ZRootTwo):
        raise TypeError(f"expected xi_ to be of type ZRootTwo, got {type(xi_).__name__}:{xi_}")

    t = diophantine(xi_)
    u = ZOmega(-1, 1, -1, 1) * DOmega(0, 0, 0, Dyadic(1, 1))
    root_half = DOmega.from_root_two(DRootTwo(0, Dyadic(1, 1)))
    return u**k__ * root_half**k_ * t


def diophantine(xi: ZRootTwo) -> ZOmega:
    """Solve the Diophantine equation."""
    if xi == zero:
        return ZOmega(0, 0, 0, 0)
    if xi < 0 or xi.adj2() < 0:
        raise DiophantineError

    t = diophantine_associate(xi)
    xi_associate = (t.conjugate() * t).to_root_two()
    u = euclid_div(xi, xi_associate)
    v = u.sqrt()
    if v is None:
        raise DiophantineError
    return ZOmega.from_root_two(v) * t


def diophantine_associate(xi: ZRootTwo) -> ZOmega:
    if xi == zero:
        return ZOmega(0, 0, 0, 0)

    d = euclid_gcd(xi, xi.adj2())
    t1 = dioph_zroottwo_selfassociate(d)

    xi_ = euclid_div(xi, d)
    t2 = dioph_zroottwo_assoc(xi_)
    return t1 * t2


def dioph_zroottwo_selfassociate(xi: ZRootTwo) -> ZOmega:
    """Diophantine."""
    if xi == zero:
        return ZOmega(0, 0, 0, 0)

    n = int(np.gcd(xi.a, xi.b))
    res = dioph_int_assoc(n)
    r = euclid_div(xi, ZRootTwo(n, 0))
    if euclid_divides(ZRootTwo(0, 1), r):
        res = ZOmega(0, 0, 1, 1) * res
    return res


def dioph_zroottwo_assoc(xi: ZRootTwo) -> ZOmega:
    """Diophantine."""
    if xi == zero:
        return ZOmega(0, 0, 0, 0)
    res = dioph_zroottwo_assoc_prime(xi)
    if res is not None:
        return res

    n = abs(xi.norm())
    a = find_factor(n)
    alpha = euclid_gcd(xi, ZRootTwo(a, 0))
    beta = euclid_div(xi, alpha)
    _, facs = relatively_prime_factors(alpha, beta)
    return dioph_zroottwo_assoc_powers(facs)


def dioph_zroottwo_assoc_prime(xi: ZRootTwo) -> ZOmega:
    """Diophantine."""
    if xi == zero:
        return ZOmega(0, 0, 0, 0)
    n = abs(xi.norm())
    n_mod_8 = n % 8
    if n_mod_8 == 1:
        h = root_of_negative_one(n)
        xi_omega = ZOmega.from_root_two(xi)
        t = euclid_gcd(ZOmega(0, 1, 0, h), xi_omega)
        if not euclid_associates(t.conjugate() * t, xi_omega):
            raise ValueError(f"t†t should equal {xi_omega}, got {t=} and t†t={t.conjugate() * t}")
        return t
    if n_mod_8 == 7:
        return None
    raise DiophantineError


def dioph_zroottwo_assoc_powers(facs: List[Tuple[ZRootTwo, int]]) -> ZOmega:
    """Diophantine."""
    vals = []
    for xi, k in facs:
        if k % 2 == 0:
            vals.append(ZOmega.from_root_two(xi ** (k // 2)))
            continue
        t = dioph_zroottwo_assoc(xi)
        if t is None:
            raise DiophantineError
        vals.append(t**k)
    return np.prod(vals)


def dioph_int_assoc(n: int) -> ZOmega:
    """Diophantine."""
    if n < 0:
        return dioph_int_assoc(-n)
    if n in {0, 1}:
        return ZOmega(0, 0, 0, n)

    res = dioph_int_assoc_prime(n)
    if res is not None:
        return res

    a = find_factor(n)
    b = n // a
    _, facs = relatively_prime_factors(a, b)
    return dioph_int_assoc_powers(facs)


def dioph_int_assoc_prime(n: int) -> ZOmega:
    """recursive prime factorization solver."""
    # pylint:disable=too-many-return-statements
    if n < 0:
        return dioph_int_assoc_prime(-n)
    if n == 0:
        return ZOmega(0, 0, 0, 0)
    if n == 2:
        return ZOmega.from_root_two(ZRootTwo(0, 1))
    n_omega = ZOmega(0, 0, 0, n)
    if n % 4 == 1:
        h = root_of_negative_one(n)
        t = euclid_gcd(ZOmega(0, 1, 0, h), n_omega)
        if t.conjugate() * t != n_omega:
            raise ValueError(f"t†t should equal {n}, got {t=} and t†t={t.conjugate() * t}")
        return t
    n_mod_8 = n % 8
    if n_mod_8 == 3:
        h = root_mod(n, -2)
        t = euclid_gcd(ZOmega(1, 0, 1, h), n_omega)
        if t.conjugate() * t != n_omega:
            raise ValueError(f"t†t should equal {n}, got {t=} and t†t={t.conjugate() * t}")
        return t
    if n_mod_8 == 7:
        root_mod(n, 2)  # if this fails, we should stop trying for this candidate
        return None

    # in the Haskell, this is an error but dioph_int_assoc_prime just diverges. Not sure why.
    raise ValueError("unexpected error: dioph_int_assoc_prime with n:", n)


def find_factor(n: int) -> int:
    """iterative factorization solver."""
    if n % 2 == 0 and n > 2:
        return 2

    for _ in range(EFFORT):
        a = np.random.randint(1, n)
        f = lambda x: (x**2 + a) % n  # pylint:disable=cell-var-from-loop
        x = 2
        y = f(2)
        for _ in range(EFFORT):
            d = np.gcd(x - y, n)
            if d == 1:
                x = f(x)
                y = f(f(y))
            elif d == n:  # try again with another random number
                break
            else:
                return int(d)

    raise DiophantineError


def relatively_prime_factors(
    a: Factorable, b: Factorable
) -> Tuple[Factorable, List[Tuple[Factorable, int]]]:
    """Find the relatively prime factors of a and b."""

    def aux2(h, fs):
        if not fs:
            return 1, [], [(h, 1)]

        (f, k) = fs.pop(0)
        if euclid_associates(h, f):
            u_ = euclid_div(h, f)
            return u_, [], [(f, k + 1)] + fs

        d = euclid_gcd(h, f)
        if is_unit(d):
            u, hs, fs_ = aux2(h, fs)
            return u, hs, [(f, k)] + fs_

        return 1, [euclid_div(h, d), d] + [euclid_div(f, d)] * k + [d] * k, fs

    def aux(u, hs, fs):
        if not hs:
            return (u, fs)
        h, t = (hs[0], hs[1:])
        if is_unit(h):
            return aux(h * u, t, fs)
        u_, hs_, fs_ = aux2(h, fs)
        return aux(u_ * u, hs_ + t, fs_)

    return aux(1, [a, b], [])


def dioph_int_assoc_powers(facs: List[Tuple[int, int]]) -> ZOmega:
    """Diophantine."""
    vals = []
    for n, k in facs:
        if k % 2 == 0:
            vals.append(ZOmega(0, 0, 0, n ** (k // 2)))
            continue
        t = dioph_int_assoc(n)
        if t is None:
            raise DiophantineError
        vals.append(t**k)
    return np.prod(vals)


def root_of_negative_one(n: int):
    """Get the root of negative one in Z[n]."""
    if n == 1:
        return 1
    for _ in range(EFFORT):
        b = np.random.randint(1, n)
        h = power_mod(b, (n - 1) // 4, n)
        r = (h * h) % n
        if r == n - 1:
            return h
        if r != 1:
            raise DiophantineError

    raise DiophantineError


def power_mod(a: int, k: int, n: int) -> int:
    """Modular exponentiation using repeated squaring."""
    if k == 0:
        return 1
    if k == 1:
        return a % n

    b = power_mod(a, k // 2, n)
    if k % 2 == 0:
        return (b * b) % n
    return (b * b * a) % n


def root_mod(n: int, a: int) -> int:
    """Get the root mod."""
    if a % n == -1:
        return root_of_negative_one(n)

    def _mul(ab, cd):
        a, b = ab
        c, d = cd
        x, y, z = a * c, a * d + b * c, b * d
        a_, b_ = y - x * r, z - x * s
        return a_ % n, b_ % n

    def _pow(x, m):
        if m <= 0:
            return 0, 1
        if m % 2 == 1:
            return _mul(x, _pow(x, m - 1))
        y = _pow(x, m // 2)
        return _mul(y, y)

    for _ in range(EFFORT):
        b = np.random.randint(n)
        r = (2 * b) % n
        s = (b**2 - a) % n
        c, d = _pow((1, 0), (n - 1) // 2)
        c_ = inv_mod(n, c)
        if c_ is not None:
            t = (1 - d) * c_
            t1 = (t + b) % n
            if (t1**2 - a) % n == 0:
                return t1

    raise DiophantineError


### euclidean stuff ###


def euclid_gcd(x: ZRing, y: ZRing) -> ZRing:
    """Compute GCD using Euclid's method."""
    if y == 0:
        return x
    _, r = _divmod(x, y)
    return euclid_gcd(y, r)


def euclid_div(x, y) -> ZRootTwo:
    """Division using Euclid's method."""
    q, _ = _divmod(x, y)
    return q


def euclid_divides(x, y) -> bool:
    """Check if y Euclid-divides x."""
    if x == 0:
        return y == 0
    _, euclid_mod = _divmod(x=y, y=x)
    return euclid_mod == 0


def euclid_associates(x, y) -> bool:
    """Check if x and y associate."""
    return euclid_divides(x, y) and euclid_divides(x=y, y=x)


def euclid_inverse(x):
    if x == 0:
        return None
    q, r = _divmod(1, x)
    return q if r == 0 else None


def extended_euclid(x, y):
    if y == 0:
        return 1, 0, 0, 1, x
    q, r = _divmod(x, y)
    a_, b_, s_, t_, d = extended_euclid(y, r)
    return b_, a_ - b_ * q, -t_, t_ * q - s_, d


def inv_mod(p, a):
    """Inverse of a in R_p."""
    b, _, _, _, d = extended_euclid(a, p)
    d_ = euclid_inverse(d)
    if d_ is None:
        return None
    _, r = _divmod(b * d_, p)
    return r


def _divmod(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return divmod(x, y)

    if isinstance(x, (int, ZRootTwo)) and isinstance(y, ZRootTwo):
        x_yadj = x * y.adj2()
        k = y.norm()
        q1 = rounddiv(x_yadj.a, k)
        q2 = rounddiv(x_yadj.b, k)
        q = ZRootTwo(q1, q2)
        return q, x - y * q

    if isinstance(x, ZOmega) and isinstance(y, ZOmega):
        prod = x * y.conjugate() * (y * y.conjugate()).adj2()
        k = y.norm()
        if k == 0:
            raise ValueError("bad norm of ", y)
        a = rounddiv(prod.a, k)
        b = rounddiv(prod.b, k)
        c = rounddiv(prod.c, k)
        d = rounddiv(prod.d, k)
        q = ZOmega(a, b, c, d)
        return q, x - y * q

    raise TypeError("unexpected:", x, y, type(x), type(y))


def rounddiv(x, y):
    return (x + int(y / 2)) // y


def is_unit(x):
    """Check if x is a unit of a Euclidean domain."""
    return euclid_inverse(x) is not None
