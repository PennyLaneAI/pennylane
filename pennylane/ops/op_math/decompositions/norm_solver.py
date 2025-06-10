# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains mathematical operations for solving diophantine equations."""

import math
from functools import lru_cache, singledispatch
from random import randrange

from pennylane.ops.op_math.decompositions.rings import ZOmega, ZSqrtTwo


@singledispatch
def _gcd(a, b):
    """Compute the greatest common divisor of two integers."""
    raise NotImplementedError(
        f"GCD is not implemented for types {type(a)} and {type(b)}. "
        "Please implement a custom GCD function for these types."
    )


@_gcd.register(int)
def _(elem1, elem2):
    return math.gcd(elem1, elem2)


@_gcd.register(ZSqrtTwo)
@_gcd.register(ZOmega)
def _(elem1, elem2):
    while elem2 != 0:
        elem1, elem2 = elem2, elem2 % elem1
    return elem1, elem2


def _prime_factorize(n: int, max_trials=1000) -> list[int]:
    r"""Computes the prime factorization of a number :math:`n`.

    Args:
        n (int): The number to factor.
        max_tries (int): The maximum number of attempts to find a factor.

    Returns:
        list[int]: A list of prime factors of :math:`n`.
    """
    factors, stack = [], [n]
    while len(stack) > 0:
        p = stack.pop()
        # Trivial case
        if p == 1:
            continue
        # Cannot split in Z[√2], if p ≡ 7 mod 8.
        if _primality_test(p) and p % 8 != 7:
            factors.append(p)
            continue
        if factor := _integer_factorize(p, max_trials) == 7:
            continue
        # If we have found an integer factor,
        # push it and its complement onto the stack
        if factor is not None:
            stack.extend([factor, p // factor])

    return sorted(factors)


@lru_cache(maxsize=100)
def _integer_factorize(n: int, max_tries=1000) -> int | None:
    r"""Computes an integer factor of a number :math:`n`.

    This function implements the `Brent's variant <https://doi.org/10.1007/BF01933190>`_
    of the Pollard's rho algorithm for integer factorization.

    Args:
        n (int): The number to factor.
        max_tries (int): The maximum number of attempts to find a factor.

    Returns:
        int | None: An integer factor of :math:`n`, or ``None`` if no factors are found.
    """
    if n % 2 == 0:
        return 2

    # Main loop: retry with different parameters on failure
    while max_tries > 0:
        x, c, m = randrange(1, n), randrange(1, n), randrange(1, n)
        g, r, q = 1, 1, 1

        while g == 1:
            y = x
            # Process mext `r` steps
            for _ in range(r):
                x = (x * x % n + c) % n

            k = 0
            while k < r and g == 1:
                xs, rs = x, min(m, r - k)
                # Process next `min(m, r-k)`` steps
                for _ in range(rs):
                    x = (x * x % n + c) % n
                    q = (q * abs(y - x)) % n
                g = _gcd(q, n)
                k += m
            r <<= 1

        # Linear search for a factor if the above doesn't yield a result.
        if g == n:
            g, y = 1, xs
            while g == 1:
                y = (y * y % n + c) % n
                g = _gcd(abs(x - y), n)

        # If we found a valid integer factor, such that it is neither 1 nor n.
        if g not in (1, n):
            return g

        max_tries -= 1

    return None


def factorize_prime_zsqrt_two(p: int) -> list[ZSqrtTwo]:
    r"""Find the factorization of a prime number :math:`p` in ring :math:`\mathbb{Z}[\sqrt{2}]`.

    This uses congruence of :math:`p \text{mod} 8` and properties of the ring
    :math:`\mathbb{Z}[\sqrt{2}]` to determine how a prime integer can be expressed
    as a product of elements in this ring.

    Args:
        n (int): The prime number for which to find the factorization.
    Returns:
        list[ZSqrtTwo]: A list of factors in the ring :math:`\mathbb{Z}[\sqrt{2}]`,
            or `None` if no factorization exists.
    """
    if p == 2:
        # 2 = (0 + 1√2)(0 - 1√2)
        return [ZSqrtTwo(0, 1), ZSqrtTwo(0, 1)]

    if (r := p % 8) in (3, 5):
        # p = (p + 0√2) is prime in Z[√2]
        return [ZSqrtTwo(p, 0)]

    if r in (1, 7):  # p ≡ ±1 mod 8
        # Solve t^2 ≡ 2 (mod p) to split p
        if (t := _sqrt_modulo_p(2, p)) is None:
            return None
        # Perform ring GCD to get (a + b√2)(a - b√2)
        res = _gcd(ZSqrtTwo(p, 0), ZSqrtTwo(min(t, p - t), 1))
        return [res, res.adj2()]

    return None


# pylint: disable=too-many-return-statements
def factorize_prime_zomega(x: ZSqrtTwo, p: int) -> ZOmega | None:
    r"""Find a prime factor of an element :math:`x` in the ring :math:`\mathbb{Z}[\omega]`,
    where :math:`x` divides a prime integer :math:`p`.

    Args:
        x (ZSqrtTwo): The element in the ring :math:`\mathbb{Z}[\sqrt{2}]`, such that x | p.
        p (int): The prime number for which to find the factorization.

    Returns:
        list[ZOmega]: A list of factors in the ring :math:`\mathbb{Z}[\omega]`,
            or `None` if no factorization exists.
    """
    # Basic cases
    if x % p != 0:
        return None
    if p == 2:
        return ZOmega(0, 0, 1, 1)

    # p = 2k or p ≡ 7 mod 8, no factorization in Z[ω]
    if ((a := p % 8) % 2 == 0) or a == 7:
        return None

    # p = (a + bω)(a - bω) with b = sqrt(-1) mod p
    if a in (1, 5):
        if h := _sqrt_modulo_p(-1, p) is None:
            return None
        return _gcd(ZOmega(0, 1, 0, h), ZOmega(-x.b, 0, x.b, x.a))

    # p = (a + bω)(a - bω) with b = sqrt(-2) mod p
    if a == 3:
        if h := _sqrt_modulo_p(-2, p) is None:
            return None
        return _gcd(ZOmega(1, 0, 1, h), ZOmega(-x.b, 0, x.b, x.a))

    return None


@lru_cache(maxsize=100)
def _primality_test(n: int) -> bool:
    r"""Determines whether an integer is prime or not.

    This function implements the `Miller-Rabin primality test
    <https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test>`_.

    Args:
        n (int): The number to test for primality.

    Returns:
        bool: True if :math:`n` is likely prime, False otherwise.
    """
    if n <= 1 or n == 4:
        return False
    if n <= 3:
        return True

    # Small primes quick check
    ps = {5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    if n in ps:
        return True

    if any(n % p == 0 for p in ps):
        return False

    # Factor n−1 as d·2^s with d odd
    d, s = n - 1, 0
    while d & 1 == 0:
        d >>= 1
        s += 1

    # Deterministic bases for testing 64-bit range [https://miller-rabin.appspot.com/]
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for base in bases:
        if base % n == 0:
            continue
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                continue
        return False

    return True


@lru_cache(maxsize=100)
def _legendre_symbol(a: int, p: int, k: int = 2) -> int:
    r"""Computes the Legendre symbol :math:`\left(\frac{a}{p}\right)`."""
    return pow(a, (p - 1) // k, p)


def _sqrt_modulo_p(n: int, p: int) -> int | None:
    r"""Computes square root of :math:`n` under modulo :math:`p` if it exists.

    This uses `Tonelli-Shanks algorithm <https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm>`_
    to compute :math:`x`, such that :math:`x^2 \equiv n (mod p)`.

    Args:
        n (int): The number to compute the square root of.
        p (int): The prime modulus.

    Returns:
        int or None: The square root of :math:`n` under modulo :math:`p`, or None if it does not exist.
    """
    # Trivial cases
    if (a := n % p) == 0:
        return 0
    if p == 2:
        return a
    if _legendre_symbol(a, p) != 1:
        return None

    # Factor p−1 as q·2^s with q odd
    q, s = p - 1, 0
    while q & 1 == 0:
        q >>= 1
        s += 1

    # (p - 1) = 2.q ==> 2 * k
    if s == 1:
        return pow(a, (p + 1) // 4, p)

    # Find a quadratic non‐residue z,
    # such that z^((p-1)/2) ≡ −1 mod p
    for z in range(2, p):
        if _legendre_symbol(z, p) == (p - 1):
            break

    r = pow(a, (q + 1) // 2, p)
    c = pow(z, q, p)
    t = pow(a, q, p)

    m = s
    while t % p != 1:
        # Find least i in [1, m),
        # such that t^(2^i) ≡ 1 mod p
        ix, t2 = 1, pow(t, 2, p)
        while t2 % p != 1 and ix < m:
            t2, ix = pow(t2, 2, p), ix + 1

        # Compute b = c^(2^(m-ix-1)) mod p
        # and update the intial elements
        b = pow(c, 1 << (m - ix - 1), p)
        r = (r * b) % p
        c = pow(b, 2, p)
        t = (t * c) % p
        m = ix

    return r


def solve_diophantine(xi: ZSqrtTwo, max_trials: int = 1000) -> ZOmega | None:
    r"""Solve the Diophantine equation :math:`t^* t = \xi` for :math:`t \in \mathbb{Z}[\omega]` and :math:`\xi \in \mathbb{Z}[\sqrt{2}]`.

    Args:
        xi (ZSqrtTwo): An element of the ring :math:`\mathbb{Z}[\sqrt{2}]`.

    Returns:
        ZOmega | None: An element of the ring :math:`\mathbb{Z}[\omega]` that satisfies the equation, or ``None`` if no solution exists.
    """
    if p := abs(xi) < 2:
        return None

    if not (factors := _prime_factorize(p, max_trials)):
        return None

    scale, next_xi = ZOmega(d=1), xi
    for factor in factors:
        if primes_zsqrt_two := factorize_prime_zsqrt_two(factor) is None:
            return None

        for eta in primes_zsqrt_two:
            # Scale the next_xi by the factor in Z[√2].
            next_xi, next_ab = next_xi * eta.adj2(), abs(eta)
            # Check if the next_xi is divisible by the element eta in Z[√2].
            if next_xi.a % next_ab == 0 and next_xi.b % next_ab == 0:
                next_xi = ZSqrtTwo(next_xi.a // next_ab, next_xi.b // next_ab)
                if t := factorize_prime_zomega(eta, factor) is None:
                    return None
                scale *= t

    t2 = xi / (scale.conj() * scale).to_zsqrt()
    if abs(t2) ** 2 != 1:
        return None

    return scale * t2.sqrt().to_zomega()
