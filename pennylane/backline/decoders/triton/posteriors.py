import triton
import triton.language as tl

from .math_utils import (
    _bp_c2v_msg,
    _bp_tanh_half,
    _get_syndrome_signs,
    _llr_from_p,
    _sign1,
)


# Adapted from Pennylane Blog: https://pennylane.ai/demos/tutorial_bp_catalyst
@triton.jit
def _sum_product_posteriors(
    syndrome,
    H: tl.constexpr,
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
):
    """Compute posterior LLRs with sum-product belief propagation."""
    L0: tl.constexpr = _llr_from_p(prob)
    NCHECKS: tl.constexpr = len(H)
    NVARS: tl.constexpr = len(H[0])

    s = _get_syndrome_signs(syndrome, NCHECKS)

    E = ()
    for _ in tl.static_range(NCHECKS):
        row = ()
        for _ in tl.static_range(NVARS):
            row += (0.0,)
        E += (row,)

    for _ in range(NITER):
        T = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    msg = L0
                    for c2 in tl.static_range(NCHECKS):
                        if c2 != c and H[c2][v]:
                            msg += E[c2][v]
                    row += (_bp_tanh_half(msg),)
                else:
                    row += (0.0,)
            T += (row,)

        newE = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    prod = 1.0
                    for v2 in tl.static_range(NVARS):
                        if v2 != v and H[c][v2]:
                            prod *= T[c][v2]
                    row += (_bp_c2v_msg(s[c], prod),)
                else:
                    row += (0.0,)
            newE += (row,)
        E = newE

    P = ()
    for v in tl.static_range(NVARS):
        post = L0
        for c in tl.static_range(NCHECKS):
            if H[c][v]:
                post += E[c][v]
        P += (post,)
    return P


@triton.jit
def _norm_min_sum_posteriors(
    syndrome,
    H: tl.constexpr,
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
    ALPHA: tl.constexpr = 0.75,
    BIG: tl.constexpr = 1.0e9,
):
    """Compute posterior LLRs with normalized min-sum decoding."""
    L0: tl.constexpr = _llr_from_p(prob)
    NCHECKS: tl.constexpr = len(H)
    NVARS: tl.constexpr = len(H[0])

    s = _get_syndrome_signs(syndrome, NCHECKS)

    E = ()
    for _ in tl.static_range(NCHECKS):
        row = ()
        for _ in tl.static_range(NVARS):
            row += (0.0,)
        E += (row,)

    for _ in range(NITER):
        V = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    msg = L0
                    for c2 in tl.static_range(NCHECKS):
                        if c2 != c and H[c2][v]:
                            msg += E[c2][v]
                    row += (msg,)
                else:
                    row += (0.0,)
            V += (row,)

        newE = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    sign_prod = s[c]
                    min_mag = BIG
                    for v2 in tl.static_range(NVARS):
                        if v2 != v and H[c][v2]:
                            m = V[c][v2]
                            sign_prod *= _sign1(m)
                            min_mag = tl.minimum(min_mag, tl.abs(m))
                    row += (ALPHA * sign_prod * min_mag,)
                else:
                    row += (0.0,)
            newE += (row,)
        E = newE

    P = ()
    for v in tl.static_range(NVARS):
        post = L0
        for c in tl.static_range(NCHECKS):
            if H[c][v]:
                post += E[c][v]
        P += (post,)
    return P
