import triton
import triton.language as tl


@triton.jit
def _hard_decision(P):
    """Pack negative posterior values into a correction mask."""
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    mask = zero
    for i in tl.static_range(len(P)):
        mask = mask | (tl.where(P[i] < 0.0, one, zero) << i)
    return mask


@triton.jit
def _osd(P, syndrome):
    """Build an order-zero one-bit correction for a nonzero syndrome."""
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    bi = zero
    best = P[0]
    for i in tl.static_range(1, len(P)):
        c = P[i] < best
        bi = tl.where(c, tl.cast(i, tl.uint64), bi)
        best = tl.where(c, P[i], best)
    return tl.where(syndrome != 0, one << bi, zero)


@triton.jit
def _apply_postprocess(P, syndrome, postprocess: tl.constexpr = "hard"):
    """Convert posterior values to a correction mask."""
    tl.static_assert(
        (postprocess == "hard") or (postprocess == "osd"),
        "unknown postprocess",
    )
    if postprocess == "osd":
        return _osd(P, syndrome)
    return _hard_decision(P)
