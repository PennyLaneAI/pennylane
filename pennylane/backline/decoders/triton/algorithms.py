import triton
import triton.language as tl

from .posteriors import _norm_min_sum_posteriors, _sum_product_posteriors
from .postprocess import _apply_postprocess


@triton.jit
def _decode_one(
    syndrome,
    H: tl.constexpr,
    bp_variant: tl.constexpr = "sum_product",
    postprocess: tl.constexpr = "hard",
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
    ALPHA: tl.constexpr = 0.75,
):
    """Decode one syndrome into a correction."""
    NCHECKS: tl.constexpr = len(H)
    NVARS: tl.constexpr = len(H[0])
    tl.static_assert(
        (bp_variant == "sum_product") or (bp_variant == "norm_min_sum"),
        "unknown belief-propagation variant",
    )
    tl.static_assert(NCHECKS <= 64, "decoder supports <= 64 check bits")
    tl.static_assert(NVARS <= 64, "decoder supports <= 64 variable bits")

    syndrome = tl.cast(syndrome, tl.uint64)

    if bp_variant == "sum_product":
        P = _sum_product_posteriors(syndrome, H, prob, NITER)
    else:
        P = _norm_min_sum_posteriors(syndrome, H, prob, NITER, ALPHA)
    return _apply_postprocess(P, syndrome, postprocess=postprocess)
