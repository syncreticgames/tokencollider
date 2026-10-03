"""Paths between landmarks: slerp, with the scaffold kept where it belongs.

Sampling a path is a different question from sampling a point. A blend at a
cursor asks "what is here"; a path asks "what lies between", and the answer
should move at a constant rate rather than sagging through the middle.

Two things have to be right.

**Alignment.** A Krea 2 export is a few phrase tokens followed by a fixed
five-token scaffold tail. Phrases are ragged (`violin` trims to 7 tokens,
`trumpet` to 8), and `blend_tokens` zero-pads on the right, which slides the
shorter phrase's tail one position left and averages its `<|im_end|>` against
the other's phrase token. That is silent nonsense at the exact positions the
model attends to most. Here the tail is matched from the END and the phrase
region from the front, so scaffold only ever meets scaffold.

**Slerp itself.** Linear interpolation cuts the chord; slerp follows the arc,
so the interpolant keeps the norm of its endpoints and moves at a constant
angular rate. On this encoder the two are close, because landmark states are
anisotropic (pairs sit at cosine 0.97 to 0.99, roughly 11 degrees apart) and
sin is nearly linear over a small angle. Close is not the same as equal, and
a 2.2 degree change on one phrase token was measured flipping a render,
so the difference sits exactly at the scale that matters.
Whether it matters in the image is a question for a render, not for this
module.
"""

import numpy as np



def slerp(a: np.ndarray, b: np.ndarray, t: float, eps: float = 1e-7) -> np.ndarray:
    """Spherical interpolation between two vectors, falling back to linear
    where the arc is undefined: a near-zero endpoint (padding), or endpoints
    so nearly parallel that sin(omega) underflows."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return ((1.0 - t) * a + t * b).astype(np.float32)
    cos = float(np.clip((a / na) @ (b / nb), -1.0, 1.0))
    omega = float(np.arccos(cos))
    if np.sin(omega) < eps:
        return ((1.0 - t) * a + t * b).astype(np.float32)
    s = np.sin(omega)
    out = (np.sin((1.0 - t) * omega) / s) * a + (np.sin(t * omega) / s) * b
    return out.astype(np.float32)


def align(a: np.ndarray, b: np.ndarray, tail: int):
    """Pair up two ragged token sequences so scaffold meets scaffold.

    Returns (a2, b2, n_phrase): both padded to one length, phrase region
    front-aligned and the tail end-aligned. The shorter phrase is padded on
    its right, INSIDE the phrase region, so the tail never slides.
    """
    pa, pb = a.shape[0] - tail, b.shape[0] - tail
    if pa < 0 or pb < 0:
        raise ValueError(f"sequence shorter than the {tail}-token tail: "
                         f"{a.shape[0]}, {b.shape[0]}")
    n = max(pa, pb)
    dim = a.shape[1]

    def pad(x, p):
        out = np.zeros((n + tail, dim), dtype=x.dtype)
        out[:p] = x[:p]                 # phrase, front-aligned
        out[n:] = x[p:]                 # tail, end-aligned
        return out

    return pad(a, pa), pad(b, pb), n


def path_tokens(a: np.ndarray, b: np.ndarray, t: float, tail: int,
                mode: str = "slerp") -> np.ndarray:
    """One frame of a path between two per-token tensors."""
    a2, b2, _ = align(a, b, tail)
    if mode == "lerp":
        return ((1.0 - t) * a2 + t * b2).astype(np.float32)
    if mode != "slerp":
        raise ValueError(f"unknown mode {mode!r}")
    return np.stack([slerp(a2[i], b2[i], t) for i in range(a2.shape[0])])
