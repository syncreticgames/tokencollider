"""Composition: put opposition INTO the blend, and bend the associations.

The finding in `tests/synthetic.py` is the reason this module exists. The
viewport's red tethers look like opposition and are not: extrapolate past any
cluster and unconstrained least squares pays for the overshoot by subtracting
every other landmark equally, whether or not a real antonym axis is there.
The opposition survives in the geometry, the solve throws it away.

Two ways to put it back, and they are independent:

- `signed_lstsq` constrains the solve. Declare a positive pole and a negative
  pole and the weights are forced to honour them, so a red tether means "you
  said this one subtracts" instead of "the cursor left the hull".
- `bend` moves the target before any solve happens. Project a named direction
  out of it and the export stops carrying that association; project with a
  negative amount and it carries more. This is the knob for redirecting an
  attraction rather than merely reading it.

No scipy: `nnls` is Lawson-Hanson, which terminates and is exact, and mixed
sign constraints reduce to it by negating the negative pole's columns and
splitting free landmarks into a positive and a negative part.
"""

import numpy as np

FREE, POSITIVE, NEGATIVE = 0, 1, -1


def nnls(A: np.ndarray, b: np.ndarray, maxiter: int | None = None):
    """min ||Ax - b|| subject to x >= 0. Lawson-Hanson active set."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    tol = 1e-10 * max(1.0, float(np.abs(A).max()))
    maxiter = maxiter or 3 * n
    for _ in range(maxiter):
        free = ~passive
        if not free.any() or w[free].max() <= tol:
            break
        j = int(np.argmax(np.where(free, w, -np.inf)))
        passive[j] = True
        for _ in range(maxiter):
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if s[passive].min() > 0:
                x = s
                break
            bad = passive & (s <= 0)
            ratio = x[bad] / np.where(x[bad] - s[bad] == 0, 1e-300, x[bad] - s[bad])
            x = x + float(ratio.min()) * (s - x)
            passive &= x > 1e-12
        else:
            break
        w = A.T @ (b - A @ x)
    return x


def signed_lstsq(A: np.ndarray, b: np.ndarray, signs) -> np.ndarray:
    """min ||Ax - b|| with each x[j] held to the sign in `signs`.

    +1 forces a landmark to add, -1 forces it to subtract, 0 leaves it free.
    A free landmark is two non-negative parts, so the whole problem is one
    NNLS and inherits its exactness.
    """
    A = np.asarray(A, dtype=np.float64)
    signs = np.asarray(signs, dtype=int)
    cols, back = [], []
    for j, s in enumerate(signs):
        if s >= 0:
            cols.append(A[:, j]); back.append((j, 1.0))
        if s <= 0:
            cols.append(-A[:, j]); back.append((j, -1.0))
    y = nnls(np.stack(cols, axis=1), b)
    x = np.zeros(A.shape[1])
    for k, (j, sgn) in enumerate(back):
        x[j] += sgn * y[k]
    return x


def poles_to_signs(phrases, positive=None, negative=None,
                   strict: bool = False) -> np.ndarray:
    """Landmark names to a sign vector.

    Unnamed landmarks stay free by default, so a partial declaration is legal.

    `strict` is what actually buys a negative NEIGHBORHOOD. Naming a negative
    pole on its own usually changes nothing, because the unconstrained solve
    already drives every non-target landmark negative to pay for an overshoot;
    the constraint is satisfied before it is applied. Under `strict` every
    unnamed landmark is held non-negative, so the declared pole is the only
    thing permitted to subtract and a red tether finally means what it looks
    like it means.
    """
    pos, neg = set(positive or ()), set(negative or ())
    clash = pos & neg
    if clash:
        raise ValueError(f"landmarks in both poles: {sorted(clash)}")
    default = POSITIVE if strict else FREE
    return np.array([POSITIVE if p in pos else NEGATIVE if p in neg else default
                     for p in phrases], dtype=int)


def bend(target: np.ndarray, directions, amounts, origin=None) -> np.ndarray:
    """Move `target` along named directions before the blend is solved.

    amount 1.0 removes that direction's component entirely (the association is
    erased), 0.0 leaves it, and a negative amount amplifies it. Directions are
    orthonormalised in the order given, so order matters when they overlap;
    the first one listed keeps its full direction and later ones keep only
    what is new.

    `origin` is where the directions are measured from, and it must be the
    same place they were built from. A group's direction is its centroid minus
    the universe mean, so bending has to happen in mean-centered space too;
    doing it on the raw vector removes the mean's own component along the
    direction as well and overshoots past zero into a sign flip.
    """
    o = 0.0 if origin is None else np.asarray(origin, dtype=np.float64)
    out = np.asarray(target, dtype=np.float64) - o
    basis = []
    for d in directions:
        u = np.asarray(d, dtype=np.float64).copy()
        for prev in basis:                       # Gram-Schmidt against earlier
            u -= (u @ prev) * prev
        norm = float(np.linalg.norm(u))
        if norm < 1e-12:                         # nothing new in this direction
            basis.append(np.zeros_like(u))
            continue
        basis.append(u / norm)
    for u, a in zip(basis, amounts):
        if np.any(u):
            out = out - float(a) * float(out @ u) * u
    return out + o
