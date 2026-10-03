"""Incremental 6D layout session over a growing set of landmarks.

The viewport shows the top 6 principal axes of the centered landmarks:
dims 1-3 as position, dims 4-6 as color (see oklab.py). Because the layout is a
LINEAR projection, any 6D coordinate maps back to a real embedding
(mean + coords @ components) — free-positioned cursor points are honest
selections in embedding space, not visual fictions.

Each landmark addition refits the PCA; the new basis is Procrustes-rotated to
best match the previous one so the world drifts instead of snapping when axes
flip or reorder between fits.
"""

from pathlib import Path

import numpy as np

from . import images
from .conditioning import Conditioning, blend_tokens
from .embedder import parse_layer_spec
from .oklab import zscores_to_hex

MAX_AXES = 6


def knn_radius(block: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row of `block` to its k-th nearest neighbor in
    `corpus`, skipping the row itself where it is a corpus member.

    Uses the |a|^2 - 2ab + |b|^2 identity. The subtract-then-norm form
    materializes an (n, m, dim) tensor — 38GB at the 2000-phrase boundary,
    which thrashed the box instead of raising. Computed in float64: in
    float32 a point's distance to itself comes out well above the self-skip
    threshold, so it was counted as its own neighbor."""
    block = np.asarray(block, dtype=np.float64)
    corpus = np.asarray(corpus, dtype=np.float64)
    if len(corpus) == 0:
        return np.full(len(block), np.nan)
    # A corpus smaller than k + 1 rows has no k-th neighbor past the point
    # itself: partition only as deep as the corpus goes, and take the
    # farthest there is rather than index out of range.
    kth = min(int(k), len(corpus) - 1)
    d2 = (
        np.sum(block**2, axis=1)[:, None]
        - 2.0 * (block @ corpus.T)
        + np.sum(corpus**2, axis=1)[None, :]
    )
    d2 = np.maximum(d2, 0.0)
    out = np.empty(len(block))
    for row in range(len(block)):
        d = np.sqrt(np.sort(np.partition(d2[row], kth)[: kth + 1]))
        start = 1 if d[0] < 1e-4 and len(d) > 1 else 0  # skip self when in corpus
        out[row] = d[min(start + k - 1, len(d) - 1)]
    return out


def procrustes_align(new_components: np.ndarray, old_components: np.ndarray) -> np.ndarray:
    """Rotate rows of new_components (k,d) to best match old_components (k,d)."""
    m = old_components @ new_components.T
    u, _s, vt = np.linalg.svd(m)
    return (u @ vt) @ new_components


class LayoutSession:
    def __init__(self, embedder, layer: int | tuple[int, int] | None = None):
        self.embedder = embedder
        # None = the embedder's configured (sampler) layer; int = one layer;
        # (lo, hi) = mean of pooled vectors over that inclusive layer range.
        self.layer = layer
        self.phrases: list[str] = []
        self.vectors: np.ndarray | None = None  # (n, d)
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (k, d), k <= MAX_AXES
        self.axis_std: np.ndarray | None = None    # (k,) std of members per axis
        self.explained: list[float] = []
        self._total_var: float = 0.0

    def _embed(self, text: str) -> np.ndarray:
        if self.layer is None:
            return self.embedder.embed(text)
        if isinstance(self.layer, tuple):
            lo, hi = self.layer
            return np.mean(
                [self.embedder.embed_layer(text, l) for l in range(lo, hi + 1)],
                axis=0,
            )
        return self.embedder.embed_layer(text, self.layer)

    @property
    def k(self) -> int:
        return 0 if self.components is None else self.components.shape[0]

    def add_landmark(self, text: str) -> None:
        self.add_landmarks([text])

    def add_landmarks(self, texts: list[str]) -> None:
        """Add many landmarks with ONE refit at the end. At universe scale
        (hundreds+) the per-addition refit is the difference between seconds
        and hours: each incremental SVD costs O(n^2 d) and they sum cubically."""
        seen = set(self.phrases)
        new = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                new.append(t)
        if not new:
            return
        if self.layer is None and hasattr(self.embedder, "embed_many"):
            block = self.embedder.embed_many(new, verbose=len(new) > 100)
        elif self.layer is not None and hasattr(self.embedder, "embed_layers_many"):
            block = self.embedder.embed_layers_many(new, self.layer,
                                                    verbose=len(new) > 100)
        else:
            block = np.vstack([self._embed(t) for t in new])
        self.phrases.extend(new)
        self.vectors = (
            block if self.vectors is None else np.vstack([self.vectors, block])
        )
        self._refit()

    def remove_landmark(self, text: str) -> bool:
        return self.remove_many([text]) > 0

    def remove_many(self, texts: list[str]) -> int:
        """Batch removal with ONE refit — a box-selected region is one
        deletion event, not len(region) sequential SVDs."""
        doomed = set(texts)
        idx = [i for i, p in enumerate(self.phrases) if p in doomed]
        if not idx:
            return 0
        for i in reversed(idx):
            self.phrases.pop(i)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        if len(self.phrases) == 0:
            self.vectors = None
        self._refit()
        return len(idx)

    def _refit(self) -> None:
        n = len(self.phrases)
        if n < 2:
            self.mean = None if n == 0 else self.vectors[0].copy()
            self.components = None
            self.axis_std = None
            self.explained = []
            return
        self.mean = self.vectors.mean(axis=0)
        centered = self.vectors - self.mean
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(MAX_AXES, n - 1)
        components = vt[:k]
        if self.components is not None and self.components.shape == components.shape:
            components = procrustes_align(components, self.components)
        self.components = components
        self._total_var = float((s**2).sum())
        self._refresh_axis_stats()

    def _refresh_axis_stats(self) -> None:
        proj = (self.vectors - self.mean) @ self.components.T  # (n, k)
        # Sample std of members along each (possibly rotated) axis; floor avoids
        # divide-by-zero on degenerate axes.
        self.axis_std = np.maximum(proj.std(axis=0), 1e-6)
        # Approximate after rotation: report per-axis member variance instead.
        # Both numerator and denominator are raw sums of squares (proj is
        # already centered) — proj.var here would shrink fractions by n.
        self.explained = list(
            ((proj**2).sum(axis=0) / self._total_var) if self._total_var > 0 else []
        )

    def align_axes(self, ref_components: np.ndarray | None) -> None:
        """Rotate this session's basis toward another session's (e.g. an
        adjacent layer's) so scrubbing across sessions morphs instead of
        snapping. A pure in-subspace rotation — geometry is unchanged."""
        if ref_components is None or self.components is None:
            return
        if ref_components.shape != self.components.shape:
            return
        self.components = procrustes_align(self.components, ref_components)
        self._refresh_axis_stats()

    def coords_of(self, vec: np.ndarray) -> np.ndarray:
        """Project to layout coords, zero-padded to MAX_AXES."""
        out = np.zeros(MAX_AXES, dtype=np.float64)
        if self.k:
            out[: self.k] = (vec - self.mean) @ self.components.T
        return out

    def zscores_of(self, coords: np.ndarray) -> np.ndarray:
        z = np.zeros(MAX_AXES, dtype=np.float64)
        if self.k:
            z[: self.k] = coords[: self.k] / self.axis_std
        return z

    def reconstruct(self, coords) -> np.ndarray:
        """6D layout coords -> full-dimensional embedding (the right-inverse)."""
        if self.k == 0:
            raise ValueError("need at least 2 landmarks to define a layout")
        coords = np.asarray(coords, dtype=np.float64)[: self.k]
        return self.mean + coords @ self.components

    def residual_norm(self, vec: np.ndarray) -> float:
        if self.k == 0:
            return 0.0
        centered = vec - self.mean
        within = ((centered @ self.components.T) @ self.components)
        return float(np.linalg.norm(centered - within))

    # A `basis` is another LayoutSession whose mean, components, and axis_std
    # do the projecting: a neighborhood drawn in its parent atlas's axes, so
    # the beachhead coordinate and the natives' positions survive the zoom.
    # None means this session's own chart. Every coordinate that comes in
    # (cursor, interrogate, blend, export) reconstructs through the same
    # basis it was read in, so a 6-vector always names one embedding.

    def landmark_entry(self, i: int, basis=None) -> dict:
        b = basis or self
        vec = self.vectors[i]
        coords = b.coords_of(vec)
        z = b.zscores_of(coords)
        return {
            "text": self.phrases[i],
            "coords": [round(float(c), 6) for c in coords],
            "color": zscores_to_hex(z[3], z[4], z[5]),
            "residual": round(b.residual_norm(vec), 4),
        }

    def layout(self, basis=None) -> dict:
        b = basis or self
        return {
            "n_landmarks": len(self.phrases),
            "n_axes": b.k,
            "axis_std": [round(float(s), 6) for s in (b.axis_std if b.k else [])],
            "explained": [round(float(e), 4) for e in b.explained],
            "landmarks": [self.landmark_entry(i, b) for i in range(len(self.phrases))],
        }

    def centroid(self, basis=None) -> dict | None:
        """The neighborhood's center, the mean of its landmarks, as a
        viewport point in the given basis. In its own basis that is the
        origin by construction; in a parent's it is where this neighborhood
        sits in the atlas. The one point two models can be compared at."""
        b = basis or self
        if self.mean is None or b.k == 0:
            return None
        coords = b.coords_of(self.mean)
        z = b.zscores_of(coords)
        return {"coords": [round(float(c), 6) for c in coords],
                "color": zscores_to_hex(z[3], z[4], z[5])}

    # `target`, where accepted, is a full-dimensional embedding that stands
    # in for the 6D coords: the neighborhood center is exactly the landmark
    # mean, which no chart but its own can name in six numbers (a coordinate
    # only reaches the chart's subspace; the mean's residual is lost). The
    # marker still sits at the projection, as every landmark's does.

    def interrogate(self, coords, k: int = 10, vocab=None, basis=None,
                    target=None) -> dict:
        """Nearest landmarks to a 6D viewport point, in the FULL space of the
        reconstructed embedding. `vocab` — an optional (texts, matrix) corpus
        of cached embeddings — additionally lights up what's nearby in the
        wider vocabulary, so a zoomed-in cursor is never navigating blind."""
        b = basis or self
        if target is None:
            target = b.reconstruct(coords)
        t_norm = np.linalg.norm(target)
        sims = []
        for i, phrase in enumerate(self.phrases):
            v = self.vectors[i]
            denom = t_norm * np.linalg.norm(v)
            sims.append((phrase, float(target @ v / denom) if denom else 0.0))
        sims.sort(key=lambda x: x[1], reverse=True)
        out = {
            "coords": [float(c) for c in np.asarray(coords, dtype=np.float64)],
            "residual_of_reconstruction": 0.0,  # by construction: point lies in-subspace
            "nearest_landmarks": [
                {"text": p, "cosine": round(s, 4)} for p, s in sims[:k]
            ],
        }
        if vocab is not None and vocab[1] is not None and t_norm > 0:
            texts, mat = vocab
            cos = (mat @ target) / (np.linalg.norm(mat, axis=1) * t_norm + 1e-12)
            landmarks = set(self.phrases)
            order = np.argsort(cos)[::-1]
            hits = [i for i in order if texts[i] not in landmarks][:k]
            # Layout coords ride along so the viewport can place ghost markers
            # where these phrases actually sit in the current chart.
            out["nearest_vocab"] = [
                {"text": texts[i], "cosine": round(float(cos[i]), 4),
                 "coords": [round(float(c), 6) for c in b.coords_of(mat[i])]}
                for i in hits
            ]
        return out

    def direction(self, phrases) -> np.ndarray:
        """The direction a named group pulls in: its centroid minus the
        universe mean, so it is an axis through the cloud rather than a point
        in it. This is what `bends` erases or amplifies."""
        idx = [self.phrases.index(p) for p in phrases if p in self.phrases]
        if not idx:
            raise ValueError(f"no landmarks named in {list(phrases)!r}")
        center = self.vectors[idx].mean(axis=0)
        return center - (self.mean if self.mean is not None else 0.0)

    def blend_weights(self, coords, basis=None, target=None,
                      positive=None, negative=None, bends=None,
                      strict: bool = False) -> dict:
        """Express a viewport point as least-squares weights over the landmark
        embeddings — the handoff format for ComfyUI conditioning blending.

        `positive` / `negative` name landmarks whose weights are held to a
        sign, which is what makes one neighborhood add and another subtract.
        Unnamed landmarks stay free, so declaring one pair is legal. With
        neither named this is the plain unconstrained solve it always was.

        `strict` holds every unnamed landmark non-negative, which is what
        actually delivers a negative neighborhood: without it, naming a
        negative pole usually changes nothing, since the free solve already
        drives every non-target landmark negative to pay for an overshoot.

        `bends` is [(phrases, amount), ...] applied to the TARGET before any
        solving: amount 1.0 erases that group's direction from the point, 0.0
        leaves it, a negative amount amplifies it. Erasing is how an
        association gets redirected instead of merely observed."""
        if target is None:
            target = (basis or self).reconstruct(coords)
        info_extra = {}
        if bends:
            from .compose import bend as _bend

            groups = [g for g, _ in bends]
            target = _bend(target, [self.direction(g) for g in groups],
                           [a for _, a in bends], origin=self.mean)
            info_extra["bends"] = [{"phrases": list(g), "amount": float(a)}
                                   for g, a in bends]
        if positive or negative:
            from .compose import poles_to_signs, signed_lstsq

            signs = poles_to_signs(self.phrases, positive, negative, strict)
            w = signed_lstsq(self.vectors.T, target, signs)
            info_extra["poles"] = {"positive": sorted(positive or ()),
                                   "negative": sorted(negative or ()),
                                   "strict": bool(strict)}
        else:
            w, *_ = np.linalg.lstsq(self.vectors.T, target, rcond=None)
        approx = self.vectors.T @ w
        err = float(np.linalg.norm(target - approx) / (np.linalg.norm(target) or 1.0))
        out = {
            "weights": {p: round(float(x), 6) for p, x in zip(self.phrases, w)},
            "relative_error": round(err, 6),
        }
        # The compass, reduced to two scalars. fraction: how much of the
        # recipe is subtraction. centroid_alignment: cosine between the
        # subtracted mass's direction and plain anti-centroid correction —
        # ~1.0 means the negative cone is just re-centering (no antonym
        # here); low alignment with real negative mass means the opposition
        # points somewhere specific: this neighborhood HAS an antonym axis.
        neg = w < 0
        total = float(np.abs(w).sum())
        fraction = float(-w[neg].sum() / total) if total > 0 else 0.0
        alignment = None
        if neg.any() and self.mean is not None:
            d_neg = ((-w[neg])[:, None] * (self.vectors[neg] - self.mean)).sum(axis=0)
            anti = self.mean - target
            denom = float(np.linalg.norm(d_neg) * np.linalg.norm(anti))
            if denom > 0:
                alignment = round(float(d_neg @ anti / denom), 4)
        out["negativity"] = {"fraction": round(fraction, 4),
                             "centroid_alignment": alignment}
        out.update(info_extra)
        return out

    # Injected blends ease back to zero influence over this many blocks above
    # the band; beyond them the model cooks the state freely to the top.
    EASE_LAYERS = 2

    def sampler_layers(self) -> tuple[int, ...] | None:
        """The layers this model's sampler reads, as the export must carry
        them. Embedders that predate the protocol fall back to their top
        hidden state, which is what a single-layer model always meant."""
        getter = getattr(self.embedder, "sampler_layers", None)
        if getter is not None:
            layers = getter()
            if layers:
                return tuple(int(l) for l in layers)
        n = self.embedder.layer_count()
        return None if n is None else (n - 1,)

    def _export_trim(self) -> int:
        """Tokens to drop from a finished export so it lands in the frame the
        sampler reads. Applied LAST, after any cook: the resumed pass needs the
        scaffold present, because attention over the prompt depends on it."""
        trim = getattr(self.embedder, "export_trim", None)
        return int(trim()) if trim is not None else 0

    def _frame_parts(self) -> tuple[int, int]:
        """(prefix tokens, tail tokens) every landmark's frame shares: the
        scaffold before and after the slot."""
        prefix = getattr(self.embedder, "template_prefix_tokens", None)
        tail = getattr(self.embedder, "template_tail_tokens", None)
        return (int(prefix()) if prefix else 0, int(tail()) if tail else 0)

    def _groups(self, weights) -> dict[str, list[tuple[str, float]]]:
        """Landmarks by modality, with their solved weights."""
        out: dict[str, list[tuple[str, float]]] = {}
        for p, w in zip(self.phrases, weights):
            kind = "image" if images.is_image_key(p) else "phrase"
            out.setdefault(kind, []).append((p, float(w)))
        return out

    def _blend_at(self, depths, weights, info: dict | None = None) -> dict[int, np.ndarray]:
        """The landmarks' per-token states at each layer, blended by weight.

        The frame is the one Krea 2's own image template uses, picture
        before text: prefix, the picture group's slot, the phrase group's
        slot, then the scaffold tail, which every landmark has at the same
        positions and so is blended over all of them. Slots are blended
        within each modality and concatenated across, because a word averaged
        with an image patch is a tensor no sequence could produce. Only a
        slot is zero-padded, so phrases of different token lengths still meet
        scaffold to scaffold; a whole-sequence pad slid a short phrase's tail
        onto a long phrase's words. With one modality and equal lengths this
        is exactly the position-by-position blend. `info`, if given,
        receives the groups, their weight sums, and the frame layout."""
        depths = [int(d) for d in depths]
        states = {p: self.embedder.conditioning_layers(p, depths)
                  for p in self.phrases}
        groups = self._groups(weights)
        if info is not None:
            info["groups"] = {k: len(v) for k, v in groups.items()}
            info["group_weights"] = {k: round(sum(w for _, w in v), 6)
                                     for k, v in groups.items()}
        prefix, tail = self._frame_parts()

        def slot(arr):
            return arr[prefix:arr.shape[0] - tail] if tail else arr[prefix:]

        out = {}
        for l in depths:
            parts = []
            if prefix:
                # The prefix rows are the same for every landmark: attention
                # is causal and the template before the slot is shared, so
                # nothing the phrase says can reach them. Copy one landmark's.
                # Blending them would scale them by the weights' sum, which
                # is 1 only for a plain local solve, and the scaled rows
                # (attention sink included) would steer the cook.
                parts.append(states[self.phrases[0]][l][:prefix])
            for kind in ("image", "phrase"):
                members = groups.get(kind, [])
                if members:
                    parts.append(blend_tokens([slot(states[p][l]) for p, _ in members],
                                              [w for _, w in members]))
            if tail:
                parts.append(blend_tokens([states[p][l][-tail:] for p in self.phrases],
                                          weights))
            out[l] = np.concatenate(parts, axis=0)
        # Only a mixed frame is recorded: it is what switches the cook to
        # the 3-D image positions, and text keeps its 1-D ones.
        if info is not None and "frame" not in info and len(groups) > 1:
            segs = []
            if prefix:
                segs.append(["prefix", prefix])
            for kind in ("image", "phrase"):
                members = groups.get(kind, [])
                if members:
                    n = max(slot(states[p][depths[0]]).shape[0] for p, _ in members)
                    segs.append([kind, n])
            if tail:
                segs.append(["tail", tail])
            info["frame"] = segs
        return out

    def _cook_positions(self, info: dict):
        """3-D rotary positions for a mixed frame's cook, None for text."""
        frame = info.get("frame")
        getter = getattr(self.embedder, "mixed_position_ids", None)
        if not frame or getter is None:
            return None
        segments = []
        for kind, n in frame:
            if kind == "image":
                # The slot of an `image`-pooled picture is start + patches +
                # end; a `tail`-pooled picture's slot is <|vision_end|> alone,
                # which is text-shaped and takes a text position.
                segments.append(("image", n) if n > 1 else ("text", n))
            else:
                segments.append(("text", n))
        return getter(segments)

    def export_conditioning(self, coords, band=None, basis=None,
                            target=None, positive=None, negative=None,
                            bends=None, strict: bool = False
                            ) -> tuple[Conditioning, dict]:
        """The take-home artifact: per-token conditioning at a viewport point,
        at every layer the sampler reads.

        Blend weights are solved in pooled space (where navigation lives), then
        applied to the landmarks' full per-token tensors.

        band=None (or a band whose top is the sampler layer): the classic
        linear blend of sampler-layer tensors.

        band=(lo, hi) with hi below the top sampler layer: hi is the inserter. The
        blend is assembled from the landmarks' layer-hi hidden states and the
        remaining blocks run forward from it (easing the placement out over
        EASE_LAYERS blocks), so the model's own dynamics pull the mixture back
        onto the manifold of real text before it becomes conditioning. One
        resumed pass fills every sampler layer above the inserter. Sampler
        layers at or below it sit upstream of the patch and cannot be cooked,
        so they stay a direct blend. The band's low edge shapes the chart
        blend weights solve in; its high edge sets where cooking begins.
        """
        blend = self.blend_weights(coords, basis, target, positive=positive,
                                   negative=negative, bends=bends,
                                   strict=strict)
        weights = list(blend["weights"].values())
        blend["cooked"] = False
        sampler = self.sampler_layers()
        if band is not None and sampler and hasattr(self.embedder, "cook_layers"):
            hi = max(int(band[0]), int(band[1]))
            top = max(sampler)
            cookable = [d for d in sampler if d > hi]
            if cookable and 0 <= hi < top:
                need = [hi] + [l for l in range(hi + 1,
                                                min(hi + 1 + self.EASE_LAYERS, top))]
                blend_at = self._blend_at(need, weights, blend)
                pulls = {
                    l: (1.0 - (l - hi) / (self.EASE_LAYERS + 1.0), blend_at[l])
                    for l in need[1:]
                }
                extra = {}
                positions = self._cook_positions(blend)
                if positions is not None:
                    extra["position_ids"] = positions
                cooked = self.embedder.cook_layers(
                    blend_at[hi], hi, pulls, stop_index=top, capture=cookable,
                    **extra)
                tensors = {d: cooked[d] for d in cookable}
                upstream = [d for d in sampler if d <= hi]
                if upstream:
                    tensors.update(self._blend_at(upstream, weights))
                blend["cooked"] = True
                blend["band"] = [int(min(band)), hi]
                return self._finish(Conditioning(tensors), blend)
        if sampler:
            return self._finish(Conditioning(self._blend_at(sampler, weights, blend)),
                                blend)
        conds = [self.embedder.conditioning(p) for p in self.phrases]
        return self._finish(Conditioning.blend(conds, weights), blend)

    def _finish(self, cond: Conditioning, blend: dict):
        trim = self._export_trim()
        blend["template_prefix_tokens"] = trim
        return (cond.trim(trim) if trim else cond), blend


class LayerStack:
    """One landmark set viewed at many layers: a primary session at the
    embedder's configured (sampler) layer plus lazily built per-layer sessions.

    Each new layer session's axes are Procrustes-chained to the nearest
    already-built layer, so scrubbing layer morphs the world instead of
    snapping between arbitrary PCA orientations. Export semantics stay honest:
    blend weights solve in whichever layer you're viewing, and the tensor is
    either a sampler-layer blend (top-of-range view) or a placement at the inserter from
    the scrubbed band's top — always a valid sampler-layer conditioning
    (see LayoutSession.export_tensor)."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.primary = LayoutSession(embedder)
        self.by_layer: dict[int, LayoutSession] = {}
        # The atlas this neighborhood was carved from, kept as a whole stack
        # so its chart can be asked for at any layer. Set when colonizing
        # in-session, or rebuilt from the universe file the header names,
        # so it survives a restart and a model swap.
        self.world: "LayerStack | None" = None
        # The universe file this stack was loaded from or written to.
        self.source_path: str | None = None
        # phrase -> "manual" | "preload"; the viewport renders them differently
        # so your own additions never blend into the surveyed background.
        self.origin: dict[str, str] = {}
        # Metadata of the universe file this stack was loaded from, if any
        # (see tokencollider.provenance) — exports cite it as their origin.
        self.provenance: dict | None = None
        self._vocab_cache: dict[str, tuple[int, list[str], object]] = {}
        # (layer_key, corpus_size) -> {phrase: kNN distance into the vocab}.
        # Per-phrase, so a landmark add over an unchanged corpus only costs
        # the newcomer. A grown corpus moves every radius, so it starts over;
        # each layer keeps only its latest corpus (see _evict_stale).
        self._density_cache: dict[tuple, dict[str, float]] = {}
        # (layer_key, corpus_size) -> ascending decile cut points of the
        # corpus's own kNN-distance distribution — the universal density scale.
        self._threshold_cache: dict[tuple, np.ndarray] = {}

    @property
    def phrases(self) -> list[str]:
        return self.primary.phrases

    def layer_key(self, layer=None) -> str:
        """The cache key string under which this view's pooled vectors live."""
        layer = self._normalize(layer)
        if layer is None:
            # "last" is a request, not a key. The embedder caches pooled
            # vectors under the layer it resolves to (str(_resolved_layer())),
            # so handing back the literal pointed vocab() at a slice nothing
            # is ever written to: an empty corpus, and every universe silently
            # demoted to the n^2 density fallback. Band configs already carry
            # their own key ("18-34") and pass through unchanged.
            declared = getattr(self.embedder, "layer", "last")
            if declared == "last":
                n = self.layer_count()
                return "last" if n is None else str(n - 1)
            return str(declared)
        if isinstance(layer, tuple):
            return f"{layer[0]}-{layer[1]}"
        return str(layer)

    def vocab(self, layer=None):
        """The interrogator's corpus: every cached embedding under this view's
        config slice, as (texts, matrix). Cache-only — never a forward pass —
        and held in memory, invalidated by row count when warms land."""
        store = getattr(self.embedder, "store", None)
        if store is None:
            return [], None
        key = self.layer_key(layer)
        model = self.embedder.model_name
        pooling = self.embedder.pooling
        template = self.embedder.template
        n = store.count_for(model, key, pooling, template)
        cached = self._vocab_cache.get(key)
        if cached is not None and cached[0] == n:
            return cached[1], cached[2]
        texts, mat = store.vectors_for(model, key, pooling, template)
        self._vocab_cache[key] = (n, texts, mat)
        return texts, mat

    DENSITY_K = 8

    def _corpus_thresholds(self, layer, mat) -> np.ndarray:
        """Decile cut points of the CORPUS's own kNN-distance distribution
        (estimated from a fixed sample). This makes density digits universal:
        a 7 means denser than ~70% of everything cached, in any universe or
        neighborhood — not merely dense relative to the loaded cluster."""
        key = (self.layer_key(layer), len(mat))
        cached = self._threshold_cache.get(key)
        if cached is not None:
            return cached
        rng = np.random.default_rng(0)  # fixed seed: stable scale per corpus
        n_sample = min(2048, len(mat))
        sample = mat[rng.choice(len(mat), n_sample, replace=False)]
        k = max(1, min(self.DENSITY_K, len(mat) - 1))
        dks = knn_radius(sample, mat, k)  # every sample point is in mat
        thresholds = np.quantile(dks, np.linspace(0.1, 0.9, 9))
        self._evict_stale(self._threshold_cache, key)
        self._threshold_cache[key] = thresholds
        return thresholds

    @staticmethod
    def _evict_stale(cache: dict, key: tuple) -> None:
        """Drop entries for the same layer under an older corpus size. They
        can never be read again, and every warm that adds rows made one."""
        for old in [k for k in cache if k[0] == key[0] and k != key]:
            del cache[old]

    @staticmethod
    def _digit(dk: float, thresholds: np.ndarray) -> int:
        # Small distance = dense = high digit; below the 10th percentile -> 9.
        return int(9 - np.searchsorted(thresholds, dk))

    def density_digits(self, layer=None) -> dict[str, int] | None:
        """Per-landmark crowding of the cached-vocab space, as decile digits:
        9 = the densest tenth of this universe, 0 = the sparsest — the void
        frontier. High-D voids don't survive a 3D projection, so density has
        to be worn by the nodes themselves. Falls back to landmark-vs-landmark
        spacing when no vocab is cached (small/dev universes only)."""
        session = self.session(layer)
        phrases = session.phrases
        if len(phrases) < 2 or session.vectors is None:
            return None
        texts, mat = self.vocab(layer)
        if mat is not None:
            key = (self.layer_key(layer), len(texts))
            if key not in self._density_cache:
                self._evict_stale(self._density_cache, key)
            known = self._density_cache.setdefault(key, {})
            missing = [i for i, p in enumerate(phrases) if p not in known]
            if missing:
                k = max(1, min(self.DENSITY_K, len(mat) - 1))
                radii = knn_radius(session.vectors[missing], mat, k)
                for row, i in enumerate(missing):
                    known[phrases[i]] = float(radii[row])
            thresholds = self._corpus_thresholds(layer, mat)
            return {p: self._digit(known[p], thresholds) for p in phrases}
        # No vocab corpus: only relative spacing exists — decile ranks within
        # the loaded set (dev/fake universes; digits are local here by nature).
        if len(phrases) >= 2000:
            return None  # n^2 fallback is for dev-scale universes only
        k = min(4, len(phrases) - 1)
        dk = knn_radius(session.vectors, session.vectors, k)
        ranks = np.argsort(np.argsort(dk))
        digits = 9 - (ranks * 10) // len(phrases)
        return {p: int(digits[i]) for i, p in enumerate(phrases)}

    def void_at(self, layer, coords, basis=None, target=None) -> dict | None:
        """How unmapped a point is: its kNN distance into the cached vocab,
        plus the ratio to the universe's median landmark crowding — ratio > 1
        means the cursor sits in sparser space than a typical landmark."""
        session = self.session(layer)
        if (basis or session).k == 0:
            return None
        texts, mat = self.vocab(layer)
        if mat is None:
            return None
        if target is None:
            target = (basis or session).reconstruct(coords)
        k = max(1, min(self.DENSITY_K, len(mat) - 1))
        # Same self-skipping radius the landmark digits wear, so the ratio and
        # digit below are measured on the scale they are compared against.
        kth = float(knn_radius(target[None, :], mat, k)[0])
        out = {"kth_distance": round(kth, 3)}
        # Same universal scale the landmark digits wear: ratio vs the corpus
        # median crowding, digit from the corpus decile thresholds.
        thresholds = self._corpus_thresholds(layer, mat)
        med = float(thresholds[4])
        if med > 0:
            out["ratio"] = round(kth / med, 2)
        out["digit"] = self._digit(kth, thresholds)
        return out

    def world_path(self) -> str | None:
        """Where the atlas lives on disk, if anywhere: the live world's file,
        else what the header names (`world.path`, falling back to the
        immediate `parent.path`)."""
        if self.world is not None and self.world.source_path:
            return self.world.source_path
        from .provenance import is_universe_file

        # Header paths are relative to the file that names them. Only a file
        # that is itself a universe is followed: a shared header could name
        # anything on disk, and this file's phrases get embedded and cached.
        base = Path(self.source_path).parent if self.source_path else Path.cwd()
        meta = self.provenance or {}
        for key in ("world", "parent"):
            raw = (meta.get(key) or {}).get("path")
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            if is_universe_file(path):
                return str(path)
        return None

    def has_world(self) -> bool:
        """Whether world axes exist for this stack, without building them."""
        return self.world is not None or self.world_path() is not None

    def ensure_world(self) -> "LayerStack | None":
        """The atlas as a stack, rebuilding it from its universe file on
        first use. Phrases come from the cache when they were warmed under
        this model; otherwise this is the warm, which is the price of asking
        for another model's view of the same atlas."""
        if self.world is None:
            path = self.world_path()
            if path is not None:
                from . import images
                from . import provenance as prov

                entries, _meta = prov.read_universe(Path(path))
                phrases = images.resolve_entries(
                    entries, Path(path).parent,
                    getattr(self.embedder, "store", None),
                    getattr(self.embedder, "image_pooling", "image"))
                if phrases:
                    world = LayerStack(self.embedder)
                    world.source_path = path
                    world.add_landmarks(phrases, source="preload")
                    self.world = world
        return self.world

    def basis(self, layer=None, axes: str = "local"):
        """The session to project through: the atlas's chart at this layer
        for "world", None (this stack's own chart) otherwise or when there
        is no atlas."""
        if axes != "world":
            return None
        world = self.ensure_world()
        return None if world is None else world.session(layer)

    def replace_landmarks(self, phrases: list[str], origin: dict[str, str],
                          provenance: dict | None = None) -> None:
        """Swap in a whole new landmark set (colonizing a neighborhood):
        fresh primary session, per-layer sessions dropped and rebuilt lazily.

        The chart being left becomes the world, unless one already exists:
        a neighborhood of a neighborhood still measures against the atlas.
        The new chart is Procrustes-aligned to it, so even local axes drift
        from the world's rather than snapping to an arbitrary orientation."""
        if self.ensure_world() is None:
            world = LayerStack(self.embedder)
            world.primary, world.by_layer = self.primary, self.by_layer
            world.origin, world.provenance = self.origin, self.provenance
            world.source_path = self.source_path
            self.world = world
        self.primary = LayoutSession(self.embedder)
        self.by_layer = {}
        self.origin = dict(origin)
        self.provenance = provenance
        self.source_path = None
        self.primary.add_landmarks(phrases)
        self.primary.align_axes(self.world.primary.components)

    def layer_count(self) -> int | None:
        return self.embedder.layer_count()

    def primary_index(self) -> int | tuple[int, int] | None:
        """The primary session's layer: a concrete hidden-state index, or
        (lo, hi) for a band-configured embedder (layer: 18-34)."""
        spec = parse_layer_spec(getattr(self.embedder, "layer", "last"))
        if spec == "last":
            n = self.layer_count()
            return None if n is None else n - 1
        return self._normalize(spec)

    def resolve_layer(self, layer):
        if layer is None:
            layer = self.primary_index()
        return list(layer) if isinstance(layer, tuple) else layer

    @staticmethod
    def _normalize(layer):
        """(n, n) collapses to n so range and single views share sessions."""
        if isinstance(layer, tuple):
            lo, hi = sorted(int(l) for l in layer)
            return lo if lo == hi else (lo, hi)
        return layer

    @staticmethod
    def _midpoint(layer) -> float:
        return sum(layer) / 2 if isinstance(layer, tuple) else float(layer)

    def session(self, layer=None) -> LayoutSession:
        layer = self._normalize(layer)
        if layer is None or layer == self.primary_index():
            return self.primary
        if layer not in self.by_layer:
            s = LayoutSession(self.embedder, layer=layer)
            s.add_landmarks(self.primary.phrases)
            # A neighborhood aligns to the world's nearest already-built
            # chart (never building one: that could mean warming the whole
            # atlas at a new layer just to orient six axes); an atlas aligns
            # to its own nearest layer.
            ref = (self.world._nearest_components(layer) if self.world is not None
                   else self._nearest_components(layer))
            s.align_axes(ref)
            self.by_layer[layer] = s
        return self.by_layer[layer]

    def _nearest_components(self, layer) -> np.ndarray | None:
        if not self.by_layer:
            return self.primary.components
        mid = self._midpoint(layer)
        nearest = min(self.by_layer, key=lambda built: abs(self._midpoint(built) - mid))
        return self.by_layer[nearest].components

    def trail_depths(self) -> range:
        """The layers a trail covers: every raw pre-norm state up to and
        including the deepest one this model's sampler reads.

        The last hidden state (index n_blocks) is the post-norm final state,
        and it is a different kind of object from every other point on the
        trail: the only normed one, with a norm an order of magnitude smaller.
        No shipped profile reads it (Z-Image samples 35, Krea 2 tops out at
        35), so charting it just puts a meaningless kink on the end of every
        trail. A config that genuinely samples it (--layer last) still gets
        it, because then it is the destination rather than an artifact."""
        n = self.layer_count()
        if n is None:
            raise ValueError("model layer count unknown — add a landmark first")
        sampler = self.primary.sampler_layers()
        top = max(sampler) if sampler else n - 1
        return range(min(top + 1, n))

    def trajectories(self) -> dict:
        """Every landmark's 6D coords at every charted layer — the
        constellation data.

        Coordinates come from each layer's own (Procrustes-chained) session,
        i.e. exactly the positions the landmark visits as the layer slider
        sweeps. Building all sessions on first call is cheap: pooled vectors
        are cache hits and each PCA fit is over (n_landmarks, dim)."""
        depths = self.trail_depths()
        out = {p: [] for p in self.phrases}
        extent = 0.0
        for layer in depths:
            s = self.session(layer)
            for i, phrase in enumerate(s.phrases):
                coords = s.coords_of(s.vectors[i])
                z = s.zscores_of(coords)
                extent = max(extent, *(abs(float(c)) for c in coords[:3]))
                out[phrase].append({
                    "coords": [round(float(c), 6) for c in coords],
                    # That layer's own color axes: entanglement per layer,
                    # same mapping the landmark spheres use.
                    "color": zscores_to_hex(z[3], z[4], z[5]),
                })
        # n_layers is the number of points per trail, which is what the
        # viewport iterates; layers names the hidden-state index of each.
        return {"n_layers": len(depths), "depths": list(depths),
                "extent": round(extent, 6), "trajectories": out}

    def add_landmark(self, text: str) -> None:
        self.add_landmarks([text])

    def add_landmarks(self, texts: list[str], source: str = "manual") -> None:
        if any(not str(t).strip() for t in texts):
            # Nothing to embed: the encoder returns no phrase tokens, which
            # used to surface as a confusing "layer out of range".
            raise ValueError("an empty phrase has no tokens to embed")
        for t in texts:
            self.origin.setdefault(t, source)
        self.primary.add_landmarks(texts)
        for s in self.by_layer.values():
            s.add_landmarks(texts)

    def remove_landmark(self, text: str) -> bool:
        return self.remove_landmarks([text]) > 0

    def remove_landmarks(self, texts: list[str]) -> int:
        removed = self.primary.remove_many(texts)
        for s in self.by_layer.values():
            s.remove_many(texts)
        for t in texts:
            self.origin.pop(t, None)
        return removed
