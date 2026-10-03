"""A synthetic encoder with PLANTED structure, for tests that can assert.

`FakeEmbedder` gives deterministic vectors with random geometry: enough to
drive the Godot client without a GPU, useless for asserting anything about a
blend. Nothing is true of a random universe, so every test built on it can
only check plumbing.

This one plants the structure the assertions need:

- **Clusters.** Each named cluster owns a direction. Its phrases sit near that
  direction with a small deterministic spread, so "which cluster is this point
  in" has a right answer.
- **Opposition.** A pair of clusters named in `opposites` shares ONE axis at
  +c and -c. That is a real antonym axis: a cursor pushed past one cluster has
  to take negative weight on the other to be expressed. Clusters not named in
  a pair get independent axes, so any negative mass they pick up is only
  re-centering. Those are the two cases `LayoutSession.blend_weights`
  distinguishes with `negativity.centroid_alignment`, and now they can be
  tested apart.
- **A layer schedule, per modality.** Low layers weight a surface feature
  (shared prefixes), high layers weight the planted cluster, peaking somewhere
  mid-network. That is the U shape the layer sweep found, so a test can assert that clusters tear apart
  and re-form as the viewport scrubs, instead of assuming it.

  Pictures and words peak at DIFFERENT layers, and that is the point. Z-Image
  had one modality and so one summit; Krea 2 reads a vision tower and a text
  tower through the same stack, and the two never coincide at any layer. A single global peak would bake the
  Z-Image assumption into every test built on this. `peaks` sets them apart,
  so "no chart layer is right for a mixed universe" becomes a property a test
  can hold the code to rather than a caveat in prose.

Cooking is faithful to the same schedule: a state placed at the inserter is
rotated toward the high-layer configuration by the passes above it. A low
inserter therefore reinterprets and a high one transcribes, which is the
qualitative claim that could not previously be checked.

Ground truth is public (`cluster_of`, `axis_of`, `opposites`) so tests assert
against what was planted rather than against a previous run's output.
"""

import hashlib
import math

import numpy as np

from .conditioning import Conditioning
from .embedder import FakeEmbedder

SAMPLER_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def _seed(*parts: str) -> int:
    h = hashlib.sha256("\x00".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


class SyntheticEmbedder(FakeEmbedder):
    """A universe whose geometry is known in advance.

    clusters: {name: [phrase, ...]}
    opposites: [(name_a, name_b), ...] sharing one axis at +c and -c.
    peaks: {"text": layer, "image": layer} — where each modality's concept is
        strongest. Defaults put text at half the stack (the measured Z-Image
        summit) and pictures higher, since the vision tower's tokens need more
        of the stack before they read as meaning.
    """

    def __init__(self, clusters: dict[str, list[str]],
                 opposites: list[tuple[str, str]] | None = None,
                 dim: int = 64, n_layers: int = 37,
                 sampler_layers=SAMPLER_LAYERS,
                 spread: float = 0.15, seed: int = 0,
                 peaks: dict[str, int] | None = None):
        super().__init__(dim=dim, n_layers=n_layers)
        self.clusters = {k: list(v) for k, v in clusters.items()}
        self.opposites = [tuple(p) for p in (opposites or [])]
        self.spread = float(spread)
        self._sampler = tuple(sorted(int(l) for l in sampler_layers))
        self.model_name = "synthetic"
        self.pooling = "mean"
        self.template = "{}"
        top = n_layers - 1
        self.peaks = {"text": int(round(top / 2.0)), "image": int(round(top * 0.72))}
        self.peaks.update({k: int(v) for k, v in (peaks or {}).items()})
        self._of = {p: name for name, ps in self.clusters.items() for p in ps}
        self._axes = self._build_axes(seed)

    # --- ground truth ----------------------------------------------------

    def cluster_of(self, phrase: str) -> str | None:
        return self._of.get(phrase)

    def axis_of(self, cluster: str) -> np.ndarray:
        """The planted direction, sign included. Opposed clusters return the
        same axis negated, which is what makes their weights trade off."""
        return self._axes[cluster].copy()

    def phrases(self) -> list[str]:
        return [p for ps in self.clusters.values() for p in ps]

    def _build_axes(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        basis = np.linalg.qr(rng.standard_normal((self.dim, self.dim)))[0]
        axes, used, col = {}, set(), 0
        for a, b in self.opposites:          # one axis, two signs
            axes[a] = basis[:, col].copy()
            axes[b] = -basis[:, col]
            used.update((a, b))
            col += 1
        for name in self.clusters:           # everyone else gets their own
            if name not in used:
                axes[name] = basis[:, col].copy()
                col += 1
        self._noise_basis = basis[:, col:]
        return axes

    # --- geometry --------------------------------------------------------

    def _semantic(self, text: str) -> np.ndarray:
        """Cluster direction plus a small deterministic within-cluster offset.
        An unknown phrase is pure offset, so it belongs nowhere on purpose."""
        rng = np.random.default_rng(_seed("sem", text))
        k = self._noise_basis.shape[1]
        off = self._noise_basis @ rng.standard_normal(k).astype(np.float32)
        off *= self.spread / (np.linalg.norm(off) or 1.0)
        name = self._of.get(text)
        v = off if name is None else self._axes[name] + off
        return v.astype(np.float32)

    def _surface(self, text: str) -> np.ndarray:
        """What low layers see: shared prefixes land near each other, which is
        the token-kinship grouping the real encoder shows at low layers.

        The key's `image:` prefix is stripped first. Without that every picture
        shares the same three leading characters and they all collapse onto one
        surface vector, which reads as a bogus low-layer peak for the whole
        modality. Picture tokens have no text prefix in common in the real
        encoder; the prefix here is only a cache key."""
        stem = text.split(":", 1)[1] if self.modality_of(text) == "image" else text
        rng = np.random.default_rng(_seed("surf", stem[:3].lower()))
        v = rng.standard_normal(self.dim).astype(np.float32)
        jitter = np.random.default_rng(_seed("surfj", text)).standard_normal(self.dim)
        return (v + 0.2 * jitter.astype(np.float32)).astype(np.float32)

    def modality_of(self, text: str) -> str:
        from .images import is_image_key

        return "image" if is_image_key(text) else "text"

    def peak_layer(self, modality: str = "text") -> int:
        """Where this modality's concept is strongest. The Z-Image sweep put
        text at 18 of 36, half the stack, and the krea2 profile charts at 20
        for the same reason. Pictures peak elsewhere, which is why a mixed
        universe has no single right chart layer."""
        return self.peaks[modality]

    def _mix(self, layer: int, modality: str = "text") -> float:
        """Semantic share at a layer: 0 at the input, 1 at that modality's
        peak, easing back down toward the top. Mirrors the measured U shape."""
        top = self.n_layers - 1
        peak = float(self.peaks[modality])
        span = max(peak, top - peak) or 1.0
        return float(max(0.0, 1.0 - abs(layer - peak) / span) ** 0.7)

    def embed_layer(self, text: str, layer: int) -> np.ndarray:
        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"layer {layer} out of range (0..{self.n_layers - 1})")
        m = self._mix(int(layer), self.modality_of(text))
        v = m * self._semantic(text) + (1.0 - m) * self._surface(text)
        # Magnitude grows with depth, like a residual stream.
        return (v * (1.0 + int(layer) / self.n_layers)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_layer(text, self._resolved_layer())

    def embed_many(self, texts, verbose: bool = False) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def embed_layers_many(self, texts, layer, verbose: bool = False) -> np.ndarray:
        if isinstance(layer, (tuple, list)):
            lo, hi = int(min(layer)), int(max(layer))
            return np.stack([
                np.mean([self.embed_layer(t, l) for l in range(lo, hi + 1)], axis=0)
                for t in texts])
        return np.stack([self.embed_layer(t, int(layer)) for t in texts])

    def _resolved_layer(self) -> int:
        return max(self._sampler)

    def sampler_layers(self) -> tuple[int, ...]:
        return self._sampler

    # --- conditioning ----------------------------------------------------

    def conditioning_tokens(self, text: str, layer: int | None = None) -> np.ndarray:
        """A short per-token sequence whose mean is exactly this layer's
        pooled vector, so pooled tests and token tests cannot disagree."""
        l = self._resolved_layer() if layer is None else int(layer)
        rng = np.random.default_rng(_seed("tok", text))
        seq = 3 + _seed("len", text) % 4
        toks = rng.standard_normal((seq, self.dim)).astype(np.float32)
        return (toks + (self.embed_layer(text, l) - toks.mean(axis=0))[None, :]
                ).astype(np.float32)

    def conditioning_layers(self, text: str, layers) -> dict[int, np.ndarray]:
        return {int(l): self.conditioning_tokens(text, int(l)) for l in layers}

    def conditioning(self, text: str) -> Conditioning:
        return Conditioning(self.conditioning_layers(text, list(self._sampler)))

    def cook_layers(self, hidden, start_index, pulls=None, stop_index=None,
                    capture=None, modality: str = "text", **kwargs):
        """Rotate a placed state toward the high-layer configuration, one pass
        at a time, easing toward `pulls` where given. Low inserter, many passes
        of rotation, so the blend is reinterpreted; high inserter, few passes,
        so it is transcribed.

        `modality` picks whose schedule the passes follow. A MIXED frame has
        no right answer here: one trajectory is applied to picture tokens and
        word tokens that peak at different layers. The real encoder has the
        same problem and nothing in the code addresses it yet."""
        stop = self.n_layers - 1 if stop_index is None else int(stop_index)
        want = set(int(c) for c in (capture or []))
        state = np.asarray(hidden, dtype=np.float32).copy()
        out = {}
        for l in range(int(start_index) + 1, stop + 1):
            step = self._mix(l, modality) - self._mix(l - 1, modality)
            state = state * (1.0 + step)          # the pass's own dynamics
            if pulls and l in pulls:
                alpha, target = pulls[l]
                state = (1.0 - alpha) * state + alpha * np.asarray(
                    target, dtype=np.float32)
            if l in want:
                out[l] = state.copy()
        if int(start_index) in want:
            out[int(start_index)] = np.asarray(hidden, dtype=np.float32).copy()
        return out
