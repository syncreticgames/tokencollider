"""A conditioning: per-token hidden states at the layers a model actually reads.

Z-Image reads one layer (raw hidden state 35, its layer_idx=-2). Krea 2 reads
twelve and mixes them with a learned aggregator inside the DiT. Both are the
same object here, an ordered map from hidden-state index to a (seq, dim)
array, so a single-layer model is a Conditioning of size one instead of a
separate code path.

The file format is a contract with other programs: docs/export-format.md.
This module is its canonical reader and writer, and the doc wins if the two
disagree.
"""

import json

import numpy as np

SINGLE_KEY = "conditioning"
LAYER_PREFIX = "layer_"


def layer_key(index: int) -> str:
    return f"{LAYER_PREFIX}{int(index):02d}"


def _infer_depth(spec) -> int | None:
    """Hidden-state index named by a stored layer field: "35" -> 35,
    "18-34" -> 34 (a band's sampler layer is its top edge), "last" -> None."""
    if spec is None:
        return None
    try:
        return int(str(spec).split("-")[-1])
    except ValueError:
        return None


def blend_tokens(tensors: list[np.ndarray], weights,
                 preserve_norm: bool = False) -> np.ndarray:
    """Weighted sum of per-token tensors, zero-padded to the longest.
    Landmarks are short phrases, so lengths stay close and the padding is
    a small, honest approximation.

    `preserve_norm` rescales each blended TOKEN back to the weighted mean of
    the norms that went into it. A weighted sum of vectors that do not point
    the same way is shorter than its inputs: the sum cuts through the interior
    of the sphere the real states live on, and a short vector there is
    improbable under the model's own distribution. The interpolation
    literature calls this norm collapse and answers it with slerp, which does
    not generalise cleanly to N landmarks with signed least-squares weights;
    rescaling to the weighted mean norm does, keeps the direction the solve
    chose, and restores only the magnitude.

    Off by default: it changes every export, and the claim that it helps is
    measured per model, not assumed.
    """
    seq_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]
    out = np.zeros((seq_len, dim), dtype=np.float32)
    for tensor, weight in zip(tensors, weights):
        out[: tensor.shape[0]] += weight * tensor
    if not preserve_norm:
        return out
    # Target norm per token: the weighted mean of the contributing norms.
    # |w| because a negative weight still contributes magnitude.
    num = np.zeros(seq_len, dtype=np.float64)
    den = np.zeros(seq_len, dtype=np.float64)
    for tensor, weight in zip(tensors, weights):
        n = tensor.shape[0]
        num[:n] += abs(float(weight)) * np.linalg.norm(tensor, axis=1)
        den[:n] += abs(float(weight))
    target = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    have = np.linalg.norm(out, axis=1)
    scale = np.divide(target, have, out=np.ones_like(target), where=have > 1e-8)
    return (out * scale[:, None]).astype(np.float32)


class Conditioning:
    """Per-token hidden states at one or more layers.

    Construct from {hidden_state_index: (seq, dim) array}. Layers are kept
    sorted so iteration order and file layout are deterministic.
    """

    def __init__(self, tensors: dict[int, np.ndarray]):
        if not tensors:
            raise ValueError("a conditioning needs at least one layer")
        self._t: dict[int, np.ndarray] = {}
        for index in sorted(int(i) for i in tensors):
            arr = np.ascontiguousarray(tensors[index], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:  # tolerate (1, seq, dim)
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(
                    f"layer {index}: expected (seq, dim), got shape {arr.shape}"
                )
            self._t[index] = arr
        dims = {a.shape[1] for a in self._t.values()}
        if len(dims) > 1:
            raise ValueError(f"layers disagree on width: {sorted(dims)}")

    @classmethod
    def single(cls, tensor: np.ndarray, layer: int) -> "Conditioning":
        return cls({int(layer): tensor})

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(self._t)

    @property
    def tensor(self) -> np.ndarray:
        """The sole per-token tensor. Only meaningful for a size-one stack;
        anything else has to say which layer it means."""
        if len(self._t) != 1:
            raise ValueError(
                f"conditioning spans {len(self._t)} layers {self.layers} — "
                "index it by layer instead of asking for 'the' tensor"
            )
        return next(iter(self._t.values()))

    def __getitem__(self, layer: int) -> np.ndarray:
        return self._t[int(layer)]

    def __len__(self) -> int:
        return len(self._t)

    def __contains__(self, layer) -> bool:
        return int(layer) in self._t

    def items(self):
        return self._t.items()

    @property
    def seq_len(self) -> int:
        return max(a.shape[0] for a in self._t.values())

    @property
    def dim(self) -> int:
        return next(iter(self._t.values())).shape[1]

    @classmethod
    def blend(cls, conds: list["Conditioning"], weights) -> "Conditioning":
        """Weighted sum, layer by layer. Every input must span the same
        layers: a blend across mismatched stacks has no honest meaning."""
        if not conds:
            raise ValueError("nothing to blend")
        layers = conds[0].layers
        for c in conds[1:]:
            if c.layers != layers:
                raise ValueError(
                    f"cannot blend conditionings over different depths: "
                    f"{layers} vs {c.layers}"
                )
        return cls({l: blend_tokens([c[l] for c in conds], weights)
                    for l in layers})

    @classmethod
    def concat(cls, conds: list["Conditioning"]) -> "Conditioning":
        """Join along the token axis, layer by layer — the training-cache
        recipe (caption tokens first, flown anchors after)."""
        if not conds:
            raise ValueError("nothing to concatenate")
        layers = conds[0].layers
        for c in conds[1:]:
            if c.layers != layers:
                raise ValueError(
                    f"cannot concatenate conditionings over different depths: "
                    f"{layers} vs {c.layers}"
                )
        return cls({l: np.concatenate([c[l] for c in conds], axis=0)
                    for l in layers})

    def to_tensors(self) -> dict[str, np.ndarray]:
        """Safetensors payload, batched to (1, seq, dim) as samplers expect.
        A size-one stack keeps the historical single key."""
        if len(self._t) == 1:
            return {SINGLE_KEY: self.tensor[None, :, :]}
        return {layer_key(l): a[None, :, :] for l, a in self._t.items()}

    def save(self, path: str, metadata: dict | None = None, framework: str = "pt") -> None:
        meta = dict(metadata or {})
        if len(self._t) > 1:
            # Only multi-layer files carry the layer list, so single-layer
            # exports stay byte-for-byte what they have always been.
            meta["layers"] = json.dumps(list(self.layers))
        tensors = self.to_tensors()
        if framework == "pt":
            import torch
            from safetensors.torch import save_file

            save_file({k: torch.from_numpy(v) for k, v in tensors.items()},
                      path, metadata=meta)
        else:
            from safetensors.numpy import save_file

            save_file(tensors, path, metadata=meta)

    @classmethod
    def load(cls, path: str) -> tuple["Conditioning", dict]:
        """Read a conditioning file written by this tool. Returns the stack
        and the file's metadata. Single-key files load as a size-one stack at
        the layer named in metadata ("layer"), falling back to 0 when the file
        predates that field — the index is a label there, not geometry."""
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as f:
            keys = list(f.keys())
            meta = dict(f.metadata() or {})
            if SINGLE_KEY in keys:
                # "layer" is the embedder's sampler config, which is the layer
                # the tensor was assembled at. "view_layer" is the chart the
                # blend was solved in and can differ (a cooked band export is
                # solved at 4-8 but assembled at the sampler layer), so it is
                # only a fallback.
                index = _infer_depth(meta.get("layer"))
                if index is None:
                    index = _infer_depth(meta.get("view_layer"))
                return cls.single(f.get_tensor(SINGLE_KEY),
                                  0 if index is None else index), meta
            indices = [int(k[len(LAYER_PREFIX):]) for k in keys
                       if k.startswith(LAYER_PREFIX)]
            if not indices:
                raise ValueError(
                    f"{path} has no conditioning tensors (keys: {sorted(keys)})"
                )
            return cls({i: f.get_tensor(layer_key(i)) for i in indices}), meta

    def trim(self, start: int) -> "Conditioning":
        """Drop the first `start` tokens at every layer.

        Used to match a sampler's token frame: Krea 2's encoder hands the DiT
        everything from the prompt onward, dropping the scaffold that precedes
        it. Cooking must happen before this, on the full sequence, because the
        model's attention depends on the context being present."""
        start = int(start)
        if start <= 0:
            return self
        for index, arr in self._t.items():
            if start >= arr.shape[0]:
                raise ValueError(
                    f"layer {index}: cannot trim {start} tokens from a "
                    f"{arr.shape[0]}-token sequence"
                )
        return Conditioning({i: a[start:] for i, a in self._t.items()})

    def relabel(self, layers) -> "Conditioning":
        """Re-key the same tensors to different layers.

        Files written before conditionings recorded their layer carry no
        reliable index, and an anchor flown from the same config sits at the
        caller's layers by construction. This states that, rather than letting
        a guess from metadata turn into a mismatched-layers error."""
        layers = tuple(int(l) for l in layers)
        if len(layers) != len(self._t):
            raise ValueError(
                f"cannot relabel a {len(self._t)}-layer conditioning as "
                f"{len(layers)} layers"
            )
        return Conditioning(dict(zip(layers, self._t.values())))

    def compare(self, other: "Conditioning", tail: int = 0) -> dict:
        """How two exports differ, per token and per layer.

        A global cosine over the flattened tensor is the wrong instrument for
        this. A Krea 2 export is a couple of phrase tokens followed by a
        five-token scaffold tail that barely moves, so the scaffold is most of
        the tensor and averaging over it drowns the phrase. Measured 08232026:
        two exports whose global cosine was 1.000 differed by a 7.5% norm drop
        on a single phrase token, and that difference flipped the render.

        `tail` is the number of trailing scaffold tokens to hold out of the
        phrase summary (`Embedder.template_tail_tokens()`).
        """
        if sorted(self.layers) != sorted(other.layers):
            raise ValueError(f"different layers: {self.layers} vs {other.layers}")
        n = min(self.seq_len, other.seq_len)
        phrase = slice(0, max(0, n - tail))
        out = {"tokens": n, "tail": tail, "per_token": {}, "per_layer": {}}
        worst = (1.0, None, None)
        for l in sorted(self.layers):
            a, b = self[l][:n], other[l][:n]
            cs, nr = [], []
            for t in range(n):
                x, y = a[t].astype(np.float64), b[t].astype(np.float64)
                den = np.linalg.norm(x) * np.linalg.norm(y)
                c = float(x @ y / den) if den else 1.0
                cs.append(c)
                nr.append(float(np.linalg.norm(y) / (np.linalg.norm(x) or 1e-9)))
                if t < phrase.stop and c < worst[0]:
                    worst = (c, l, t)
            out["per_token"][l] = {"cosine": cs, "norm_ratio": nr}
            ph = cs[phrase] or [1.0]
            out["per_layer"][l] = {"phrase_min_cosine": float(min(ph)),
                                   "phrase_mean_cosine": float(np.mean(ph))}
        out["worst_phrase_token"] = {"cosine": worst[0], "layer": worst[1],
                                     "token": worst[2]}
        return out

    def __repr__(self) -> str:
        return (f"Conditioning(layers={list(self.layers)}, "
                f"seq={self.seq_len}, dim={self.dim})")
