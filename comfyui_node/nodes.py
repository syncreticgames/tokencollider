"""ComfyUI node: load a tokencollider safetensors conditioning export.

Loads the file written by tokencollider's POST /export (default format):
key "conditioning", shape (1, seq, dim), with blend weights / coords /
relative_error in the safetensors metadata. Outputs a standard CONDITIONING
plus the metadata as a string for wiring into notes or filename stamps.

Exports from a model whose sampler reads several encoder layers carry one
"layer_NN" key per layer instead, and list them in the "layers" metadata
field. ComfyUI's CONDITIONING holds a single tensor, and its own Krea 2 text
encoder (comfy/text_encoders/krea2.py) solves that by flattening the
(B, 12, seq, 2560) stack to (B, seq, 12*2560), layer-major, which the DiT
(comfy/ldm/krea2/model.py) reshapes back. With `layer` at -1 these nodes do
the same, so a Krea 2 export samples; `layer` >= 0 picks one layer for
inspection, which is NOT a shape the Krea 2 model accepts.
"""

import json

from safetensors import safe_open

SINGLE_KEY = "conditioning"
LAYER_PREFIX = "layer_"


def fuse_stack(tensors):
    """Flatten per-layer (1, seq, dim) tensors, in the order given, to the
    (1, seq, n*dim) layout ComfyUI's Krea 2 encoder hands the DiT: layer i
    owns features [i*dim, (i+1)*dim). Mirrors
    `out.permute(0, 2, 1, 3).reshape(b, seq, n * h)` on a (B, n, seq, h)
    stack, which comfy/ldm/krea2/model.py undoes with
    `context.reshape(b, seq, txtlayers, txtdim)`."""
    import torch

    stacked = torch.stack(tensors, dim=1)  # (1, n, seq, dim)
    b, n, seq, h = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(b, seq, n * h)


def read_conditioning(path, layer=-1):
    """One (1, seq, features) tensor, the file's metadata, and the layers it
    holds.

    A single-layer export stores key "conditioning" and reports no layers.
    A multi-layer export stores "layer_NN" per layer. `layer` < 0 fuses every
    layer, ascending, into (1, seq, n*dim), the shape a multi-layer sampler
    (Krea 2) reads; `layer` >= 0 picks that one layer at (1, seq, dim)."""
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        if SINGLE_KEY in keys:
            tensor = f.get_tensor(SINGLE_KEY)
            depths = []
        else:
            depths = sorted(int(k[len(LAYER_PREFIX):]) for k in keys
                            if k.startswith(LAYER_PREFIX))
            if not depths:
                raise ValueError(
                    f"{path}: no conditioning tensors (keys {sorted(keys)})")
            if int(layer) < 0:
                tensors = [f.get_tensor(f"{LAYER_PREFIX}{d:02d}") for d in depths]
                tensors = [t[None, :, :] if t.dim() == 2 else t for t in tensors]
                tensor = fuse_stack(tensors)
            else:
                pick = int(layer)
                if pick not in depths:
                    raise ValueError(
                        f"{path}: no layer {pick} — this export carries {depths}")
                tensor = f.get_tensor(f"{LAYER_PREFIX}{pick:02d}")
    if tensor.dim() == 2:  # tolerate un-batched (seq, dim) files
        tensor = tensor[None, :, :]
    return tensor, metadata, depths


class LoadConditioningSafetensors:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                # Strength for ConcatConditioning workflows. Z-Image's DiT
                # RMSNorms every context token (cap_embedder), so scalar
                # multiplies are cancelled — attention influence scales with
                # the NUMBER of tokens instead. repeat tiles the export's
                # tokens along the sequence. (Krea 2's text-fusion blocks are
                # residual, so a multiply is NOT cancelled there; repeat
                # still works, but it is not the only knob. Unmeasured.)
                "repeat": ("INT", {"default": 1, "min": 1, "max": 64}),
                # -1 = fuse every layer of a multi-layer export into the
                # (1, seq, n*dim) stack Krea 2 samples from; N = that one
                # layer alone, for inspection. Ignored by single-layer files.
                "layer": ("INT", {"default": -1, "min": -1, "max": 128}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "metadata")
    FUNCTION = "load"
    CATEGORY = "tokencollider"

    @classmethod
    def IS_CHANGED(cls, path, repeat, layer=-1):
        # Re-run when the file content changes, not just the path string.
        import hashlib
        try:
            with open(path, "rb") as f:
                return (hashlib.sha256(f.read()).hexdigest()
                        + str(repeat) + str(layer))
        except OSError:
            return ""

    def load(self, path, repeat=1, layer=-1):
        tensor, metadata, depths = read_conditioning(path, layer)
        if repeat > 1:
            tensor = tensor.repeat(1, repeat, 1)
        if depths:
            metadata = {**metadata, "loaded_layers": depths,
                        "loaded_layer": "fused" if int(layer) < 0 else int(layer),
                        "loaded_shape": list(tensor.shape)}
        return ([[tensor, {}]], json.dumps(metadata, indent=2))


class ConditioningInfo:
    """Pass-through inspector: reports each conditioning entry's token count,
    effective (non-padding) token count, and dim. On models that RMSNorm
    their context (Z-Image), relative influence in a Concat tracks the ratio
    of effective token counts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT")
    RETURN_NAMES = ("conditioning", "info", "effective_tokens")
    FUNCTION = "inspect"
    CATEGORY = "tokencollider"
    OUTPUT_NODE = True

    def inspect(self, conditioning):
        lines = []
        total_effective = 0
        for i, (tensor, extras) in enumerate(conditioning):
            seq = tensor.shape[-2]
            # Zero-padded rows normalize to null keys and pull no attention;
            # only count rows that actually participate.
            norms = tensor.float().norm(dim=-1).reshape(-1)
            effective = int((norms > 1e-6).sum())
            total_effective += effective
            desc = f"entry {i}: {seq} tokens ({effective} effective), dim {tensor.shape[-1]}"
            if extras:
                desc += f", extras: {sorted(extras.keys())}"
            lines.append(desc)
        info = "\n".join(lines)
        return {"ui": {"text": [info]}, "result": (conditioning, info, total_effective)}


class LoadConditioningStack:
    """An inserter stack as ONE schedule-carrying conditioning.

    `pattern` is a glob over the stack's shared stem (the extension is
    optional) or a directory holding exactly one stack. Inserters parse from the `_cookNN` filename
    suffix; the deepest cook (lowest NN, the encoder's furthest paraphrase)
    is gated onto the earliest sigmas, where composition is decided, and the
    shallowest onto the latest. Each entry carries its own start/end percent.
    Mirror (negative) exports are never part of a stack.

    `overlap` widens each window on both sides; `repeat` tiles every entry's
    tokens.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": ("STRING", {"default": "", "multiline": False}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 64}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.45,
                                      "step": 0.05}),
                "layer": ("INT", {"default": -1, "min": -1, "max": 128}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "load"
    CATEGORY = "tokencollider"

    @staticmethod
    def _matches(pattern):
        import glob
        import os
        import re

        # A directory means every cook file in it; a glob is taken as is,
        # with or without the extension (the inserter is read from the
        # `_cookNN.` in each filename either way).
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*_cook*.safetensors")
        # Every name the exporter writes for a cook: `_cookNN`, then an
        # optional `_mirror`, an optional 6-hex digest (added when the name
        # was taken), and an optional `.mirror` (a named export's mirror).
        # Mirrors are the negative half of a polarity pair and stay out.
        cook = re.compile(
            r"_cook(\d+)(_mirror)?(?:_[0-9a-f]{6})?(\.mirror)?\.safetensors$")
        groups = {}
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            m = cook.search(name)
            if m and not (m.group(2) or m.group(3)):
                groups.setdefault(name[:m.start()], []).append(
                    (int(m.group(1)), path))
        # One stack per load. Everything before `_cookNN` names a stack (the
        # exporter tags each stack `_stack<id>`), so files from different
        # cursors or exports never get sliced into one schedule.
        if len(groups) > 1:
            shown = ", ".join(sorted(groups)[:5])
            more = f" and {len(groups) - 5} more" if len(groups) > 5 else ""
            raise ValueError(
                f"{pattern!r} holds {len(groups)} different stacks ({shown}{more}). "
                f"Point at one with a glob like <folder>/<name>_cook*.")
        out = next(iter(groups.values()), [])
        return sorted(out)  # ascending cook NN = deepest first

    @classmethod
    def IS_CHANGED(cls, pattern, repeat, overlap, layer=-1):
        import hashlib
        h = hashlib.sha256()
        try:
            matches = cls._matches(pattern)
        except ValueError:
            return float("nan")  # always re-run, so load() reports the problem
        for _depth, path in matches:
            try:
                with open(path, "rb") as f:
                    h.update(f.read())
            except OSError:
                pass
        return h.hexdigest() + str(repeat) + str(overlap) + str(layer)

    def load(self, pattern, repeat=1, overlap=0.0, layer=-1):
        matches = self._matches(pattern)
        if not matches:
            raise ValueError(
                f"no *_cookNN.safetensors files match {pattern!r}")
        n = len(matches)
        conditioning = []
        lines = []
        for i, (depth, path) in enumerate(matches):
            tensor, _meta, _depths = read_conditioning(path, layer)
            if repeat > 1:
                tensor = tensor.repeat(1, repeat, 1)
            start = max(0.0, i / n - overlap)
            end = min(1.0, (i + 1) / n + overlap)
            conditioning.append([tensor, {"start_percent": start,
                                          "end_percent": end}])
            lines.append(f"cook{depth:02d}: sigma {start:.2f}-{end:.2f}  "
                         f"({tensor.shape[-2]} tokens)")
        return (conditioning, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "LoadConditioningSafetensors": LoadConditioningSafetensors,
    "ConditioningInfo": ConditioningInfo,
    "LoadConditioningStack": LoadConditioningStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadConditioningSafetensors": "Load Conditioning (safetensors)",
    "ConditioningInfo": "Conditioning Info",
    "LoadConditioningStack": "Load Conditioning Stack (cook depths)",
}
