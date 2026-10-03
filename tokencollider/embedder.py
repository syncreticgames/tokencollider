"""Embed phrases through a language-model text encoder (any decoder
transformers can build; the built-in profiles use Qwen3-4B and Qwen3-VL-4B's
text tower) into pooled vectors and per-token conditionings.

The model is loaded lazily on first embed; cached phrases never touch the model,
so `rank`/`compare` over a warm cache are instant and model-free.
"""

import os
from pathlib import Path

import numpy as np

from . import images
from .conditioning import Conditioning
from .store import EmbeddingStore

# Hard offline: this app must never touch HF Hub or any online service.
# Set before transformers/huggingface_hub are imported (all imports below are
# lazy, so module import order guarantees this runs first).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# No module-level default weights path. Which checkpoint to load is a
# per-profile setting with no sensible global fallback (see tokencollider.profiles).
# Phrases embed inside this context, pooled ONLY over the phrase's own tokens.
# Bare single-token phrases have no context to compute over — their lone hidden
# state is position-dominated, not semantic ('Link' vs 'Table' raw cosine 0.99).
# Profiles carry the real per-model templates; this is only the fallback for
# direct construction in tests and scripts.
DEFAULT_TEMPLATE = "an image of {}"
# Free VRAM a load needs beyond its own bf16 weights: activations, hidden
# states for every layer of a bulk batch, the CUDA context. Below the total,
# fail rather than thrash.
VRAM_HEADROOM_BYTES = 2 * 1024**3
# ComfyUI-style fp8 checkpoints store a quantized matrix as three tensors:
# the weight itself (float8_e4m3fn), a scalar weight_scale (f32), and
# comfy_quant, a 64-byte JSON descriptor. The real weight is
# weight * weight_scale. Casting the fp8 bytes without applying the scale is
# off by roughly three orders of magnitude and yields plausible-looking
# garbage rather than an error, so this is not optional.
SCALE_SUFFIX = ".weight_scale"
QUANT_SUFFIX = ".comfy_quant"
# Phrases per batched forward pass during bulk warms. Short templated phrases
# make activations cheap; the win is amortizing per-call and transfer overhead.
BATCH_SIZE = int(os.environ.get("TOKENCOLLIDER_BATCH", "32"))


def strip_tower_prefix(key: str) -> str:
    """Text-tower key as the model class wants it, unprefixed.

    ComfyUI-style encoder files flatten the text tower to 'model.layers...',
    while a stock HF VLM checkpoint nests it at 'model.language_model.layers...'.
    Both mean the same parameter, and the text model classes want neither
    prefix, so strip whichever is present."""
    for prefix in ("model.language_model.", "language_model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


# The config fields of the Qwen-VL image-token protocol: pad, start, end.
IMAGE_TOKEN_IDS = ("image_token_id", "vision_start_token_id", "vision_end_token_id")

# Where vision-language checkpoints keep their vision tower and projector,
# across the layouts transformers and ComfyUI use. Dropped when only the text
# tower is built.
VISION_PREFIXES = ("model.visual.", "visual.", "model.vision_tower.",
                   "vision_tower.", "model.multi_modal_projector.",
                   "multi_modal_projector.")


def vl_full_key(key: str) -> str | None:
    """A checkpoint key as transformers' Qwen3VLModel (vision + text) wants
    it, or None for keys that model has no place for (lm_head).

    ComfyUI's encoder file flattens to `model.layers.*` / `model.visual.*`;
    an HF checkpoint nests `model.language_model.*` / `model.visual.*`. The
    full model wants `language_model.*` / `visual.*`."""
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            return "language_model." + key[len(prefix):]
    for prefix in ("model.visual.", "visual."):
        if key.startswith(prefix):
            return "visual." + key[len(prefix):]
    if key.startswith("lm_head") or key.startswith("model.lm_head"):
        return None
    if key.startswith("model."):
        return "language_model." + key[len("model."):]
    return key


def parse_layer_spec(spec):
    """The one grammar for a layer setting: 'last' | '35' | '18-34'.

    Returns "last", an int layer, or an inclusive (lo, hi) band. Three call
    sites used to re-implement this (the embedder's band check, its resolved
    sampler layer, and the layer stack's primary index), which is how a band
    config once silently lost the ability to cook."""
    if isinstance(spec, tuple):
        lo, hi = sorted(int(v) for v in spec)
        return lo if lo == hi else (lo, hi)
    if isinstance(spec, int):
        return spec
    spec = str(spec)
    if spec == "last":
        return "last"
    if "-" in spec:
        lo, _, hi = spec.partition("-")
        lo, hi = int(lo), int(hi)
        if lo > hi:
            raise ValueError(f"band {spec!r}: need lo <= hi")
        return lo if lo == hi else (lo, hi)
    return int(spec)


def canonical_layer(spec) -> str:
    """A layer setting as the one string every cache key and header uses:
    'last', '20' or '18-34'. A degenerate band '20-20' is layer 20."""
    parsed = parse_layer_spec(spec)
    if isinstance(parsed, tuple):
        return f"{parsed[0]}-{parsed[1]}"
    return str(parsed)


class _Progress:
    """The bulk-warm progress line, in one place. Prints rate and ETA, keeps
    a single live line on a tty and periodic lines in logs."""

    def __init__(self, label: str, total: int, verbose: bool):
        import sys
        import time

        self.label = label
        self.total = total
        self.verbose = verbose and total > 0
        self.done_count = 0
        self.t0 = time.monotonic()
        self.tty = sys.stdout.isatty()

    def step(self, n: int, last_text: str | None = None) -> None:
        import time

        self.done_count += n
        if not self.verbose:
            return
        elapsed = time.monotonic() - self.t0
        rate = self.done_count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done_count) / rate if rate > 0 else 0.0
        msg = (f"[tokencollider] {self.label} {self.done_count}/{self.total}  {rate:.1f}/s  "
               f"eta {int(eta // 60)}m{int(eta % 60):02d}s")
        if last_text:
            msg += f"  {last_text[:40]!r}"
        last = self.done_count >= self.total
        print(f"{msg:<100}", end="\n" if last or not self.tty else "\r", flush=True)


def require_cuda(need_bytes: int) -> str:
    """GPU-only by design. Fail loudly and early, never fall back to CPU.
    `need_bytes` is what this model's load needs (Embedder.vram_needed)."""
    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        free, _total = torch.cuda.mem_get_info()
    except Exception as e:
        raise SystemExit(f"[tokencollider] no usable GPU ({e}): is a "
                         "training or inference job holding it?")
    if free < need_bytes:
        raise SystemExit(
            f"[tokencollider] only {free / 1024**3:.1f}GB VRAM free, this model needs "
            f"~{need_bytes / 1024**3:.1f}GB. Free the GPU and retry."
        )
    return "cuda"


class Embedder:
    def __init__(
        self,
        store: EmbeddingStore,
        model_name: str,
        layer: str = "last",
        pooling: str = "mean",
        device: str | None = None,
        template: str = DEFAULT_TEMPLATE,
        config_dir: str | None = None,
        light: bool = False,
        sampler_layers=None,
        config_repo: str | None = None,
        trim_template_prefix: bool = False,
        image_pooling: str = "image",
    ):
        self.store = store
        # How an image landmark is pooled: over its image tokens, or over the
        # scaffold tail that attended to it. Part of the landmark key, so
        # the two never share cache rows. See tokencollider.images.
        if image_pooling not in images.POOLINGS:
            raise ValueError(f"image_pooling must be one of {images.POOLINGS}")
        self.image_pooling = image_pooling
        # Whether the loaded model carries the vision tower. Loaded lazily
        # on the first image, replacing the text-only model.
        self._has_vision = False
        self._image_processor = None
        # Light mode: warm stores only the sampler-layer pooled vector — no
        # per-layer scrub cache, no conditioning tensor. ~50x smaller on disk,
        # sized for wordlist-scale universes (20k phrases ~ 200MB vs ~10GB).
        self.light = light
        self.model_name = model_name
        self.config_dir = config_dir
        # Where config + tokenizer come from when no config_dir is set.
        # Resolved against the local HF cache only; nothing is downloaded.
        self.config_repo = config_repo
        self.layer = canonical_layer(layer)
        self.pooling = pooling
        self.device = device
        self._auto_device = device is None
        self.template = template
        # Layers the sampler reads. None means the single layer `layer`
        # resolves to; a model whose DiT aggregates several encoder layers
        # (Krea 2 reads twelve) passes them explicitly. See tokencollider.profiles.
        self._sampler_layers = (
            None if sampler_layers is None
            else tuple(sorted(int(l) for l in sampler_layers))
        )
        # Whether an export drops the scaffold preceding the prompt, matching
        # what the sampler is actually handed. Per-model; see tokencollider.profiles.
        self.trim_template_prefix = trim_template_prefix
        if template.count("{}") != 1:
            raise ValueError("template must contain exactly one {}")
        self._prefix, _, self._suffix = template.partition("{}")
        self._model = None
        self._tokenizer = None
        if self._band() is not None and pooling != "mean":
            raise ValueError("band layers only support mean pooling")
        # Number of hidden-state entries (n_blocks + 1; index 0 is the
        # embedding-table output). Unknown until the first forward pass or
        # cache probe.
        self.n_layers: int | None = None

    def weight_files(self) -> list[Path]:
        """The safetensors holding the weights: a single ComfyUI-style file, or
        every shard in an HF-style directory.

        When the directory carries a `model.safetensors.index.json`, that names
        the shards authoritatively and is used. Globbing instead would sweep up
        anything else parked in the folder (a consolidated copy alongside the
        shards, a stray adapter) and turn it into unexpected keys."""
        import json

        path = Path(self.model_name)
        if not path.is_dir():
            if not path.exists():
                raise SystemExit(f"[tokencollider] no such model weights: {path}")
            return [path]
        index = path / "model.safetensors.index.json"
        if index.exists():
            names = sorted(set(
                json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()))
            shards = [path / name for name in names]
            missing = [str(s) for s in shards if not s.exists()]
            if missing:
                raise SystemExit(
                    f"[tokencollider] {index.name} names shards that are not here: "
                    f"{', '.join(missing)}"
                )
            if shards:
                return shards
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise SystemExit(
                f"[tokencollider] no .safetensors in {path} — point the model setting at "
                "a weights file or a directory of shards"
            )
        return shards

    def _is_safetensors(self) -> bool:
        path = Path(self.model_name)
        return path.suffix == ".safetensors" or (
            path.is_dir() and any(path.glob("*.safetensors")))

    def config_source(self) -> str:
        """Where config + tokenizer come from. Never the network."""
        if self.config_dir is not None:
            return self.config_dir
        model = Path(self.model_name)
        if model.is_dir() and (model / "config.json").exists():
            return self.model_name  # an HF directory carries its own
        if self._is_safetensors():
            if self.config_repo is None:
                raise SystemExit(
                    f"[tokencollider] {self.model_name} carries no config or tokenizer, "
                    "and the profile names no config_dir or config_repo to "
                    "take them from.")
            return self.config_repo  # local HF cache only; offline forced above
        return self.model_name

    def _from_pretrained(self, cls):
        if self.config_dir is not None and not Path(self.config_dir).is_dir():
            raise SystemExit(f"[tokencollider] config_dir {self.config_dir!r} is not a directory")
        try:
            return cls.from_pretrained(self.config_source(), local_files_only=True)
        except OSError as e:
            raise SystemExit(
                f"[tokencollider] cannot load {cls.__name__} from {self.config_source()!r} "
                f"offline ({e}).\n"
                "This tool never downloads. Set the profile's config_dir to a local "
                "directory containing config.json, tokenizer.json, and "
                "tokenizer_config.json for the model."
            )

    def _language_model(self):
        """The decoder stack, whichever wrapper it sits in: the text-only
        model is the stack itself; the full VL model nests it."""
        return getattr(self._model, "language_model", self._model)

    @staticmethod
    def _text_tower(config):
        """The class to build, the config to build it from, and the key
        prefixes to drop.

        Whatever transformers registers for the config's model_type: the
        encoder is not hard-wired to one family. A vision-language checkpoint
        contributes its text tower, built from `text_config`, and its vision
        keys are dropped deliberately, so the strict state-dict check still
        catches a genuinely wrong file instead of being loosened to tolerate
        them."""
        from transformers import MODEL_MAPPING

        text_config = getattr(config, "text_config", None)
        vlm = (text_config is not None
               and getattr(config, "vision_config", None) is not None)
        build = text_config if vlm else config
        try:
            model_cls = MODEL_MAPPING[type(build)]
        except KeyError:
            raise SystemExit(
                f"[tokencollider] transformers has no model class for model_type "
                f"{getattr(build, 'model_type', '?')!r}")
        return model_cls, build, VISION_PREFIXES if vlm else ()

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = self._from_pretrained(AutoTokenizer)

    def _span(self, text: str) -> tuple[int, int]:
        """Token index range of the phrase inside its templated string.

        Uses character offsets, not token-count arithmetic: BPE merges across
        the prefix/phrase boundary (' of ' + 'Mario' -> ' of', ' Mario'), so
        counting prefix tokens alone under- or over-shoots.
        """
        self._ensure_tokenizer()
        if not text:
            # An empty phrase (a caption-less training image) has no tokens
            # of its own: the conditioning is the scaffold alone, which is
            # what the sampler's own encoder would produce for it, and the
            # pooled vector does not exist.
            n = self.template_prefix_tokens()
            return n, n
        enc = self._tokenizer(
            self._prefix + text + self._suffix,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        lo, hi = len(self._prefix), len(self._prefix) + len(text)
        idx = [
            i for i, (a, b) in enumerate(enc["offset_mapping"])
            if a < hi and b > lo
        ]
        if not idx:
            raise ValueError(f"empty token span for {text!r} in template {self.template!r}")
        return idx[0], idx[-1] + 1

    def template_prefix_tokens(self) -> int:
        """Tokens the template contributes before the phrase.

        This is ComfyUI's `template_end` for the Krea 2 encoder, reached by a
        shorter route: it locates the <|im_start|> opening the user turn and
        adds 3, which is precisely the length of the prefix. Stable across
        phrases because a special token precedes the slot, so no BPE merge
        crosses the boundary (verified over a spread of phrases, including
        empty-ish and non-ASCII ones).
        """
        self._ensure_tokenizer()
        return len(self._tokenizer(self._prefix, add_special_tokens=False)["input_ids"])

    def template_tail_tokens(self) -> int:
        """Tokens the template contributes after the phrase (the trailing
        turn, 5 for both shipped scaffolds)."""
        self._ensure_tokenizer()
        return len(self._tokenizer(self._suffix, add_special_tokens=False)["input_ids"])

    def mixed_position_ids(self, segments):
        """Krea 2's 3-D rotary positions for a frame assembled from several
        modalities: `segments` is a list of ("text", n) and ("image", n)
        runs, where an image run is `<|vision_start|>`, the patch tokens,
        `<|vision_end|>`. Text tokens count up; an image block takes the
        (t, h, w) grid positions the real forward would give it. None when
        no image run is present (a flat range is then exactly right)."""
        import torch

        if not any(kind == "image" for kind, _ in segments):
            return None
        self._ensure_vision()
        self._ensure_image_processor()
        pad, start, end = (self._image_ids[k] for k in ("pad", "start", "end"))
        side = images.IMAGE_SIZE // self._image_processor.patch_size
        n_patch = (side // self._image_processor.merge_size) ** 2
        ids, grids = [], []
        for kind, n in segments:
            if kind == "image":
                if n != n_patch + 2:
                    raise ValueError(
                        f"an image run is {n_patch + 2} tokens on this grid, not {n}")
                ids += [start] + [pad] * n_patch + [end]
                grids.append([1, side, side])
            else:
                ids += [0] * n
        input_ids = torch.tensor([ids], device=self.device)
        grid = torch.tensor(grids, device=self.device)
        attn = torch.ones_like(input_ids)
        model = self._model
        try:
            position_ids, _ = model.get_rope_index(
                input_ids, grid, None, attn)
        except TypeError:
            position_ids = model.compute_3d_position_ids(
                input_ids=input_ids, inputs_embeds=None, image_grid_thw=grid,
                attention_mask=attn, mm_token_type_ids=(input_ids == pad).int())
        return position_ids  # (3, 1, S)

    def export_trim(self) -> int:
        """Tokens an export should drop from the front, 0 when this model's
        sampler is handed the whole templated sequence."""
        return self.template_prefix_tokens() if self.trim_template_prefix else 0

    def _load_model(self, vision: bool = False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = require_cuda(self.vram_needed(vision))
        dtype = torch.bfloat16
        print(f"[tokencollider] loading {self.model_name} on {self.device} ({dtype})"
              f"{' with the vision tower' if vision else ''} ...")
        self._ensure_tokenizer()
        if self._is_safetensors():
            self._model = self._load_local_safetensors(dtype, vision=vision)
        else:
            self._model = AutoModel.from_pretrained(
                self.model_name, dtype=dtype, local_files_only=True
            )
            self._model.to(self.device)
            vision = hasattr(self._model, "visual")
        self._model.eval()
        self._has_vision = vision

    def _ensure_vision(self):
        """An image needs the vision tower. The text-only model is replaced
        by the full one (about 1 GB more VRAM); text results are unchanged,
        since the decoder stack is the same weights."""
        if self._model is not None and self._has_vision:
            return
        if self._model is not None:
            print("[tokencollider] reloading with the vision tower for image landmarks ...")
            self.release()
        self._load_model(vision=True)

    def vram_needed(self, vision: bool = False) -> int:
        """Bytes of free VRAM a load needs: the weights that will actually
        load, at the bf16 they run in whatever their stored precision (fp8
        doubles, fp32 halves), plus VRAM_HEADROOM_BYTES. Read from the
        safetensors headers, so it scales with the model instead of
        assuming one size. Other formats fall back to their size on disk."""
        import json
        import math
        import struct

        if not self._is_safetensors():
            on_disk = sum(f.stat().st_size for f in Path(self.model_name).rglob("*")
                          if f.is_file())
            return on_disk + VRAM_HEADROOM_BYTES
        elements = 0
        for path in self.weight_files():
            with open(path, "rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(length))
            header.pop("__metadata__", None)
            for key, info in header.items():
                if key.endswith((SCALE_SUFFIX, QUANT_SUFFIX)):
                    continue  # folded into their weight on load
                if not vision and key.startswith(VISION_PREFIXES):
                    continue  # the text tower alone leaves these behind
                if "lm_head" in key:
                    continue
                elements += math.prod(info["shape"])
        return elements * 2 + VRAM_HEADROOM_BYTES

    def _peek_keys(self) -> list[str]:
        """Tensor names from the safetensors headers alone, no weight data.
        Unions every shard."""
        import json
        import struct

        keys: list[str] = []
        for path in self.weight_files():
            with open(path, "rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(length))
            header.pop("__metadata__", None)
            keys.extend(header)
        return keys

    def _read_state(self, dtype, skip_prefixes=(), rename=None):
        """Weights from a single local safetensors file: unprefixed, fp8
        dequantized, cast to `dtype`.

        ComfyUI-style text-encoder files carry CausalLM-prefixed keys
        ('model.embed_tokens...') with no LM head; the text model classes want
        them unprefixed, so strip 'model.' as we load. Returns the state dict
        plus counts of what was dequantized and what was skipped.
        """
        import torch
        from safetensors import safe_open

        state, scales, skipped = {}, {}, 0
        for path in self.weight_files():
            with safe_open(str(path), framework="pt", device=self.device) as f:
                for key in f.keys():
                    if skip_prefixes and key.startswith(tuple(skip_prefixes)):
                        skipped += 1
                        continue
                    name = rename(key) if rename else strip_tower_prefix(key)
                    if name is None:
                        skipped += 1
                        continue
                    if name.endswith(QUANT_SUFFIX):
                        continue  # a descriptor; the scale carries the number
                    if name.endswith(SCALE_SUFFIX):
                        scales[name[: -len(SCALE_SUFFIX)]] = f.get_tensor(key)
                        continue
                    state[name] = f.get_tensor(key)
        for base, scale in scales.items():
            key = f"{base}.weight"
            if key not in state:
                raise RuntimeError(
                    f"{base}: a weight_scale with no weight to scale — "
                    "the checkpoint is malformed or keyed unexpectedly"
                )
            # float32 intermediate: fp8 e4m3 carries 3 mantissa bits and bf16
            # carries 8, so the product should not be rounded twice.
            state[key] = (state[key].to(torch.float32)
                          * scale.to(torch.float32)).to(dtype)
        for key, tensor in state.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                state[key] = tensor.to(dtype)
        return state, len(scales), skipped

    def _load_local_safetensors(self, dtype, vision: bool = False):
        """Build the text tower (or, with `vision`, the whole VL model) from
        an HF config, weights from local safetensors."""
        from transformers import AutoConfig

        config = self._from_pretrained(AutoConfig)
        model_cls, text_config, skip_prefixes = self._text_tower(config)
        rename = None
        if vision:
            # Image landmarks are built on the Qwen-VL image-token protocol
            # (vision start, end and pad tokens around the patch run), so
            # what is checked is that capability, not a model name.
            if (getattr(config, "vision_config", None) is None
                    or not all(hasattr(config, a) for a in IMAGE_TOKEN_IDS)):
                raise SystemExit(
                    f"[tokencollider] {self.model_name} (model_type "
                    f"{getattr(config, 'model_type', '?')!r}) has no vision "
                    "tower taking Qwen-VL style image tokens, which image "
                    "landmarks are built on.")
            from transformers import MODEL_MAPPING

            model_cls, text_config, skip_prefixes = MODEL_MAPPING[type(config)], config, ()
            rename = vl_full_key
        # A VLM's text tower has the SAME shapes as its text-only sibling
        # (Qwen3-VL-4B and Qwen3-4B are both 36 x 2560), so a mismatched
        # config loads without a single complaint and is wrong exactly where
        # it matters: position encoding and rope scaling. Catch it here.
        if not vision and not skip_prefixes and any(
            k.startswith(VISION_PREFIXES) for k in self._peek_keys()
        ):
            raise SystemExit(
                f"[tokencollider] {self.model_name} carries a vision tower, but its config "
                f"resolved to model_type "
                f"{str(getattr(config, 'model_type', '?'))!r} from "
                f"{self.config_source()!r}.\n"
                "Loading a VLM text tower under a text-only config succeeds "
                "silently and produces wrong hidden states. Point the profile's "
                "config_dir at the matching VLM config "
                "directory."
            )
        # Construct the skeleton on CPU (NOT meta: computed buffers like the
        # rotary inv_freq must be materialized; NOT cuda: torch.empty commits
        # VRAM immediately, and an fp32 4B skeleton is 16GB). CPU empty pages
        # are lazily committed and never written — assign=True replaces every
        # param with the loaded GPU tensor, so only the real bf16 weights
        # (~8GB) ever land in VRAM.
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            import contextlib

            no_init_weights = contextlib.nullcontext
        with no_init_weights():
            model = model_cls(text_config)
        state, n_dequantized, n_skipped = self._read_state(dtype, skip_prefixes, rename)
        shards = self.weight_files()
        if len(shards) > 1:
            print(f"[tokencollider] {len(shards)} weight shards")
        if n_dequantized:
            print(f"[tokencollider] dequantized {n_dequantized} fp8 matrices to {dtype}")
        if n_skipped:
            print(f"[tokencollider] skipped {n_skipped} keys outside the model "
                  f"({', '.join(skip_prefixes) or 'lm_head'})")
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"state dict mismatch: missing={missing} unexpected={unexpected}"
            )
        model.to(self.device)  # move the computed buffers; params are already there
        return model

    def release(self) -> dict:
        """Free the GPU without killing the session: drop the model weights
        and return the VRAM (e.g. to ComfyUI). Everything cached keeps
        working; the next operation that truly needs a forward pass reloads
        lazily — re-running the free-VRAM gate, so a GPU still occupied by
        another app fails loudly instead of thrashing."""
        import gc

        had = self._model is not None
        self._model = None
        if self._auto_device:
            self.device = None  # re-run require_cuda() on the next load
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {"released": had}

    def _forward_all(self, text: str) -> list[np.ndarray]:
        """Templated text through the model -> per-token hidden states for
        EVERY layer: a list of (seq, dim) arrays, index 0 = embedding-table
        output, index i = output of block i."""
        return self._forward_all_batch([text])[0]

    def _forward_all_batch(self, texts: list[str]) -> list[list[np.ndarray]]:
        """One padded forward pass for many texts. Right-padding keeps every
        real token's position and (causal) attention identical to a solo run;
        pad rows are sliced away per item, so results match _forward_all up to
        bf16 batching noise."""
        import torch

        if self._model is None:
            self._load_model()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        templated = [self._prefix + t + self._suffix for t in texts]
        inputs = self._tokenizer(
            templated, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # One device->host transfer for the whole batch, not L x B small ones.
        stacked = torch.stack(out.hidden_states).float().cpu().numpy()  # (L,B,S,D)
        masks = inputs["attention_mask"].bool().cpu().numpy()
        self.n_layers = stacked.shape[0]
        return [
            [np.ascontiguousarray(stacked[l, b][masks[b]]) for l in range(self.n_layers)]
            for b in range(len(texts))
        ]

    def _ensure_image_processor(self):
        if self._image_processor is None:
            from transformers import AutoConfig

            # The PIL backend by name: the default picks torchvision when
            # present and warns otherwise, and this venv does not carry it.
            try:
                from transformers import Qwen2VLImageProcessorPil as ImageProcessor
            except ImportError:
                from transformers import Qwen2VLImageProcessor as ImageProcessor

            config = self._from_pretrained(AutoConfig)
            v = getattr(config, "vision_config", None)
            # Qwen3-VL's image processor is Qwen2-VL's with this config's
            # patch geometry (16 / 2 / 2 for the 4B), verified from the
            # vendored config; the vendored directory carries no
            # preprocessor_config.json, so it is built explicitly.
            # Pictures arrive already cropped and resized to IMAGE_SIZE
            # square (tokencollider.images.prepare), so the processor's own resizing
            # is pinned to exactly that many pixels and never fires.
            px = images.IMAGE_SIZE * images.IMAGE_SIZE
            self._image_processor = ImageProcessor(
                patch_size=getattr(v, "patch_size", 16),
                merge_size=getattr(v, "spatial_merge_size", 2),
                temporal_patch_size=getattr(v, "temporal_patch_size", 2),
                size={"shortest_edge": px, "longest_edge": px},
            )
            self._image_ids = {k: getattr(config, a)
                               for k, a in zip(("pad", "start", "end"),
                                               IMAGE_TOKEN_IDS)}
        return self._image_processor

    def _forward_image_all(self, key: str) -> tuple[list[np.ndarray], tuple[int, int], list[np.ndarray]]:
        """An image landmark through the full VL model, inside the same
        scaffold a phrase gets: prefix + <|vision_start|> image tokens
        <|vision_end|> + suffix.

        Returns every layer's (seq, dim) states, the span to pool (the image
        tokens, or the tail from <|vision_end|> on, per `image_pooling`),
        and the per-token frame conditionings are stored from: the whole
        sequence for `image`; for `tail` the prefix plus the tail, the image
        tokens cut out, so the export's tokens are the pooled ones, as they
        are for a phrase."""
        import torch

        hit = images.lookup(self.store, images.key_sha(key))
        if hit is None:
            raise ValueError(f"{key}: image not registered in this store")
        path = hit[0]
        self._ensure_vision()
        proc = self._ensure_image_processor()
        pix = proc(images=[images.prepare(path)], return_tensors="pt")
        grid = pix["image_grid_thw"]
        n_tokens = int(grid.prod(-1).sum()) // (proc.merge_size ** 2)
        pad, start, end = (self._image_ids[k] for k in ("pad", "start", "end"))
        self._ensure_tokenizer()
        tok = self._tokenizer
        prefix_ids = tok(self._prefix, add_special_tokens=False)["input_ids"]
        suffix_ids = tok(self._suffix, add_special_tokens=False)["input_ids"]
        ids = prefix_ids + [start] + [pad] * n_tokens + [end] + suffix_ids
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(input_ids)
        # transformers 5 derives the 3-D positions from which tokens are
        # image tokens; the processor would hand this back, so build it.
        mm_types = (input_ids == pad).int()
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=pix["pixel_values"].to(self.device, torch.bfloat16),
                image_grid_thw=grid.to(self.device),
                mm_token_type_ids=mm_types,
                output_hidden_states=True)
        stacked = torch.stack(out.hidden_states).float().cpu().numpy()  # (L,1,S,D)
        self.n_layers = stacked.shape[0]
        states = [np.ascontiguousarray(stacked[l, 0]) for l in range(self.n_layers)]
        i_start = len(prefix_ids) + 1          # first image token
        i_end = i_start + n_tokens             # one past the last
        if images.key_mode(key) == "tail":
            span = (i_end, len(ids))           # <|vision_end|> .. end
            keep = list(range(len(prefix_ids))) + list(range(i_end, len(ids)))
            frame = [s[keep] for s in states]
        else:
            span = (i_start, i_end)
            frame = states
        return states, span, frame

    def _states_for(self, text: str):
        """(states, span, frame) for any landmark: a phrase through the text
        path, an image through the vision path."""
        if images.is_image_key(text):
            return self._forward_image_all(text)
        states = self._forward_all(text)
        return states, self._span(text), states

    def _warm(self, text: str, light: bool | None = None) -> None:
        """One forward pass fills the pooled-vector cache at every layer plus
        the sampler-layer conditioning tensor — the marginal cost of keeping
        all layers is pooling, so never throw them away. In light mode, store
        the sampler-layer pooled vector alone (see __init__)."""
        if light is None:
            light = self.light
        states, span, frame = self._states_for(text)
        with self.store.bulk():
            self._store_states(text, states, light, span=span, frame=frame)

    def _store_states(self, text: str, states: list[np.ndarray], light: bool,
                      pooled_layers=None, span=None, frame=None) -> None:
        """Cache one phrase's forward pass.

        `pooled_layers` names the layers a LIGHT warm should keep, overriding
        the sampler layer it would otherwise pick. A bulk layer view needs
        pooled vectors at the layer being viewed; without this it fell through
        to a full warm, which on a multi-layer model means writing a whole
        conditioning stack per phrase. At wordlist scale that is the difference
        between tens of megabytes and tens of gigabytes."""
        start, end = span if span is not None else self._span(text)
        frame = states if frame is None else frame
        band = self._band()
        if end <= start:
            # Nothing to pool (an empty phrase); keep the conditionings only.
            for index in self.sampler_layers() or ():
                if 0 <= index < len(frame):
                    self.store.put_conditioning(
                        self.model_name, str(index), self.template, text, frame[index])
            return
        if light:
            if pooled_layers is not None:
                idxs = [int(l) for l in pooled_layers]
            elif band is not None:
                idxs = range(band[0], band[1] + 1)
            elif self.layer == "last":
                idxs = [len(states) - 1]
            else:
                idxs = [int(self.layer)]
            for i in idxs:
                self.store.put(self.model_name, str(i), self.pooling, self.template,
                               text, self._pool(states[i][start:end]))
            return
        for i, hidden in enumerate(states):
            self.store.put(self.model_name, str(i), self.pooling, self.template,
                           text, self._pool(hidden[start:end]))
        # Conditionings cache under numeric layer keys, the same keys
        # conditioning_layers() reads, so a stack spanning many layers is just
        # many rows. A band's high edge is its sampler layer, matching export
        # semantics. (_conditioning_cached still honours the older symbolic
        # key, so rows warmed under "last" are never recomputed.)
        for index in self.sampler_layers() or ():
            if 0 <= index < len(frame):
                self.store.put_conditioning(
                    self.model_name, str(index), self.template, text, frame[index])

    def _band(self) -> tuple[int, int] | None:
        """The configured 'lo-hi' band (e.g. '18-24'); None for single layers.
        A band embeds as the mean of the pooled vectors over its layers."""
        spec = parse_layer_spec(self.layer)
        return spec if isinstance(spec, tuple) else None

    def _resolved_layer(self) -> int | None:
        spec = parse_layer_spec(self.layer)
        if isinstance(spec, tuple):
            # The band's top edge is its sampler layer: conditionings cache
            # from there, and cooks stop there. (Returning None here silently
            # disabled cooking for band configs entirely.)
            return spec[1]
        if spec == "last":
            return None if self.n_layers is None else self.n_layers - 1
        return spec

    def sampler_layers(self) -> tuple[int, ...] | None:
        """Every hidden-state layer a conditioning must carry for this model.

        A single-layer model reports the one layer `layer` resolves to.
        Krea 2 reports twelve, mixed inside its DiT by a learned aggregator,
        so its conditioning is a stack rather than a tensor. None until the
        model layer is known (no forward pass yet, cold cache)."""
        if self._sampler_layers is not None:
            return self._sampler_layers
        idx = self._resolved_layer()
        return None if idx is None else (idx,)

    def layer_count(self) -> int | None:
        """Hidden-state entries: from the model if it has run, else the
        checkpoint's config, else (no config to be found) the cache."""
        if self.n_layers is None:
            self.n_layers = self._config_layer_count()
        if self.n_layers is None:
            # Last resort, and only a lower bound: a light warm stores the
            # charting layer alone, so the highest cached layer can sit far
            # below the top of the model.
            numeric = [
                int(l) for l in self.store.layers_cached(self.model_name)
                if l.isdigit()
            ]
            if numeric:
                self.n_layers = max(numeric) + 1
        return self.n_layers

    def _config_layer_count(self) -> int | None:
        """Decoder blocks + 1 (index 0 is the embedding-table output), read
        from the config the tokenizer already needs, so no forward pass. A
        vision-language checkpoint counts its text tower's blocks."""
        from transformers import AutoConfig

        try:
            config = self._from_pretrained(AutoConfig)
            _cls, build, _skip = self._text_tower(config)
        except SystemExit:
            return None
        n = getattr(build, "num_hidden_layers", None)
        return None if n is None else int(n) + 1

    def _forward_missing(self, texts: list[str], store_fn, verbose: bool = False,
                         label: str = "embedding") -> None:
        """Batched forward passes over `texts`, handing each phrase's per-layer
        hidden states to store_fn(text, states).

        The single place that owns batching, the one-fsync-per-batch bulk
        transaction, and the progress line. Three call sites used to carry
        their own copy of this loop, and one of them (the layer-sweep pre-warm)
        never got batched at all."""
        prog = _Progress(label, len(texts), verbose)
        # Images go one forward each through the vision path; phrases batch.
        image_keys = [t for t in texts if images.is_image_key(t)]
        phrases = [t for t in texts if not images.is_image_key(t)]
        for key in image_keys:
            states, span, frame = self._forward_image_all(key)
            with self.store.bulk():
                store_fn(key, states, span=span, frame=frame)
            prog.step(1, key)
        for lo in range(0, len(phrases), BATCH_SIZE):
            chunk = phrases[lo : lo + BATCH_SIZE]
            all_states = self._forward_all_batch(chunk)
            with self.store.bulk():  # one fsync per batch, not per put
                for text, states in zip(chunk, all_states):
                    store_fn(text, states)
            prog.step(len(chunk), chunk[-1])

    def _pool(self, tokens: np.ndarray) -> np.ndarray:
        if self.pooling == "mean":
            return tokens.mean(axis=0)
        if self.pooling == "last":
            return tokens[-1].copy()
        raise ValueError(f"unknown pooling: {self.pooling}")

    def _conditioning_cached(self, text: str, index: int) -> np.ndarray | None:
        """Cache-only lookup for one layer, mirroring embed()'s two keys.

        The sampler layer also lives under the symbolic config key ("last",
        "18-34") for anything warmed before conditionings were keyed by layer.
        Probe that before declaring a miss, or a warmed universe re-runs the
        model purely to re-key rows already on disk."""
        hit = self.store.get_conditioning(
            self.model_name, str(index), self.template, text
        )
        if hit is not None:
            return hit
        if (self._sampler_layers is None
                and str(self.layer) != str(index)
                and index == self._resolved_layer()):
            return self.store.get_conditioning(
                self.model_name, str(self.layer), self.template, text
            )
        return None

    def conditioning(self, text: str) -> Conditioning:
        """Per-token hidden states at every layer the sampler reads — the
        export representation. A size-one stack for a single-layer model."""
        layers = self.sampler_layers()
        if layers is None:
            self._warm(text, light=False)  # a forward pass learns the layer
            layers = self.sampler_layers()
            if layers is None:
                raise RuntimeError("model layer count still unknown after a forward pass")
        return Conditioning(self.conditioning_layers(text, list(layers)))

    def conditioning_layers(self, text: str, layers: list[int]) -> dict[int, np.ndarray]:
        """Per-token hidden states at arbitrary layers, cached like pooled
        vectors. One forward pass fills every miss at once."""
        out: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for l in layers:
            l = int(l)
            cached = self._conditioning_cached(text, l)
            if cached is None:
                missing.append(l)
            else:
                out[l] = cached
        if missing:
            _states, _span, frame = self._states_for(text)
            for l in missing:
                if not 0 <= l < len(frame):
                    raise ValueError(
                        f"layer {l} out of range — model has {len(frame)} "
                        "hidden-state entries (0 = embedding output)"
                    )
                self.store.put_conditioning(
                    self.model_name, str(l), self.template, text, frame[l]
                )
                out[l] = frame[l]
        return out

    def warm_all_conditionings(self, texts: list[str], layers: list[int],
                               verbose: bool = True) -> None:
        """One forward per phrase fills the conditioning cache at every
        requested layer — the layer-sweep pre-warm. Without it, per-layer
        misses cost one forward per phrase PER LAYER."""
        layers = [int(l) for l in layers]
        pending = [
            t for t in texts
            if any(self._conditioning_cached(t, l) is None for l in layers)
        ]
        if not pending:
            return

        def store(text, states, span=None, frame=None):
            frame = states if frame is None else frame
            for l in layers:
                if 0 <= l < len(frame):
                    self.store.put_conditioning(
                        self.model_name, str(l), self.template, text, frame[l])

        self._forward_missing(pending, store, verbose, "pre-warming")

    @staticmethod
    def _check_capture(capture, start_index: int, stop_index: int) -> None:
        outside = [int(c) for c in sorted({int(c) for c in capture})
                   if not start_index <= int(c) <= stop_index]
        if outside:
            raise ValueError(
                f"capture depths {outside} lie outside the cooked range "
                f"{start_index}..{stop_index} — layers below the inserter "
                "cannot be cooked, blend them from the landmarks instead"
            )

    def cook_layers(self, hidden: np.ndarray, start_index: int,
                    pulls: dict[int, tuple[float, np.ndarray]] | None = None,
                    stop_index: int | None = None,
                    capture=(), position_ids=None) -> dict[int, np.ndarray]:
        """Resume the forward pass from an injected hidden state, handing back
        the running state at every requested layer.

        `hidden` is a (seq, dim) array at hidden-state index `start_index`
        (0 = embedding output, i = input to block i). Runs blocks up to
        `stop_index` — the hidden-state index to produce. None or n_blocks
        means the post-norm final state; a smaller index yields that raw
        pre-norm state (e.g. 35 = ComfyUI's Z-Image layer_idx=-2, which is
        never final-normed). Either way the result is a valid conditioning
        for a sampler configured at that layer.

        `capture` names additional layers to return along the way; the states
        it hands back are raw pre-norm, matching what output_hidden_states
        reports at those indices. `stop_index` is always included. One resumed
        pass therefore fills a whole multi-layer stack, which is what a
        multi-layer consumer reads.

        `pulls` re-mixes the running state right after a given index is
        produced: {state_index: (alpha, target)}, h <- (1-a)*h + a*target.
        Targets must be raw pre-norm states at indices < stop_index.

        `position_ids` overrides the flat range: a frame holding image tokens
        needs Krea 2's 3-D positions (see mixed_position_ids). DeepStack
        (visual features added in the first three blocks) is not replayed,
        so a cook over image tokens should start at index 3 or later.
        """
        import torch

        # Argument checks that do not need the model come first: a bad capture
        # list should not cost an 8GB load before it is rejected.
        if stop_index is not None:
            if start_index > stop_index or start_index < 0:
                raise ValueError(
                    f"need 0 <= start {start_index} <= stop {stop_index}")
            self._check_capture(capture, start_index, stop_index)
        if self._model is None:
            self._load_model()
        model = self._language_model()
        missing = [a for a in ("layers", "rotary_emb", "norm")
                   if not hasattr(model, a)]
        if missing:
            raise SystemExit(
                f"[tokencollider] cooking runs the decoder blocks one at a time, and "
                f"{type(model).__name__} has no {', '.join(missing)}; this "
                "model can embed and export but not cook")
        n_blocks = len(model.layers)
        if stop_index is None:
            stop_index = n_blocks
            self._check_capture(capture, start_index, stop_index)
        if not 0 <= start_index <= stop_index <= n_blocks:
            raise ValueError(
                f"need 0 <= start {start_index} <= stop {stop_index} <= {n_blocks}"
            )
        wanted = {int(c) for c in capture} | {stop_index}
        out: dict[int, np.ndarray] = {}
        if start_index in wanted:
            out[start_index] = np.ascontiguousarray(hidden, dtype=np.float32)
        if start_index == stop_index:  # already the requested state
            return out
        h = torch.from_numpy(np.ascontiguousarray(hidden)).to(
            self.device, torch.bfloat16
        )[None]
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=self.device)[None]
        else:
            position_ids = position_ids.to(self.device)
            if position_ids.shape[-1] != h.shape[1]:
                raise ValueError(
                    f"position ids cover {position_ids.shape[-1]} tokens, "
                    f"the frame has {h.shape[1]}")
        pos_emb = model.rotary_emb(h, position_ids)

        def snapshot(t):
            return t[0].float().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            for j in range(start_index, stop_index):
                block = model.layers[j](
                    h,
                    attention_mask=None,  # None = causal, matching _forward_all
                    position_ids=position_ids,
                    position_embeddings=pos_emb,
                )
                h = block[0] if isinstance(block, tuple) else block
                index = j + 1
                if pulls and index in pulls and index < stop_index:
                    alpha, target = pulls[index]
                    t = torch.from_numpy(np.ascontiguousarray(target)).to(
                        self.device, torch.bfloat16
                    )[None]
                    h = (1.0 - alpha) * h + alpha * t
                if index in wanted and index < stop_index:
                    out[index] = snapshot(h)
            if stop_index == n_blocks:
                h = model.norm(h)
            out[stop_index] = snapshot(h)
        return out

    def cook(self, hidden: np.ndarray, start_index: int,
             pulls: dict[int, tuple[float, np.ndarray]] | None = None,
             stop_index: int | None = None) -> np.ndarray:
        """Single-layer cook: the size-one case of cook_layers()."""
        states = self.cook_layers(hidden, start_index, pulls, stop_index)
        return states[max(states)]

    def embed(self, text: str) -> np.ndarray:
        """Pooled vector at the configured (sampler) layer or band."""
        band = self._band()
        if band is not None:
            # Band mean caches under its own 'lo-hi' key; per-layer vectors
            # come from one warm (light stores exactly the band's layers).
            cached = self.store.get(
                self.model_name, self.layer, self.pooling, self.template, text
            )
            if cached is not None:
                return cached
            vals = []
            for l in range(band[0], band[1] + 1):
                v = self.store.get(
                    self.model_name, str(l), self.pooling, self.template, text
                )
                if v is None:
                    self._warm(text)
                    v = self.store.get(
                        self.model_name, str(l), self.pooling, self.template, text
                    )
                    if v is None:
                        raise ValueError(
                            f"layer {l} out of range — model has {self.n_layers} "
                            "hidden-state entries (0 = embedding output)"
                        )
                vals.append(v)
            vec = np.mean(vals, axis=0).astype(np.float32)
            self.store.put(self.model_name, self.layer, self.pooling,
                           self.template, text, vec)
            return vec
        # Pre-layer-scrub caches stored under the symbolic name ("last");
        # warm() writes numeric keys only, so try both before a forward pass.
        cached = self.store.get(
            self.model_name, self.layer, self.pooling, self.template, text
        )
        if cached is not None:
            return cached
        idx = self._resolved_layer()
        if idx is not None:
            cached = self.store.get(
                self.model_name, str(idx), self.pooling, self.template, text
            )
            if cached is not None:
                return cached
        self._warm(text)
        return self.store.get(
            self.model_name, str(self._resolved_layer()), self.pooling,
            self.template, text,
        )

    def embed_layer(self, text: str, layer: int) -> np.ndarray:
        """Pooled vector at an arbitrary layer. Cache-hit whenever the phrase
        has ever been embedded — _warm stores every layer per forward pass."""
        key = str(int(layer))
        if self.layer_count() is not None and not 0 <= int(layer) < self.n_layers:
            raise ValueError(
                f"layer {layer} out of range — model has {self.n_layers} "
                "hidden-state entries (0 = embedding output)"
            )
        cached = self.store.get(self.model_name, key, self.pooling, self.template, text)
        if cached is None:
            self._warm(text)
            cached = self.store.get(self.model_name, key, self.pooling, self.template, text)
            if cached is None and self.light:
                # Light warm only stored the sampler layer/band; an arbitrary
                # layer request means this phrase is being probed — warm it
                # in full so every layer is available from one forward pass.
                self._warm(text, light=False)
                cached = self.store.get(
                    self.model_name, key, self.pooling, self.template, text
                )
            if cached is None:
                raise ValueError(
                    f"layer {layer} out of range — model has {self.n_layers} "
                    "hidden-state entries (0 = embedding output)"
                )
        return cached

    def _cached_pooled(self, text: str) -> np.ndarray | None:
        """Cache-only lookup mirroring embed()'s two keys (symbolic + numeric),
        without ever triggering a forward pass. For a band config, a miss on
        the band key falls back to composing the mean from cached per-layer
        vectors — a phrase warmed under any config that stored those layers
        never costs a second forward pass just to re-key it as a band."""
        cached = self.store.get(
            self.model_name, self.layer, self.pooling, self.template, text
        )
        if cached is not None:
            return cached
        band = self._band()
        if band is not None:
            vals = []
            for l in range(band[0], band[1] + 1):
                v = self.store.get(
                    self.model_name, str(l), self.pooling, self.template, text
                )
                if v is None:
                    return None
                vals.append(v)
            vec = np.mean(vals, axis=0).astype(np.float32)
            self.store.put(self.model_name, self.layer, self.pooling,
                           self.template, text, vec)
            return vec
        idx = self._resolved_layer()
        if idx is not None:
            return self.store.get(
                self.model_name, str(idx), self.pooling, self.template, text
            )
        return None

    def embed_layers_many(self, texts: list[str], layer,
                          verbose: bool = False) -> np.ndarray:
        """Batched pooled vectors at an arbitrary layer or (lo, hi) band view.
        Cache misses warm in batches (one forward per TOKENCOLLIDER_BATCH phrases) —
        never the one-phrase-one-forward loop that makes universe-scale
        layer views crawl for minutes."""
        layers = (list(range(layer[0], layer[1] + 1))
                  if isinstance(layer, tuple) else [int(layer)])

        def compose(text: str) -> np.ndarray | None:
            vals = []
            for l in layers:
                v = self.store.get(self.model_name, str(l), self.pooling,
                                   self.template, text)
                if v is None:
                    return None
                vals.append(v)
            return vals[0] if len(vals) == 1 else np.mean(vals, axis=0).astype(np.float32)

        vecs: list[np.ndarray | None] = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            vecs[i] = compose(t)
            if vecs[i] is None:
                missing.append(i)
        if verbose and missing:
            print(f"[tokencollider] layer view {layer}: {len(texts) - len(missing)}/"
                  f"{len(texts)} cached, warming {len(missing)} in batches ...")
        # Honour light mode here: this is a bulk view over a whole universe,
        # not a single probed phrase, so it caches the layers it is about to
        # read and nothing else.
        self._forward_missing(
            [texts[i] for i in missing],
            lambda text, states, span=None, frame=None: self._store_states(
                text, states, self.light, pooled_layers=layers, span=span, frame=frame),
            verbose, "warming")
        for i in missing:
            vecs[i] = compose(texts[i])
            if vecs[i] is None:
                raise ValueError(
                    f"layer {layer} out of range — model has "
                    f"{self.n_layers} hidden-state entries"
                )
        return np.stack(vecs)

    def embed_many(self, texts: list[str], verbose: bool = False) -> np.ndarray:
        self.layer_count()  # hydrate n_layers from the cache for _cached_pooled
        vecs: list[np.ndarray | None] = [None] * len(texts)
        missing = []
        # bulk(): band configs compose-and-cache inside _cached_pooled, and
        # one fsync per phrase turns a 2k-universe load into minutes of disk.
        with self.store.bulk():
            for i, t in enumerate(texts):
                vecs[i] = self._cached_pooled(t)
                if vecs[i] is None:
                    missing.append(i)
        if verbose and missing:
            print(f"[tokencollider] cache: {len(texts) - len(missing)}/{len(texts)} hit, "
                  f"embedding {len(missing)} ...")
        self._forward_missing(
            [texts[i] for i in missing],
            lambda text, states, span=None, frame=None: self._store_states(
                text, states, self.light, span=span, frame=frame),
            verbose, "embedding")
        for i in missing:
            vecs[i] = self.embed(texts[i])  # cache hit; band means compose here
        return np.stack(vecs)


class FakeEmbedder:
    """Deterministic hash-seeded vectors — develop clients without the GPU.

    Geometry is random (no semantics), but stable across runs: the same word
    always lands in the same place, so a Godot session behaves consistently.
    Layers rotate smoothly between two independent configurations, so the
    layer-scrub UI has real motion to show.
    """

    def __init__(self, dim: int = 2560, n_layers: int = 37):
        self.dim = dim
        self.n_layers = n_layers
        self.layer = "last"  # mirror Embedder's sampler-layer attribute
        self.image_pooling = "image"

    def layer_count(self) -> int:
        return self.n_layers

    def template_prefix_tokens(self) -> int:
        return 0

    def template_tail_tokens(self) -> int:
        return 0

    def _resolved_layer(self) -> int:
        return self.n_layers - 1

    def sampler_layers(self) -> tuple[int, ...]:
        return (self.n_layers - 1,)

    def _config(self, text: str, offset: int) -> np.ndarray:
        import hashlib

        seed = int.from_bytes(
            hashlib.sha256(text.encode()).digest()[offset : offset + 8], "little"
        )
        rng = np.random.default_rng(seed)
        # A shared base direction gives fake vectors CLIP-like anisotropy.
        base = np.random.default_rng(0).standard_normal(self.dim)
        return (base * 3.0 + rng.standard_normal(self.dim)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_layer(text, self.n_layers - 1)

    def embed_layer(self, text: str, layer: int) -> np.ndarray:
        import math

        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"layer {layer} out of range (0..{self.n_layers - 1})")
        t = int(layer) / max(1, self.n_layers - 1)
        a = t * math.pi / 2  # low layers ~ config A, top layers ~ config B
        return (
            math.cos(a) * self._config(text, 0) + math.sin(a) * self._config(text, 8)
        ).astype(np.float32)

    def embed_many(self, texts: list[str], verbose: bool = False) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def conditioning_tokens(self, text: str) -> np.ndarray:
        import hashlib

        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[8:16], "little")
        rng = np.random.default_rng(seed)
        seq_len = 3 + seed % 6
        tokens = rng.standard_normal((seq_len, self.dim)).astype(np.float32)
        # Keep fake pooled/tensor views consistent: mean of tokens == embed(text).
        tokens += (self.embed(text) - tokens.mean(axis=0))[None, :]
        return tokens

    def conditioning(self, text: str) -> Conditioning:
        return Conditioning.single(self.conditioning_tokens(text),
                                   self.n_layers - 1)

    def conditioning_layers(self, text: str, layers) -> dict[int, np.ndarray]:
        # Fake geometry has no layer; every layer is the same tokens.
        return {int(l): self.conditioning_tokens(text) for l in layers}


def load_universe_file(path: Path) -> list[str]:
    phrases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            phrases.append(line)
    return phrases
