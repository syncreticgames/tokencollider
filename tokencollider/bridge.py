"""Bridge to Ostris ai-toolkit: pre-write its text-embedding cache.

ai-toolkit with `cache_text_embeddings: true` reads one safetensors per
training image from `<dataset>/_t_e_cache/`, keyed by an md5 of (caption,
arch, version), and SKIPS computing any file that already exists. So the
bridge is a file-drop, not a patch: for each image we embed its caption the
way the trainer's own encoder pass would, then concatenate the flown anchor
tensor(s) after it — the training-time mirror of the inference-time
ConditioningConcat recipe. The caption absorbs the nameable incidentals; the
anchor receives the concept. "The trigger word is this exact tensor."

Two archs, two file layouts, both verified against ostris/ai-toolkit main:

- `zimage` (@ 817f3dc, 08072026): key "text_embed", a (seq, dim) tensor of
  the raw hidden state 35; the whole templated sequence, nothing sliced.
- `krea2` (08222026): `extensions_built_in/diffusion_models/krea2` encodes
  one prompt at natural length, stacks hidden states 2, 5, ..., 35 as
  (L, 12, 2560), drops the 34-token system prefix, keeps the trailing turn,
  and flattens the stack layer-major to (L, 12*2560) so the toolkit's
  batching reads the list length as the batch size. `predict_velocity`
  restores the layer axis. The cache file is `AdvancedPromptEmbeds.save`:
  key "text_embeds", bf16, metadata class_name "AdvancedPromptEmbeds",
  which is what `PromptEmbeds.load` dispatches on. That frame (prefix
  dropped, suffix kept, no padding) is exactly the export frame the krea2
  profile already produces, and the flatten is the one ComfyUI uses.

Shared (both archs):
- filename: <basename>_<b64url(md5(json({caption, text_embedding_space_version,
  text_embedding_version})))>.safetensors, "=" stripped; the space version is
  the arch string (`BaseModel.text_embedding_space_version` returns `arch`)
- trigger words are string-substituted into captions BEFORE hashing
  ("[trigger]" replaced; else "<trigger> " prepended)
"""

import base64
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .conditioning import Conditioning, check_frame
from . import provenance

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
TEXT_EMBEDDING_VERSION = 1

# ai-toolkit's cache file layout per arch, as its own loaders read it. Which
# encoder conventions an arch expects is not here: that is the built-in
# profile declaring the same trainer_arch (see check_dialect).
#   key    the tensor's name in the cache file
#   fused  a multi-layer stack flattened layer-major to (L, n*dim), bf16, as
#          AdvancedPromptEmbeds; otherwise one (seq, dim) tensor
ARCHS = {
    "zimage": {"key": "text_embed", "fused": False},
    "krea2": {"key": "text_embeds", "fused": True},
}


def check_dialect(arch: str, embedder) -> str | None:
    """Why this session's encoder output would not match what ai-toolkit's
    own encoder pass produces for `arch`, or None when it would. The
    reference is the built-in profile that declares this trainer_arch, so the
    contract lives in the profile data, not here."""
    from . import profiles
    from .embedder import canonical_layer

    refs = [p for p in profiles.builtins().values() if p.trainer_arch == arch]
    if not refs:
        return f"no built-in profile declares trainer_arch {arch!r}"
    ref = refs[0]
    want_layers = (tuple(ref.sampler_layers) if ref.sampler_layers
                   else (canonical_layer(ref.layer),))
    have = embedder.sampler_layers()
    have_layers = (tuple(have) if have and ref.sampler_layers
                   else (canonical_layer(embedder.layer),))
    off = []
    if have_layers != want_layers:
        off.append(f"layers {list(have_layers)} (expects {list(want_layers)})")
    if embedder.template != ref.template:
        off.append("a different template")
    if bool(embedder.trim_template_prefix) != bool(ref.trim_template_prefix):
        off.append(f"trim {embedder.trim_template_prefix} "
                   f"(expects {ref.trim_template_prefix})")
    if not off:
        return None
    return f"arch {arch!r} follows the {ref.name!r} profile; this session has " + ", ".join(off)


def inject_trigger(caption: str, trigger: str | None) -> str:
    """ai-toolkit's inject_trigger_into_prompt with add_if_not_present, as of
    ai-toolkit ecee894 (toolkit/prompt_utils.py). The trigger is baked in
    before hashing, exactly as the trainer does, or the hash differs and
    ai-toolkit silently encodes the caption itself, with no anchor.

    `[name]` and `[trigger]` are always replaced (with nothing when there's
    no trigger). A trigger that still isn't in the caption is prepended with
    a space, even to an empty caption, so the dropout caption for "sks" is
    "sks ". A blank trigger is never added."""
    trigger = trigger or ""
    for token in ("[name]", "[trigger]"):
        caption = caption.replace(token, trigger)
    if trigger.strip() != "" and caption.count(trigger) == 0:
        caption = trigger + " " + caption
    return caption


def cache_hash(caption: str, space: str) -> str:
    key = json.dumps(
        OrderedDict([
            ("caption", caption),
            ("text_embedding_space_version", space),
            ("text_embedding_version", TEXT_EMBEDDING_VERSION),
        ]),
        sort_keys=True,
    )
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def load_anchor(path: Path) -> tuple[Conditioning, dict]:
    return Conditioning.load(str(path))


def fuse_layers(cond: Conditioning) -> np.ndarray:
    """(L, n*dim), layer i owning features [i*dim, (i+1)*dim): the layout
    ai-toolkit's krea2 arch caches and ComfyUI's encoder emits."""
    stacked = np.stack([cond[l] for l in cond.layers], axis=1)  # (L, n, dim)
    return stacked.reshape(stacked.shape[0], -1)


ANCHOR_SUFFIX = ".anchor.safetensors"


def dropout_caption(trigger: str | None) -> str:
    """ai-toolkit's `FileItemDTO.get_dropout_caption`: a dropped caption is
    empty, but the trigger word is still injected downstream, so it survives.
    The blank embedding is cached under THIS caption, not under ""."""
    return inject_trigger("", trigger)


def _save(arch: str, combined: Conditioning, out: Path, metadata: dict) -> None:
    layout = ARCHS[arch]
    if layout["fused"]:
        import torch
        from safetensors.torch import save_file

        fused = torch.from_numpy(fuse_layers(combined)).to(torch.bfloat16)
        save_file({layout["key"]: fused}, str(out),
                  metadata={"class_name": "AdvancedPromptEmbeds", **metadata})
    else:
        if len(combined) != 1:
            raise SystemExit(
                f"[tokencollider] this conditioning spans layers {list(combined.layers)}, "
                f"but ai-toolkit's {arch} cache holds a single tensor.")
        from safetensors.numpy import save_file

        save_file({layout["key"]: combined.tensor}, str(out), metadata=metadata)


def alt_suffix(i: int) -> str:
    """Sidecar name for one ALTERNATIVE anchor. Numbered sidecars are picked
    from at random per step; the unnumbered one is the single fixed anchor."""
    return f".anchor.{i:02d}.safetensors"


def _save_teachers(arch: str, student: Conditioning, anchors: list[Conditioning],
                   out: Path, metadata: dict, alternates: bool) -> None:
    """The teacher sidecars beside a student cache file: the student followed
    by every anchor in one `.anchor` file, or with `alternates` one numbered
    sidecar per anchor. The blank (dropout) caption gets the same treatment
    as a real one, or a dropout step would train toward a different target."""
    meta = {**metadata, "role": "teacher"}
    blocks = [a.relabel(student.layers) for a in anchors]
    if alternates:
        for i, block in enumerate(blocks):
            one = Conditioning.concat([student, block])
            _save(arch, one, out.with_name(out.stem + alt_suffix(i)),
                  {**meta, "alternate": str(i),
                   "anchor_tokens": str(one.seq_len - student.seq_len)})
    else:
        combined = Conditioning.concat([student, *blocks])
        _save(arch, combined, out.with_name(out.stem + ANCHOR_SUFFIX),
              {**meta, "anchor_tokens": str(combined.seq_len - student.seq_len)})


def write_cache(dataset: Path, embedder, anchors: list[Path], arch: str,
                trigger: str | None = None, caption_ext: str = ".txt",
                default_caption: str | None = None,
                jumpstart: bool = False, alternates: bool = False) -> list[Path]:
    """One cache file per image: concat(caption tokens, anchor tokens...), in
    the arch's file layout. Returns the written paths.

    `jumpstart` writes the caption alone at the canonical path and the
    anchored version beside it as `<name>.anchor.safetensors`: the student
    and teacher conditionings for trainers/tokencollider_jumpstart, which
    distils the anchor into a LoRA.

    `alternates` changes what repeated `--anchor` means. By default the anchors
    CONCATENATE into one teacher, which is a single point in the coordinate
    space, and a LoRA distilled from one point can only ever learn one point.
    With `alternates` each anchor gets its own numbered teacher sidecar and the
    trainer samples one per step, so the LoRA learns the concept's local
    manifold instead of its centre. Feed it a cook stack or a handful of
    neighbours from the same chart.

    It also writes the DROPOUT pair, because
    caption dropout does work with a cached text encoder and takes a separate
    path through ai-toolkit (verified against main, 08232026):
    `FileItemDTO.load_prompt_embedding` rolls the dice per item and loads
    `get_blank_text_embedding_path()` on a drop. That file hashes the dropout
    caption under the same three-field key everything else uses (`text_only`
    drops control conditioning from the key, and a plain image dataset has
    none), so the bridge can write it. Without it ai-toolkit would fall
    through to encoding the blank live, which needs the text encoder the
    bridge exists to avoid, and the resulting embedding would carry no
    anchor."""
    # ai-toolkit's own configs write `caption_ext: txt`, without the dot, and
    # Path.with_suffix refuses that, so accept both spellings.
    if not caption_ext.strip("."):
        raise SystemExit(f"[tokencollider] caption extension {caption_ext!r} is empty")
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext
    if arch not in ARCHS:
        raise SystemExit(f"[tokencollider] no ai-toolkit cache contract for arch {arch!r} "
                         f"(known: {', '.join(ARCHS)})")
    if alternates and not jumpstart:
        raise SystemExit("[tokencollider] alternates are teacher sidecars, which only "
                         "the jumpstart layout writes")
    # The caption must land in the frame the trainer's own encoder pass would
    # produce. For krea2 that drops the system prefix, which is the export
    # trim the profile already applies; for zimage it is 0.
    trim_fn = getattr(embedder, "export_trim", None)
    trim = int(trim_fn()) if trim_fn is not None else 0
    anchor_conds = []
    anchor_meta = {}
    for i, p in enumerate(anchors):
        cond, meta = load_anchor(p)
        # An anchor is concatenated after captions this embedder makes, so it
        # must come from the same frame (docs/export-format.md).
        try:
            # The same config the export wrote, read the same way.
            config = provenance.embedder_config(embedder)
            for note in check_frame(cond, meta, template=config["template"],
                                    layers=embedder.sampler_layers(),
                                    prefix_tokens=trim, model=config["model"]):
                print(f"[tokencollider] note: {p.name}: {note}")
        except ValueError as e:
            raise SystemExit(f"[tokencollider] {p} doesn't fit this profile: {e}")
        anchor_conds.append(cond)
        anchor_meta[f"anchor_{i}"] = json.dumps(
            {"path": str(p), "universe": meta.get("universe"),
             "band": meta.get("band"), "cooked": meta.get("cooked"),
             "axes": meta.get("axes"), "center": meta.get("center")})
    images = sorted(
        p for p in dataset.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )
    if not images:
        raise SystemExit(f"[tokencollider] no images found in {dataset}")
    cache_dir = dataset / "_t_e_cache"
    cache_dir.mkdir(exist_ok=True)

    # One dropout caption for the whole dataset, so embed it once. Skipped
    # when it hashes to the same file as a real caption (a dataset with no
    # captions and no trigger word: the blank IS the caption).
    blank_caption = dropout_caption(trigger)
    blank_cap = None
    if jumpstart:
        blank_cap = embedder.conditioning(blank_caption)
        if trim:
            blank_cap = blank_cap.trim(trim)

    written = []
    for img in images:
        caption_path = img.with_suffix(caption_ext)
        # As ai-toolkit's load_caption reads it (dataloader_mixins.py,
        # ecee894): the file exactly as written, trailing newline included.
        # Only a blank file, or a missing one, falls back to the default.
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
            if caption.strip() == "" and default_caption is not None:
                caption = default_caption
        else:
            caption = default_caption if default_caption is not None else ""
        caption = inject_trigger(caption, trigger)
        cap = embedder.conditioning(caption)
        if trim:
            cap = cap.trim(trim)
        # Anchors were flown under this same config, so they sit at these
        # layers by construction — say so rather than trusting whatever layer
        # their metadata happens to name.
        blocks = [cap] + [a.relabel(cap.layers) for a in anchor_conds]
        combined = Conditioning.concat(blocks)
        cap_len = cap.seq_len
        out = cache_dir / f"{img.stem}_{cache_hash(caption, arch)}.safetensors"
        metadata = {"caption": caption,
                    "caption_tokens": str(cap_len),
                    "anchor_tokens": str(combined.seq_len - cap_len),
                    "layers": json.dumps(list(combined.layers)),
                    "arch": arch,
                    **anchor_meta}
        if jumpstart:
            # Student conditioning at the canonical path, teachers beside it.
            _save(arch, cap, out, {**metadata, "anchor_tokens": "0", "role": "student"})
            _save_teachers(arch, cap, anchor_conds, out, metadata, alternates)
            blank_hash = cache_hash(blank_caption, arch)
            if blank_hash != cache_hash(caption, arch):
                blank_out = cache_dir / f"{img.stem}_{blank_hash}.safetensors"
                blank_meta = {**metadata, "caption": blank_caption,
                              "caption_tokens": str(blank_cap.seq_len),
                              "dropout": "1"}
                _save(arch, blank_cap, blank_out,
                      {**blank_meta, "anchor_tokens": "0", "role": "student"})
                _save_teachers(arch, blank_cap, anchor_conds, blank_out,
                               blank_meta, alternates)
                written.append(blank_out)
        else:
            _save(arch, combined, out, metadata)
        written.append(out)
    return written
