"""Pack an HF-sharded Qwen3-VL-4B (a finetune, typically) into one safetensors
file that ComfyUI's Krea 2 CLIPLoader accepts.

TokenCollider reads the shards directly (a profile's `model:` may be the
directory), so this is only for ComfyUI. What it does, and why:

- Merges every shard the index names into one file, in the layout ComfyUI's
  own `qwen3vl_4b_fp8_scaled.safetensors` uses: `model.layers.*`,
  `model.embed_tokens`, `model.norm`, `model.visual.*`. ComfyUI's loader
  (`comfy/sd.py`) detects Qwen3-VL-4B by `model.visual.deepstack_merger_list.
  0.norm.weight` and the merger's 2560 width, then renames prefixes itself,
  so the HF nesting `model.language_model.*` would also load; the flat layout
  is chosen so the packed file matches the stock one key for key.
- Casts floating tensors to bf16 (the precision ComfyUI and this tool run
  the encoder at). fp8-scaled shards are dequantized first (`weight *
  weight_scale`), so the output is always a plain bf16 file.
- Drops `lm_head.weight`: Qwen3-VL-4B ties it to the embedding table and
  ComfyUI's 4B config has no head, so the key would only be "unexpected".
- Requires the vision tower, because ComfyUI's detection keys live there. A
  text-only finetune (shards of `Qwen3VLTextModel`) gets it spliced in from
  `--vision-from`, which may be the stock ComfyUI encoder file. The vision
  tower is frozen during a text finetune, so the base's is the right one.

Refuses to write a file that would not be recognized: 36 text layers, the
embedding table, the final norm, and the vision tower all have to be there.
"""

import argparse
import json
import sys
from pathlib import Path

SCALE_SUFFIX = ".weight_scale"
QUANT_SUFFIX = ".comfy_quant"
N_LAYERS = 36
WIDTH = 2560


def flat_key(key: str) -> str:
    """The ComfyUI file's layout for a text-tower or vision key, whatever
    nesting it arrived in."""
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            return "model." + key[len(prefix):]
    if key.startswith("visual."):
        return "model." + key
    if key.startswith(("layers.", "embed_tokens.", "norm.")):
        return "model." + key
    return key


def shard_files(path: Path) -> list[Path]:
    """The shards the index names, else every safetensors in the directory,
    else the single file given."""
    if path.is_file():
        return [path]
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()))
        shards = [path / n for n in names]
        missing = [str(s) for s in shards if not s.exists()]
        if missing:
            raise SystemExit(f"index names shards that are not here: {', '.join(missing)}")
        if shards:
            return shards
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors in {path}")
    return shards


def read_tensors(files: list[Path], keep=lambda k: True) -> tuple[dict, int, int]:
    """Every kept tensor from the files as {flat_key: bf16 tensor}. Returns
    the dict plus counts of fp8 matrices dequantized and keys dropped."""
    import torch
    from safetensors import safe_open

    out, scales, dropped = {}, {}, 0
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                name = flat_key(key)
                if name.endswith(QUANT_SUFFIX):
                    continue
                if name.endswith(SCALE_SUFFIX):
                    base = name[: -len(SCALE_SUFFIX)]
                    # A scale follows its weight: a filtered-out weight (the
                    # text tower when only the vision tower is wanted) must
                    # not leave its scale behind as an orphan.
                    if keep(f"{base}.weight"):
                        scales[base] = sf.get_tensor(key)
                    continue
                if not keep(name):
                    dropped += 1
                    continue
                out[name] = sf.get_tensor(key)
    # Only the scalar `.weight_scale` convention is understood. Anything else
    # that scales an fp8 weight (`weight_scale_inv` block scales, ComfyUI's
    # older `scale_weight`, `input_scale`) would pass through as a stray key
    # while its weight was cast without scaling: a file that check() passes
    # and that holds garbage. Refuse both signs of it.
    stray = sorted(k for k in out if "scale" in k.rsplit(".", 1)[-1])
    if stray:
        raise SystemExit(
            f"unsupported fp8 scale format: {', '.join(stray[:3])}"
            + (f" and {len(stray) - 3} more" if len(stray) > 3 else "")
            + ". Only per-tensor `.weight_scale` checkpoints can be packed; "
            "dequantize this one first, or use the bf16 release.")
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    unscaled = sorted(k for k, t in out.items()
                      if t.dtype in fp8 and k[: -len(".weight")] not in scales)
    if unscaled:
        raise SystemExit(f"fp8 weights with no `.weight_scale`: {', '.join(unscaled[:3])}"
                         + (f" and {len(unscaled) - 3} more" if len(unscaled) > 3 else ""))
    for base, scale in scales.items():
        key = f"{base}.weight"
        if key not in out:
            raise SystemExit(f"{base}: a weight_scale with no weight to scale")
        out[key] = (out[key].to(torch.float32) * scale.to(torch.float32))
    for key, t in out.items():
        if t.is_floating_point() and t.dtype != torch.bfloat16:
            out[key] = t.to(torch.bfloat16).contiguous()
    return out, len(scales), dropped


def check(state: dict) -> list[str]:
    """What ComfyUI's loader and the DiT will look for. Empty means fine."""
    problems = []
    for i in range(N_LAYERS):
        if f"model.layers.{i}.self_attn.q_proj.weight" not in state:
            problems.append(f"text layer {i} missing")
            break
    if f"model.layers.{N_LAYERS}.self_attn.q_proj.weight" in state:
        problems.append(f"more than {N_LAYERS} text layers: not a 4B")
    for key in ("model.embed_tokens.weight", "model.norm.weight",
                "model.visual.deepstack_merger_list.0.norm.weight",
                "model.visual.merger.linear_fc2.weight"):
        if key not in state:
            problems.append(f"{key} missing")
    merger = state.get("model.visual.merger.linear_fc2.weight")
    if merger is not None and merger.shape[0] != WIDTH:
        problems.append(f"vision merger is {merger.shape[0]} wide, not {WIDTH}: "
                        "this vision tower belongs to another size")
    embed = state.get("model.embed_tokens.weight")
    if embed is not None and embed.shape[1] != WIDTH:
        problems.append(f"embedding width {embed.shape[1]}, not {WIDTH}")
    return problems


def pack(source: Path, out: Path, vision_from: Path | None = None) -> dict:
    is_text = lambda k: k.startswith("model.") and not k.startswith("model.visual.")
    is_vision = lambda k: k.startswith("model.visual.")
    # lm_head is tied on the 4B; ComfyUI's config carries no head.
    state, n_deq, n_dropped = read_tensors(
        shard_files(source), keep=lambda k: (is_text(k) or is_vision(k)) and not k.startswith("model.lm_head"))
    n_vision = sum(1 for k in state if is_vision(k))
    spliced = 0
    if n_vision == 0:
        if vision_from is None:
            raise SystemExit(
                f"{source} carries no vision tower (model.visual.*), and ComfyUI "
                "recognizes Qwen3-VL by it; --vision-from names a checkpoint "
                "to take one from.")
        vision, _, _ = read_tensors(shard_files(vision_from), keep=is_vision)
        if not vision:
            raise SystemExit(f"{vision_from} carries no model.visual.* keys either")
        state.update(vision)
        spliced = len(vision)
    problems = check(state)
    if problems:
        raise SystemExit("refusing to write an unloadable file:\n  " + "\n  ".join(problems))
    from safetensors.torch import save_file

    out.parent.mkdir(parents=True, exist_ok=True)
    # Names only: a full path would publish the packing machine's layout to
    # anyone the file is shared with.
    save_file(state, str(out), metadata={
        "format": "pt",
        "packed_from": source.name,
        "vision_from": (vision_from if spliced else source).name,
        "packed_by": "tokencollider tools/pack_encoder.py",
    })
    n_bytes = sum(t.numel() * t.element_size() for t in state.values())
    return {"path": str(out), "tensors": len(state), "text": len(state) - n_vision - spliced,
            "vision": n_vision + spliced, "vision_spliced": spliced,
            "dequantized": n_deq, "dropped": n_dropped, "gib": n_bytes / 1024**3}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="HF shard directory (or one safetensors file)")
    ap.add_argument("out", help="single safetensors to write, for ComfyUI's models/text_encoders/")
    ap.add_argument("--vision-from", default=None,
                    help="checkpoint to take model.visual.* from when the source is text-only")
    args = ap.parse_args(argv)
    res = pack(Path(args.source), Path(args.out),
               Path(args.vision_from) if args.vision_from else None)
    print(f"[pack] wrote {res['path']}: {res['tensors']} tensors, {res['gib']:.2f} GiB bf16")
    print(f"[pack]   text tower {res['text']} keys, vision tower {res['vision']} keys"
          + (f" (spliced from --vision-from)" if res["vision_spliced"] else ""))
    if res["dequantized"]:
        print(f"[pack]   dequantized {res['dequantized']} fp8 matrices")
    if res["dropped"]:
        print(f"[pack]   dropped {res['dropped']} keys outside the two towers (lm_head, etc.)")


if __name__ == "__main__":
    main()
