"""Checks that need the real weights and a free GPU. Not part of the
model-free suite.

Covers, in one pass over each encoder:

- Z-Image (Qwen3-4B, bf16): hidden-state trace pinned by hash, cook() capture
  against a plain forward pass.
- fp8 arithmetic against ground truth: quantize the known-good bf16 weights the
  way the ComfyUI files do, dequantize with the loader's own arithmetic, and
  measure what it costs a 36-layer trace.
- Krea 2's encoder (Qwen3-VL-4B, fp8): loads, dequantizes, skips the vision
  tower, survives multimodal rotary position encoding in cook(), and produces a
  semantically real space.

Paths come from the environment so this survives a machine move:
TOKENCOLLIDER_ZIMAGE_MODEL, TOKENCOLLIDER_KREA2_MODEL, TOKENCOLLIDER_KREA2_CONFIG_DIR.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider import profiles
from tokencollider.embedder import Embedder
from tokencollider.store import EmbeddingStore

REPO = Path(__file__).resolve().parent.parent
ZIMAGE = os.environ.get("TOKENCOLLIDER_ZIMAGE_MODEL",
                        "/path/to/models/clip/qwen_3_4b.safetensors")
VL = os.environ.get("TOKENCOLLIDER_KREA2_MODEL", os.environ.get("TOKENCOLLIDER_VL_MODEL",
                    "/path/to/models/clip/qwen3vl_4b_fp8_scaled.safetensors"))
VL_CONFIG = os.environ.get(
    "TOKENCOLLIDER_KREA2_CONFIG_DIR",
    os.environ.get("TOKENCOLLIDER_VL_CONFIG_DIR",
                   "/path/to/models/qwen3_vl_4b_instruct_config"))

TEXT = "utter desolation"
CAPTURE = [4, 12, 20, 28, 35]
FP8_MAX = 448.0  # largest finite value of float8_e4m3fn
# Pinned from the bf16 Qwen3-4B at commit 63fda1b. A change here means the
# encoder path moved, which is either a bug or a decision worth recording.
ZIMAGE_TRACE_SHA = "8f9c7c12dc5e42fe691edfa43c6e530c9cd8ecb2ba0b80fb773c52117de341cc"

# Matrices the real fp8 files quantize; norms and embeddings stay bf16.
QUANT_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj")


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def trace_sha(states):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(s).tobytes() for s in states)).hexdigest()


def check_cook(emb, truth, label):
    """A resumed pass must reproduce the states a plain forward pass reports."""
    got = emb.cook_layers(truth[0], 0, None, stop_index=35, capture=CAPTURE)
    worst = min(cosine(got[d], truth[d]) for d in got)
    for d in sorted(got):
        print(f"    L{d:<3} cosine {cosine(got[d], truth[d]):.6f}")
    assert worst > 0.999, f"{label}: cook() diverged, worst cosine {worst}"
    print(f"  ok: {label} cook() capture matches the forward pass "
          f"(worst {worst:.6f})")


def test_zimage(store):
    print("\n=== Z-Image encoder (Qwen3-4B, bf16) ===")
    zimage = profiles.builtins()["zimage"]
    emb = Embedder(store, model_name=ZIMAGE, layer=zimage.layer,
                   template=zimage.template, config_repo=zimage.config_repo)
    assert emb.export_trim() == 0, "Z-Image is handed the whole sequence"
    truth = emb._forward_all(TEXT)
    sha = trace_sha(truth)
    print(f"  {len(truth)} states, seq {truth[0].shape[0]}, dim {truth[0].shape[1]}")
    assert sha == ZIMAGE_TRACE_SHA, f"trace moved: {sha} != {ZIMAGE_TRACE_SHA}"
    print(f"  ok: hidden-state trace unchanged ({sha[:16]}...)")
    check_cook(emb, truth, "Z-Image")
    return emb, truth


def test_fp8_roundtrip(emb, truth):
    """Ground truth for the dequant arithmetic: quantize the known-good bf16
    weights the way the files do, dequantize the way the loader does."""
    print("\n=== fp8 dequant against ground truth ===")
    n, worst_matrix = 0, 0.0
    with torch.no_grad():
        for name, param in emb._model.named_parameters():
            if param.dim() != 2 or not any(t in name for t in QUANT_TARGETS):
                continue
            w = param.data.to(torch.float32)
            scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12)
            q = (w / scale).to(torch.float8_e4m3fn)          # what the file stores
            deq = q.to(torch.float32) * scale.to(torch.float32)  # what we do
            worst_matrix = max(worst_matrix,
                               float((deq - w).norm() / w.norm()))
            param.data.copy_(deq.to(param.dtype))
            n += 1
    print(f"  quantized {n} matrices, worst relative error {worst_matrix:.5f}")
    assert n == 252, f"expected 252 matrices (36 x 7), got {n}"

    got = emb._forward_all(TEXT)
    worst = 1.0
    for d in (0, 12, 20, 35, 36):
        c = cosine(got[d], truth[d])
        worst = min(worst, c)
        note = "  <- post-norm, the most fp8-sensitive depth" if d == 36 else ""
        print(f"    L{d:<3} cosine {c:.6f}{note}")
    assert worst > 0.99, f"fp8 round-trip degraded the trace: {worst}"

    # Direction is not a coin flip: dividing overshoots by orders of magnitude.
    w = torch.randn(64, 64) * 0.03
    scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12)
    q = (w / scale).to(torch.float8_e4m3fn)
    assert abs(float((q.to(torch.float32) * scale).abs().max())
               - float(w.abs().max())) < 1e-3
    assert float((q.to(torch.float32) / scale).abs().max()) > 1e5
    print(f"  ok: weight * weight_scale recovers the trace (worst {worst:.6f})")


def test_vl(store):
    print("\n=== Krea 2 encoder (Qwen3-VL-4B, fp8) ===")
    emb = Embedder(store, model_name=VL, layer="35", template="an image of {}",
                   config_dir=VL_CONFIG)
    truth = emb._forward_all(TEXT)
    assert type(emb._model).__name__ == "Qwen3VLTextModel", type(emb._model)
    assert len(truth) == 37, len(truth)
    for state in truth:
        assert np.isfinite(state).all()
    print(f"  {len(truth)} states, dim {truth[0].shape[1]}, "
          f"class {type(emb._model).__name__}")

    # cook() builds a flat arange for position_ids. Qwen3-VL's rotary
    # embedding broadcasts a 2D position_ids across all three mRoPE
    # components, which is exactly right for a text-only sequence: the real
    # get_rope_index gives text tokens identical t/h/w positions. Image
    # tokens are where that stops being true, and that is the image-landmark
    # project's problem, not this one.
    check_cook(emb, truth, "Qwen3-VL")
    return emb


def test_vl_semantics(emb):
    """Finite numbers prove the loader works; geometry proves the weights do."""
    print("\n=== Qwen3-VL geometry ===")
    words = ["cat", "kitten", "hammer", "dog", "puppy", "sorrow", "grief",
             "joy", "wrench", "screwdriver", "cardboard box", "packing tape",
             "chocolate cake"]
    pairs = [("cat", "kitten", "hammer"), ("dog", "puppy", "sorrow"),
             ("hammer", "wrench", "joy"), ("sorrow", "grief", "screwdriver"),
             ("cardboard box", "packing tape", "chocolate cake")]
    vecs = emb.embed_many(words)
    vecs = vecs - vecs.mean(axis=0)  # universe centering, as the tool does
    idx = {w: i for i, w in enumerate(words)}
    wins = 0
    for q, near, far in pairs:
        cn, cf = cosine(vecs[idx[q]], vecs[idx[near]]), cosine(vecs[idx[q]], vecs[idx[far]])
        wins += cn > cf
        print(f"    {q:<15}{near:<15}{cn:>7.3f}   {far:<15}{cf:>7.3f}")
    assert wins == len(pairs), f"geometry is not semantic: {wins}/{len(pairs)}"
    print(f"  ok: {wins}/{len(pairs)} related pairs beat unrelated ones")


def test_krea2_profile(store):
    """The whole layer-stack path, end to end on the real encoder: Krea 2's
    verified template and its twelve sampler depths, blended and cooked."""
    from tokencollider import profiles
    from tokencollider.conditioning import Conditioning
    from tokencollider.layout import LayerStack
    from tokencollider.server import export_conditioning

    print("\n=== Krea 2 profile, twelve-layer export ===")
    profile = profiles.get("krea2")
    emb = Embedder(store, model_name=VL, layer=profile.layer,
                   template=profile.template, config_dir=VL_CONFIG,
                   sampler_layers=profile.sampler_layers,
                   trim_template_prefix=profile.trim_template_prefix)
    assert emb.sampler_layers() == (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    # ComfyUI finds the <|im_start|> opening the user turn and adds 3. For this
    # scaffold that is 34, which is exactly the prefix length.
    assert emb.template_prefix_tokens() == 34, emb.template_prefix_tokens()
    assert emb.export_trim() == 34

    words = ["cardboard box", "packing tape", "chocolate cake", "coffee beans",
             "wooden crate", "brown paper"]
    stack = LayerStack(emb)
    stack.add_landmarks(words)
    coords = [0.3, -0.2, 0.1, 0.05, -0.05, 0.0]

    with tempfile.TemporaryDirectory() as d:
        flat = export_conditioning(stack, None, coords,
                                   str(Path(d) / "flat.safetensors"),
                                   "safetensors", None)
        assert flat["layers"] == list(profile.sampler_layers), flat["layers"]
        assert flat["cooked"] is False
        cond, meta = Conditioning.load(flat["path"])
        assert cond.layers == profile.sampler_layers
        assert cond.dim == 2560
        # The export lands in ComfyUI's frame: prompt tokens onward, with the
        # trailing turn scaffold kept and the 34-token preamble dropped.
        untrimmed = max(emb.conditioning_layers(w, [35])[35].shape[0]
                        for w in words)
        assert cond.seq_len == untrimmed - 34, (cond.seq_len, untrimmed)
        assert meta["template_prefix_tokens"] == "34"
        print(f"  flat export: {len(cond)} depths {list(cond.layers)}, "
              f"seq {cond.seq_len}, dim {cond.dim}")

        # A band view at 14-20 sits inside the projector's positive region.
        # Depths above 20 cook from the injection; 2..20 are upstream of it
        # and stay a direct blend.
        cooked = export_conditioning(stack, (14, 20), coords,
                                     str(Path(d) / "cook.safetensors"),
                                     "safetensors", (14, 20))
        assert cooked["cooked"] is True and cooked["band"] == [14, 20]
        assert cooked["layers"] == list(profile.sampler_layers)
        stacked, _ = Conditioning.load(cooked["path"])

        session = stack.session((14, 20))
        weights = list(session.blend_weights(coords)["weights"].values())
        upstream = [d for d in profile.sampler_layers if d <= 20]
        downstream = [d for d in profile.sampler_layers if d > 20]
        # The reference blend is untrimmed, so compare against its tail: the
        # export trims last, after cooking, which is the whole point.
        blended = session._blend_at(upstream + downstream, weights)
        trim = emb.export_trim()
        for depth in upstream:
            assert np.allclose(stacked[depth], blended[depth][trim:]), depth
        for depth in downstream:
            assert not np.allclose(stacked[depth], blended[depth][trim:]), depth
        assert stacked.seq_len == blended[upstream[0]].shape[0] - trim
        print(f"  cooked export: {upstream} blended upstream, "
              f"{downstream} cooked from the L20 injection")

    # The scaffold conditions attention without contributing tokens: pooling
    # covers the phrase span alone, so a longer template must not change the
    # token count the phrase occupies.
    start, end = emb._span("cardboard box")
    print(f"  phrase span inside the scaffold: tokens [{start}, {end})")
    assert end - start == 3, (start, end)
    assert start == emb.template_prefix_tokens()
    # A caption-less image: the conditioning is the scaffold alone, trimmed
    # to the five tail tokens, the same as the sampler would encode "".
    empty = emb.conditioning("")
    assert empty.seq_len == emb.template_prefix_tokens() + emb.template_tail_tokens(), empty.seq_len
    assert empty.trim(emb.export_trim()).seq_len == emb.template_tail_tokens()
    print("  ok: Krea 2 profile exports a verified twelve-depth stack")
    return emb


def test_krea2_images(store):
    """Image landmarks through the vision tower, inside the same scaffold a
    phrase gets, under both poolings, and the question the layer stack is
    for: at which depth do pictures and words come closest."""
    from tokencollider import images, profiles
    from tokencollider.conditioning import Conditioning

    print("\n=== Krea 2 image landmarks (vision tower) ===")
    from PIL import Image, ImageDraw
    profile = profiles.get("krea2")
    results = {}
    with tempfile.TemporaryDirectory() as d:
        # Two synthetic pictures with a nameable difference: a red disc on
        # white, a blue square on black.
        red = Image.new("RGB", (320, 240), "white")
        ImageDraw.Draw(red).ellipse((90, 50, 230, 190), fill=(220, 30, 30))
        blue = Image.new("RGB", (320, 240), "black")
        ImageDraw.Draw(blue).rectangle((90, 50, 230, 190), fill=(30, 60, 220))
        red.save(Path(d) / "red_circle.png"); blue.save(Path(d) / "blue_square.png")
        words = ["a red circle", "a blue square", "a cardboard box"]
        for pooling in ("image", "tail"):
            emb = Embedder(store, model_name=VL, layer=profile.layer,
                           template=profile.template, config_dir=VL_CONFIG,
                           sampler_layers=profile.sampler_layers,
                           trim_template_prefix=profile.trim_template_prefix,
                           image_pooling=pooling)
            keys = images.register_many(store, [Path(d) / "red_circle.png",
                                                Path(d) / "blue_square.png"], pooling)
            assert all(images.key_mode(k) == pooling for k in keys)
            states, span, frame = emb._forward_image_all(keys[0])
            assert len(states) == 37 and all(np.isfinite(s).all() for s in states)
            n_img = span[1] - span[0] if pooling == "image" else None
            print(f"  [{pooling}] seq {states[0].shape[0]} tokens, pooled span {span}, "
                  f"frame seq {frame[0].shape[0]}")
            if pooling == "image":
                # 512 px square at patch 16, merge 2: 16 x 16 tokens, for
                # every picture regardless of its file's shape.
                assert n_img == (images.IMAGE_SIZE // 32) ** 2, n_img
                assert frame[0].shape[0] == states[0].shape[0]
            else:
                assert frame[0].shape[0] == emb.template_prefix_tokens() + (span[1] - span[0])
            # Pooled vectors at every depth for pictures and words alike.
            vecs = {k: [emb.embed_layer(k, l) for l in range(37)] for k in keys + words}
            # The sweep: cosine between each picture and its own phrase, per
            # depth, against the unrelated phrase.
            print(f"  [{pooling}] depth  red~'red circle'  red~'box'   blue~'blue square'  blue~'box'")
            for l in (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 36):
                r = cosine(vecs[keys[0]][l], vecs[words[0]][l]); rb = cosine(vecs[keys[0]][l], vecs[words[2]][l])
                b = cosine(vecs[keys[1]][l], vecs[words[1]][l]); bb = cosine(vecs[keys[1]][l], vecs[words[2]][l])
                print(f"  [{pooling}] L{l:<3}   {r:+.3f}          {rb:+.3f}      {b:+.3f}            {bb:+.3f}")
            # The two pictures are distinguishable from each other at the
            # charted depth, and the export frame carries every tap.
            assert cosine(vecs[keys[0]][20], vecs[keys[1]][20]) < 0.9999
            cond = emb.conditioning(keys[0])
            assert cond.layers == profile.sampler_layers
            results[pooling] = cond.seq_len
            emb.release()
    print(f"  ok: image landmarks embed under both poolings "
          f"(conditioning seq: {results})")

    # A mixed cursor: one picture, one phrase. The export keeps them in
    # separate blocks, and a cook over the mixed frame runs with Krea 2's
    # 3-D positions for the picture block.
    from tokencollider.layout import LayerStack
    from tokencollider.server import export_conditioning
    with tempfile.TemporaryDirectory() as d:
        red = Image.new("RGB", (320, 240), "white")
        ImageDraw.Draw(red).ellipse((90, 50, 230, 190), fill=(220, 30, 30))
        red.save(Path(d) / "red.png")
        emb = Embedder(store, model_name=VL, layer=profile.layer,
                       template=profile.template, config_dir=VL_CONFIG,
                       sampler_layers=profile.sampler_layers,
                       trim_template_prefix=profile.trim_template_prefix,
                       image_pooling="image")
        pic = images.register(store, Path(d) / "red.png", "image")
        # First: the cook reproduces a real forward over an image frame when
        # given the 3-D positions (from block 4 on, past DeepStack).
        truth, _span, _frame = emb._forward_image_all(pic)
        n_patch = (images.IMAGE_SIZE // 32) ** 2
        P, T = emb.template_prefix_tokens(), emb.template_tail_tokens()
        pos = emb.mixed_position_ids([("text", P), ("image", n_patch + 2), ("text", T)])
        got = emb.cook_layers(truth[4], 4, None, stop_index=35, capture=[12, 20, 35],
                              position_ids=pos)
        worst = min(cosine(got[k], truth[k]) for k in got)
        print(f"  image-frame cook with 3-D positions: worst cosine {worst:.6f}")
        assert worst > 0.999, worst
        flat_pos = emb.cook_layers(truth[4], 4, None, stop_index=35, capture=[35])
        print(f"  ...and with flat positions (wrong for patches): "
              f"{cosine(flat_pos[35], truth[35]):.6f}")

        stack = LayerStack(emb)
        stack.add_landmarks([pic, "a red circle", "a blue square"])
        coords = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        flat = export_conditioning(stack, None, coords, str(Path(d) / "mix.safetensors"),
                                   "safetensors", None)
        cond, meta = Conditioning.load(flat["path"])
        frame = json.loads(meta["frame"])
        print(f"  mixed flat export: frame {frame}, seq {cond.seq_len} after trim")
        assert frame[0] == ["prefix", P] and frame[1] == ["image", n_patch + 2]
        assert frame[-1] == ["tail", T]
        assert cond.seq_len == (n_patch + 2) + frame[2][1] + T
        cooked = export_conditioning(stack, (14, 20), coords, str(Path(d) / "mixc.safetensors"),
                                     "safetensors", (14, 20))
        c2, _ = Conditioning.load(cooked["path"])
        assert cooked["cooked"] is True and c2.seq_len == cond.seq_len
        assert all(np.isfinite(c2[l]).all() for l in c2.layers)
        print(f"  mixed cooked export from 20: seq {c2.seq_len}, taps {list(c2.layers)}")
        emb.release()
    print("  ok: mixed-modality export and 3-D-position cook")


def test_template_boundary(config_dir):
    """The export trim is only correct if no BPE merge crosses the phrase slot.

    `Embedder.template_prefix_tokens` counts the template's prefix ONCE and
    trims that many tokens off every export. That is right only while the
    phrase cannot change how the prefix tokenizes. A special token precedes
    the slot, so it should not, but "should not" is what a test is for: a
    merge across the boundary would silently shift every export by a token and
    nothing downstream would notice.

    Tokenizer only, no weights and no GPU.
    """
    from transformers import AutoTokenizer

    from tokencollider.profiles import load

    prof = load()["krea2"]
    tok = AutoTokenizer.from_pretrained(config_dir)
    pre, suf = prof.template.split("{}")
    n_pre = len(tok(pre, add_special_tokens=False)["input_ids"])
    n_suf = len(tok(suf, add_special_tokens=False)["input_ids"])
    ref_pre = tok(pre, add_special_tokens=False)["input_ids"]
    ref_suf = tok(suf, add_special_tokens=False)["input_ids"]
    # ComfyUI's krea2 encoder finds template_end dynamically; we take a fixed
    # count. They must agree, and the fixed count must hold for any phrase.
    assert n_pre == 34 and n_suf == 5, (n_pre, n_suf)

    cases = ["marble", "a", "", " ", "ЖЕЛЕЗО", "日本語", "naïve café",
             "  leading space", "trailing space  ", "<|im_end|>", "🜛 alchemical",
             "123", "-", "hyphen-word", "a very long phrase with many words"]
    for c in cases:
        ids = tok(prof.template.format(c), add_special_tokens=False)["input_ids"]
        assert ids[:n_pre] == ref_pre, f"prefix shifted for {c!r}"
        assert ids[len(ids) - n_suf:] == ref_suf, f"suffix shifted for {c!r}"
        assert len(ids) >= n_pre + n_suf, (c, len(ids))
    print(f"ok: template boundary stable over {len(cases)} phrases "
          f"(prefix {n_pre}, suffix {n_suf}, matches ComfyUI's template_end)")


if __name__ == "__main__":
    for path in (ZIMAGE, VL):
        if not Path(path).exists():
            raise SystemExit(f"[tokencollider] missing weights: {path}")
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "real.db")
        emb, truth = test_zimage(store)
        test_fp8_roundtrip(emb, truth)
        emb.release()  # hand the VRAM back before the second encoder
        vl = test_vl(store)
        test_vl_semantics(vl)
        vl.release()
        krea = test_krea2_profile(store)
        krea.release()
        test_krea2_images(store)
        test_template_boundary(os.environ.get(
            "TOKENCOLLIDER_KREA2_CONFIG_DIR", "/path/to/models/Qwen3-VL-4B-Instruct"))
        store.close()
    print("\nall real-model checks passed")
