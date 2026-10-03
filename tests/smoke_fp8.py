"""Model-free tests for the fp8 checkpoint loader: key handling, scale
application, and the guards around loading a VLM tower under the wrong config.

The arithmetic itself is validated against the real Qwen3-4B in
tests/real_model.py; these cover the parts a GPU-free run can reach.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider.embedder import Embedder
from tokencollider.store import EmbeddingStore

FP8_MAX = 448.0  # largest finite value of float8_e4m3fn
DESCRIPTOR = b'{"format": "float8_e4m3fn", "full_precision_matrix_mult": false}'


def write_checkpoint(path, *, quantized=True, with_vision=False,
                     orphan_scale=False, lm_head=None):
    """A miniature of the real ComfyUI-style file: 'model.'-prefixed keys, a
    quantized matrix as weight + weight_scale + comfy_quant, and BF16 norms."""
    from safetensors.torch import save_file

    rng = np.random.default_rng(0)
    w = torch.from_numpy((rng.standard_normal((16, 8)) * 0.03).astype(np.float32))
    tensors = {"model.norm.weight": torch.ones(8, dtype=torch.bfloat16)}
    if quantized:
        scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12)
        tensors["model.layers.0.self_attn.q_proj.weight"] = (
            (w / scale).to(torch.float8_e4m3fn))
        tensors["model.layers.0.self_attn.q_proj.weight_scale"] = scale.float()
        tensors["model.layers.0.self_attn.q_proj.comfy_quant"] = torch.frombuffer(
            bytearray(DESCRIPTOR.ljust(64, b"\x00")), dtype=torch.uint8)
        truth = ((w / scale).to(torch.float8_e4m3fn).to(torch.float32)
                 * scale).numpy()
    else:
        tensors["model.layers.0.self_attn.q_proj.weight"] = w.to(torch.bfloat16)
        truth = w.to(torch.bfloat16).to(torch.float32).numpy()
    if orphan_scale:
        tensors["model.layers.0.mlp.up_proj.weight_scale"] = torch.tensor(2.0)
    if lm_head:  # an untied output head, as HF checkpoints of most LLMs carry
        tensors[lm_head] = torch.zeros(32, 8, dtype=torch.bfloat16)
    if with_vision:
        tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(
            4, 4, dtype=torch.bfloat16)
        tensors["model.visual.merger.linear_fc2.weight"] = torch.zeros(
            4, 4, dtype=torch.bfloat16)
    save_file(tensors, str(path))
    return truth


def make_embedder(path):
    store = EmbeddingStore(Path(path).parent / "t.db")
    # The config source a profile would name; nothing is assumed by default.
    return Embedder(store, model_name=str(path), layer="35", device="cpu",
                    config_repo="Qwen/Qwen3-4B"), store


def test_dequantizes_and_strips_prefix():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "q.safetensors"
        truth = write_checkpoint(path, quantized=True)
        emb, store = make_embedder(path)
        state, n_deq, n_skip = emb._read_state(torch.bfloat16)

        assert n_deq == 1 and n_skip == 0
        # 'model.' stripped, and the descriptor is not a weight.
        assert set(state) == {"norm.weight", "layers.0.self_attn.q_proj.weight"}, \
            sorted(state)
        got = state["layers.0.self_attn.q_proj.weight"]
        assert got.dtype == torch.bfloat16
        # The scale is applied, so the weight lands at its true magnitude
        # rather than ~448x too large.
        rel = float(np.linalg.norm(got.float().numpy() - truth)
                    / np.linalg.norm(truth))
        assert rel < 0.01, rel
        assert abs(float(got.abs().max()) - float(np.abs(truth).max())) < 1e-3
        store.close()
    print("ok: fp8 dequantized, prefix stripped, descriptor dropped")


def test_unquantized_file_unchanged():
    """A plain bf16 checkpoint must load exactly as it always did."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.safetensors"
        truth = write_checkpoint(path, quantized=False)
        emb, store = make_embedder(path)
        state, n_deq, n_skip = emb._read_state(torch.bfloat16)
        assert n_deq == 0 and n_skip == 0
        got = state["layers.0.self_attn.q_proj.weight"].float().numpy()
        assert np.array_equal(got, truth)
        store.close()
    print("ok: unquantized checkpoints load untouched")


def test_untied_lm_head_is_dropped():
    """The text tower has no output head, so an untied `lm_head.weight` (most
    HF checkpoints that don't tie embeddings) must be skipped, not handed to
    load_state_dict as an unexpected key."""
    for key in ("lm_head.weight", "model.lm_head.weight"):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "h.safetensors"
            write_checkpoint(path, quantized=False, lm_head=key)
            emb, store = make_embedder(path)
            state, _n_deq, n_skip = emb._read_state(torch.bfloat16)
            assert not any("lm_head" in k for k in state), (key, sorted(state))
            assert n_skip == 1, (key, n_skip)
            store.close()
    print("ok: an untied lm_head is skipped, in either spelling")


def test_orphan_scale_is_loud():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "o.safetensors"
        write_checkpoint(path, quantized=True, orphan_scale=True)
        emb, store = make_embedder(path)
        try:
            emb._read_state(torch.bfloat16)
            raise AssertionError("expected a refusal on a scale with no weight")
        except RuntimeError as e:
            assert "no weight to scale" in str(e), e
        store.close()
    print("ok: a weight_scale with no weight fails loudly")


def test_skip_prefixes():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.safetensors"
        write_checkpoint(path, quantized=True, with_vision=True)
        emb, store = make_embedder(path)
        state, _n_deq, n_skip = emb._read_state(
            torch.bfloat16, skip_prefixes=("model.visual.",))
        assert n_skip == 2, n_skip
        assert not any("visual" in k for k in state), sorted(state)
        # Without the skip list those keys arrive and would trip the strict
        # state-dict check, which is the point: they are dropped on purpose,
        # never tolerated silently.
        state, _n, n_skip = emb._read_state(torch.bfloat16)
        assert n_skip == 0 and any("visual" in k for k in state)
        store.close()
    print("ok: sibling-tower keys are skipped only when asked")


def test_vision_config_mismatch_guard():
    """A VLM text tower has the same shapes as its text-only sibling, so a
    mismatched config would load clean and be wrong. Refuse instead."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.safetensors"
        write_checkpoint(path, quantized=True, with_vision=True)
        emb, store = make_embedder(path)
        assert any(k.startswith("model.visual.") for k in emb._peek_keys())
        try:
            emb._load_local_safetensors(torch.bfloat16)
            raise AssertionError("expected a refusal on a vision/config mismatch")
        except SystemExit as e:
            assert "carries a vision tower" in str(e), e
        store.close()
    print("ok: vision tower under a text-only config is refused")


def test_sharded_directory():
    """Weights may arrive as a directory of HF-style shards, in either dtype.
    A bf16 finetune therefore needs no conversion to be loadable."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        shards = Path(d) / "model"
        shards.mkdir()
        # Shard 1 quantized, shard 2 plain, to prove both survive one read.
        write_checkpoint(shards / "model-00001-of-00002.safetensors",
                         quantized=True)
        save_file({"model.layers.1.mlp.up_proj.weight":
                   torch.ones(4, 4, dtype=torch.bfloat16)},
                  str(shards / "model-00002-of-00002.safetensors"))
        emb, store = make_embedder(shards / "unused.safetensors")
        emb.model_name = str(shards)
        assert len(emb.weight_files()) == 2
        state, n_deq, _ = emb._read_state(torch.bfloat16)
        assert n_deq == 1
        assert "layers.0.self_attn.q_proj.weight" in state
        assert "layers.1.mlp.up_proj.weight" in state
        assert "norm.weight" in state
        # _peek_keys unions the shards rather than reading only the first.
        keys = emb._peek_keys()
        assert any("layers.1" in k for k in keys) and any("layers.0" in k for k in keys)
        store.close()
    print("ok: sharded weight directories (mixed dtypes) load as one")


def test_shard_index_wins_over_globbing():
    """An HF directory names its shards in model.safetensors.index.json.
    Globbing instead would sweep up anything else parked in the folder."""
    import json as _json
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "finetune"
        m.mkdir()
        for i in (1, 2):
            save_file({f"model.language_model.layers.{i}.mlp.up_proj.weight":
                       torch.zeros(4, 4, dtype=torch.bfloat16)},
                      str(m / f"model-0000{i}-of-00002.safetensors"))
        save_file({"junk": torch.zeros(2, 2)}, str(m / "some_lora.safetensors"))
        (m / "model.safetensors.index.json").write_text(_json.dumps({"weight_map": {
            f"model.language_model.layers.{i}.mlp.up_proj.weight":
            f"model-0000{i}-of-00002.safetensors" for i in (1, 2)}}), encoding="utf-8")

        emb, store = make_embedder(m / "unused.safetensors")
        emb.model_name = str(m)
        assert [f.name for f in emb.weight_files()] == [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        state, _, _ = emb._read_state(torch.bfloat16)
        assert sorted(state) == ["layers.1.mlp.up_proj.weight",
                                 "layers.2.mlp.up_proj.weight"], sorted(state)
        assert "junk" not in state

        # A self-contained HF directory is its own config source.
        assert emb.config_source() != str(m)  # no config.json yet
        (m / "config.json").write_text('{"model_type": "qwen3_vl"}', encoding="utf-8")
        assert emb.config_source() == str(m)

        # An index naming a shard that is not there fails loudly.
        (m / "model.safetensors.index.json").write_text(_json.dumps(
            {"weight_map": {"x": "model-00009-of-00002.safetensors"}}), encoding="utf-8")
        try:
            emb.weight_files()
            raise AssertionError("expected a refusal on a missing shard")
        except SystemExit as e:
            assert "not here" in str(e), e
        store.close()
    print("ok: shard index beats globbing; self-contained dirs carry config")


def test_nested_tower_prefix():
    """A stock HF VLM nests the text tower under model.language_model.*;
    ComfyUI's files flatten it to model.*. Both mean the same parameter."""
    from tokencollider.embedder import strip_tower_prefix

    flat = "model.layers.0.self_attn.q_proj.weight"
    nested = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert strip_tower_prefix(flat) == strip_tower_prefix(nested)
    assert strip_tower_prefix(flat) == "layers.0.self_attn.q_proj.weight"
    assert strip_tower_prefix("model.embed_tokens.weight") == "embed_tokens.weight"
    assert strip_tower_prefix("norm.weight") == "norm.weight"
    print("ok: flat and nested text-tower prefixes agree")


def test_text_tower_dispatch():
    """The class comes from transformers' registry, not a hard-wired pair:
    the two built-in encoders resolve to exactly the classes they always
    used, and another decoder family resolves to its own."""
    from transformers import (LlamaConfig, LlamaModel, PretrainedConfig,
                              Qwen3Config, Qwen3Model, Qwen3VLConfig,
                              Qwen3VLTextModel)

    from tokencollider.embedder import VISION_PREFIXES

    cls, cfg, skip = Embedder._text_tower(Qwen3Config())
    assert cls is Qwen3Model and skip == ()

    vl = Qwen3VLConfig()
    cls, cfg, skip = Embedder._text_tower(vl)
    assert cls is Qwen3VLTextModel and cfg is vl.text_config
    assert skip == VISION_PREFIXES and "model.visual." in skip

    cls, cfg, skip = Embedder._text_tower(LlamaConfig())
    assert cls is LlamaModel and skip == ()

    class Unknown(PretrainedConfig):
        model_type = "not_a_real_model"
    try:
        Embedder._text_tower(Unknown())
        raise AssertionError("expected a refusal on an unknown model type")
    except SystemExit as e:
        assert "not_a_real_model" in str(e), e
    print("ok: text-tower dispatch through the transformers registry")


def test_vram_needed_scales_with_the_model():
    """The free-VRAM gate is sized from the checkpoint, not fixed at one
    model's 10 GB: weights that load, at bf16 whatever their stored
    precision, plus headroom. fp8 scales, a vision tower the text load
    leaves behind, and lm_head do not count."""
    from safetensors.torch import save_file

    from tokencollider.embedder import VRAM_HEADROOM_BYTES

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.safetensors"
        save_file({
            "model.layers.0.mlp.up_proj.weight": torch.zeros(64, 32).to(torch.float8_e4m3fn),
            "model.layers.0.mlp.up_proj.weight_scale": torch.tensor(1.0),
            "model.norm.weight": torch.zeros(32, dtype=torch.float32),
            "model.visual.blocks.0.weight": torch.zeros(16, 16, dtype=torch.bfloat16),
            "lm_head.weight": torch.zeros(100, 32, dtype=torch.bfloat16),
        }, str(path))
        emb, store = make_embedder(path)
        text = (64 * 32 + 32) * 2
        assert emb.vram_needed() == text + VRAM_HEADROOM_BYTES, emb.vram_needed()
        assert emb.vram_needed(vision=True) == text + 16 * 16 * 2 + VRAM_HEADROOM_BYTES
        store.close()
    print("ok: the VRAM gate is sized from the checkpoint")


def test_peek_keys_reads_header_only():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "q.safetensors"
        write_checkpoint(path, quantized=True)
        emb, store = make_embedder(path)
        keys = emb._peek_keys()
        assert "model.layers.0.self_attn.q_proj.weight_scale" in keys
        assert "__metadata__" not in keys
        # Header-only: the declared header length is far short of the file.
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            json.loads(f.read(header_len))  # parses cleanly on its own
        assert 8 + header_len < path.stat().st_size
        store.close()
    print("ok: _peek_keys reads the header, not the weights")


def test_pack_encoder():
    """HF shards of a Qwen3-VL-4B finetune become one ComfyUI-layout file:
    text keys flattened, lm_head dropped, bf16, vision tower present (spliced
    from the base when the finetune is text-only), and a file ComfyUI would
    not recognise is refused."""
    import importlib.util
    import json

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location(
        "packer", Path(__file__).resolve().parent.parent / "tools" / "pack_encoder.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    W = packer.WIDTH

    def text_tower(prefix):
        sd = {f"{prefix}embed_tokens.weight": torch.zeros(8, W, dtype=torch.float32),
              f"{prefix}norm.weight": torch.ones(W)}
        for i in range(packer.N_LAYERS):
            sd[f"{prefix}layers.{i}.self_attn.q_proj.weight"] = torch.full((4, W), float(i), dtype=torch.float16)
        return sd

    vision = {"model.visual.deepstack_merger_list.0.norm.weight": torch.ones(4),
              "model.visual.merger.linear_fc2.weight": torch.zeros(W, 8)}

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # A full VLM finetune: HF nesting, two shards, a tied lm_head to drop.
        full = d / "full"; full.mkdir()
        a = text_tower("model.language_model.")
        a["lm_head.weight"] = torch.zeros(8, W)
        b = dict(vision)
        save_file(a, str(full / "a.safetensors")); save_file(b, str(full / "b.safetensors"))
        (full / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {**{k: "a.safetensors" for k in a}, **{k: "b.safetensors" for k in b}}}), encoding="utf-8")
        res = packer.pack(full, d / "full.safetensors")
        with safe_open(str(d / "full.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            assert "model.layers.35.self_attn.q_proj.weight" in keys
            assert "model.embed_tokens.weight" in keys and "model.norm.weight" in keys
            assert "model.visual.merger.linear_fc2.weight" in keys
            assert not any("language_model" in k or "lm_head" in k for k in keys), keys
            assert f.get_tensor("model.layers.3.self_attn.q_proj.weight").dtype == torch.bfloat16
            assert f.metadata()["packed_by"].startswith("tokencollider")
        assert res["dropped"] == 1 and res["vision_spliced"] == 0, res

        # A text-only finetune needs the vision tower from the base.
        text_only = d / "text"; text_only.mkdir()
        save_file(text_tower("model."), str(text_only / "m.safetensors"))
        try:
            packer.pack(text_only, d / "nope.safetensors")
            raise AssertionError("expected a refusal without a vision tower")
        except SystemExit as e:
            assert "--vision-from" in str(e), e
        # The base is fp8-scaled like ComfyUI's own file: its text scales
        # must not survive the vision-only filter as orphans.
        base = d / "base.safetensors"
        scaled = text_tower("model.")
        q = "model.layers.0.self_attn.q_proj.weight"
        scaled[q] = scaled[q].to(torch.float8_e4m3fn)
        scaled[q + "_scale"] = torch.tensor(0.5)
        save_file({**scaled, **vision}, str(base))
        res = packer.pack(text_only, d / "spliced.safetensors", vision_from=base)
        assert res["vision_spliced"] == 2, res
        with safe_open(str(d / "spliced.safetensors"), framework="pt") as f:
            assert "model.visual.deepstack_merger_list.0.norm.weight" in f.keys()
            # file names only, never the packing machine's paths
            meta = f.metadata()
            assert (meta["packed_from"], meta["vision_from"]) == ("text", "base.safetensors"), meta

        # A wrong-size vision tower is refused rather than written.
        wrong = d / "wrong.safetensors"
        save_file({"model.visual.deepstack_merger_list.0.norm.weight": torch.ones(4),
                   "model.visual.merger.linear_fc2.weight": torch.zeros(4096, 8)}, str(wrong))
        try:
            packer.pack(text_only, d / "nope2.safetensors", vision_from=wrong)
            raise AssertionError("expected a refusal on a mismatched vision tower")
        except SystemExit as e:
            assert "another size" in str(e), e
    print("ok: pack_encoder merges shards into ComfyUI's layout (and refuses junk)")


def test_unsupported_fp8_formats_are_refused():
    """Only per-tensor `.weight_scale` fp8 is understood. Block scales
    (`weight_scale_inv`), ComfyUI's older `scale_weight`, and fp8 weights
    with no scale at all must be refused, not cast to bf16 unscaled, which
    gives a file that loads and holds garbage."""
    import importlib.util
    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location(
        "packer", Path(__file__).resolve().parent.parent / "tools" / "pack_encoder.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)

    w = torch.zeros(16, 8).to(torch.float8_e4m3fn)
    cases = {
        "block scales": {"model.layers.0.mlp.up_proj.weight": w,
                         "model.layers.0.mlp.up_proj.weight_scale_inv": torch.ones(1, 1)},
        "old comfy scale": {"model.layers.0.mlp.up_proj.weight": w,
                            "model.layers.0.mlp.up_proj.scale_weight": torch.tensor(2.0)},
        "no scale": {"model.layers.0.mlp.up_proj.weight": w},
    }
    with tempfile.TemporaryDirectory() as d:
        for what, tensors in cases.items():
            path = Path(d) / f"{what.replace(' ', '_')}.safetensors"
            save_file(tensors, str(path))
            try:
                packer.read_tensors([path])
                raise AssertionError(f"pack_encoder accepted {what}")
            except SystemExit as e:
                assert "scale" in str(e), (what, e)
        # The text tower's own loader: no scale at all has no other tell.
        emb, store = make_embedder(Path(d) / "no_scale.safetensors")
        try:
            emb._read_state(torch.bfloat16)
            raise AssertionError("the embedder loaded unscaled fp8")
        except RuntimeError as e:
            assert "weight_scale" in str(e), e
        store.close()
    print("ok: unsupported fp8 scale formats are refused, not cast unscaled")


def test_fp32_checkpoint_peaks_at_target_size():
    """Each tensor is cast to the target dtype as it is read. Holding a whole
    fp32 checkpoint before casting peaked at twice what the VRAM gate allowed
    for. Measured on the GPU, where the allocator records the peak; skipped
    without one."""
    if not torch.cuda.is_available():
        print("skip: fp32 load peak (no CUDA device)")
        return
    from safetensors.torch import save_file

    n, shape = 8, (1024, 1024)  # 8 x 4 MiB at fp32
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "f32.safetensors"
        save_file({f"model.layers.{i}.mlp.up_proj.weight": torch.randn(shape)
                   for i in range(n)}, str(path))
        store = EmbeddingStore(Path(d) / "t.db")
        emb = Embedder(store, model_name=str(path), layer="35", device="cuda",
                       config_repo="Qwen/Qwen3-4B")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        state, _n, _skip = emb._read_state(torch.bfloat16)
        peak = torch.cuda.max_memory_allocated() - base
        fp32_total = n * shape[0] * shape[1] * 4
        assert all(t.dtype == torch.bfloat16 for t in state.values())
        # bf16 total is half of fp32; allow one fp32 tensor in flight.
        assert peak <= fp32_total / 2 + fp32_total / n + (1 << 20), (peak, fp32_total)
        del state
        torch.cuda.empty_cache()
        store.close()
    print("ok: an fp32 checkpoint loads without holding it whole at fp32")


if __name__ == "__main__":
    test_pack_encoder()
    test_dequantizes_and_strips_prefix()
    test_unquantized_file_unchanged()
    test_untied_lm_head_is_dropped()
    test_unsupported_fp8_formats_are_refused()
    test_fp32_checkpoint_peaks_at_target_size()
    test_orphan_scale_is_loud()
    test_skip_prefixes()
    test_vision_config_mismatch_guard()
    test_sharded_directory()
    test_shard_index_wins_over_globbing()
    test_nested_tower_prefix()
    test_text_tower_dispatch()
    test_peek_keys_reads_header_only()
    test_vram_needed_scales_with_the_model()
    print("all fp8 loader tests passed")
