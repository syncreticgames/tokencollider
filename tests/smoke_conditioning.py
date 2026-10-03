"""Model-free tests for the layer-stack conditioning: value type, file format,
and a golden regression pinning the single-depth export bytes.

The golden digests were captured from the single-tensor code that predates
Conditioning. They must not move: a single-depth model is meant to be the
size-one case of the stack, not a different answer.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider.conditioning import Conditioning
from tokencollider.embedder import FakeEmbedder
from tokencollider.layout import LayerStack
from tokencollider.server import export_conditioning

WORDS = ["Mario", "Luigi", "Bowser", "Link", "Zelda", "Kirby", "Samus", "Ridley"]
COORDS = [0.3, -0.2, 0.1, 0.05, -0.05, 0.0]

# Captured from tokencollider.server.export_conditioning at commit f94ceeb (the last
# single-tensor revision), driven by the StubEmbedder below. sha256 over the
# concatenated tensor bytes.
GOLDEN = {
    "classic": "6b7e0b027964edaa3918614db808d47171e9d3ad9feaf68f5dd93330bcb17823",
    "band_cook": "6d943b3ed5661a99dde24e2bb83ac182e0b8655c80e173c9c2fd1af26a8ea331",
}


class StubEmbedder(FakeEmbedder):
    """FakeEmbedder plus a deterministic cook and per-depth conditionings, so
    the patch-and-cook export path is exercised without a GPU. `sampler`
    overrides which depths the model reads, standing in for a multi-layer
    consumer."""

    store = None
    model_name = "stub"
    pooling = "mean"
    template = "an image of {}"

    def __init__(self, dim=64, n_layers=13, sampler=None):
        super().__init__(dim=dim, n_layers=n_layers)
        self._sampler = tuple(sampler) if sampler else (n_layers - 1,)

    def _resolved_layer(self):
        return max(self._sampler)

    def sampler_layers(self):
        return self._sampler

    def conditioning(self, text):
        return Conditioning(self.conditioning_layers(text, list(self._sampler)))

    def conditioning_layers(self, text, layers):
        base = self.conditioning_tokens(text)
        out = {}
        for l in layers:
            seed = int.from_bytes(
                hashlib.sha256(f"{text}|{int(l)}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            out[int(l)] = (base + 0.01 * l * rng.standard_normal(base.shape)
                           ).astype(np.float32)
        return out

    def cook_layers(self, hidden, start_index, pulls=None, stop_index=None,
                    capture=(), position_ids=None):
        h = np.asarray(hidden, dtype=np.float32).copy()
        stop = self.n_layers - 1 if stop_index is None else stop_index
        wanted = {int(c) for c in capture} | {stop}
        out = {}
        if start_index in wanted:
            out[start_index] = h.copy()
        for j in range(start_index, stop):
            h = h * 1.01 + 0.001 * j
            if pulls and (j + 1) in pulls and (j + 1) < stop:
                a, t = pulls[j + 1]
                h = (1.0 - a) * h + a * t
            if (j + 1) in wanted and (j + 1) < stop:
                out[j + 1] = h.astype(np.float32)
        out[stop] = h.astype(np.float32)
        return out


def build_stack(**kwargs):
    stack = LayerStack(StubEmbedder(**kwargs))
    stack.add_landmarks(WORDS)
    return stack


def tensor_digest(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        blob = b"".join(np.ascontiguousarray(f.get_tensor(k)).tobytes()
                        for k in sorted(f.keys()))
    return hashlib.sha256(blob).hexdigest()


def test_value_type():
    a = Conditioning({4: np.ones((3, 8), np.float32),
                      8: np.full((3, 8), 2.0, np.float32)})
    assert a.layers == (4, 8) and a.seq_len == 3 and a.dim == 8
    assert len(a) == 2 and 4 in a and 5 not in a

    # Depth order is sorted regardless of construction order.
    assert Conditioning({8: np.ones((2, 4), np.float32),
                         4: np.ones((2, 4), np.float32)}).layers == (4, 8)

    # (1, seq, dim) is accepted and unbatched.
    assert Conditioning({0: np.ones((1, 5, 4), np.float32)})[0].shape == (5, 4)

    # "the" tensor only means something for a size-one stack.
    try:
        a.tensor
        raise AssertionError("expected a refusal on a multi-depth stack")
    except ValueError:
        pass
    assert Conditioning.single(np.ones((2, 4), np.float32), 35).tensor.shape == (2, 4)

    # Blending and concatenation happen depth by depth.
    b = Conditioning({4: np.zeros((3, 8), np.float32),
                      8: np.zeros((3, 8), np.float32)})
    mixed = Conditioning.blend([a, b], [0.25, 0.75])
    assert np.allclose(mixed[4], 0.25) and np.allclose(mixed[8], 0.5)
    joined = Conditioning.concat([a, b])
    assert joined[4].shape == (6, 8) and joined.layers == (4, 8)

    # Mismatched depth sets are refused rather than silently aligned.
    odd = Conditioning({4: np.zeros((3, 8), np.float32)})
    for call in (lambda: Conditioning.blend([a, odd], [1.0, 1.0]),
                 lambda: Conditioning.concat([a, odd])):
        try:
            call()
            raise AssertionError("expected a refusal on mismatched depths")
        except ValueError:
            pass

    # Ragged blends zero-pad to the longest, as landmark phrases do.
    ragged = Conditioning.blend(
        [Conditioning.single(np.ones((2, 4), np.float32), 0),
         Conditioning.single(np.ones((5, 4), np.float32), 0)], [1.0, 1.0])
    assert ragged[0].shape == (5, 4)
    assert np.allclose(ragged[0][0], 2.0) and np.allclose(ragged[0][4], 1.0)

    # Trimming drops leading tokens at every depth, together.
    trimmed = a.trim(1)
    assert trimmed.layers == (4, 8) and trimmed.seq_len == 2
    assert np.array_equal(trimmed[4], a[4][1:])
    assert a.trim(0) is a
    try:
        a.trim(3)  # exactly the sequence length: nothing would be left
        raise AssertionError("expected a refusal on an over-long trim")
    except ValueError as e:
        assert "cannot trim" in str(e), e

    relabelled = a.relabel((10, 20))
    assert relabelled.layers == (10, 20)
    assert np.array_equal(relabelled[10], a[4])
    print("ok: conditioning value type")


def test_file_format():
    with tempfile.TemporaryDirectory() as d:
        # A size-one stack writes exactly the historical shape: one
        # "conditioning" key, and no "layers" field added to the metadata.
        one = Conditioning.single(
            np.arange(12, dtype=np.float32).reshape(3, 4), 35)
        p1 = Path(d) / "one.safetensors"
        one.save(str(p1), {"layer": "35", "note": "hi"}, framework="np")
        from safetensors import safe_open
        with safe_open(str(p1), framework="np") as f:
            assert list(f.keys()) == ["conditioning"]
            assert f.get_tensor("conditioning").shape == (1, 3, 4)
            assert "layers" not in (f.metadata() or {})
        back, meta = Conditioning.load(str(p1))
        assert back.layers == (35,) and meta["note"] == "hi"
        assert np.array_equal(back[35], one[35])

        # A band export is solved at 18-34 but assembled at the sampler depth,
        # so "layer" wins over "view_layer" when naming the depth.
        p2 = Path(d) / "band.safetensors"
        one.save(str(p2), {"layer": "18-34", "view_layer": "4-8"}, framework="np")
        assert Conditioning.load(str(p2))[0].layers == (34,)

        # Multi-depth writes one key per depth plus the depth list.
        many = Conditioning({2: np.ones((3, 4), np.float32),
                             5: np.full((3, 4), 7.0, np.float32)})
        p3 = Path(d) / "many.safetensors"
        many.save(str(p3), {"layer": "krea"}, framework="np")
        with safe_open(str(p3), framework="np") as f:
            assert sorted(f.keys()) == ["layer_02", "layer_05"]
            assert json.loads(f.metadata()["layers"]) == [2, 5]
        back, _ = Conditioning.load(str(p3))
        assert back.layers == (2, 5) and np.allclose(back[5], 7.0)
    print("ok: conditioning file format (single-depth shape preserved)")


def test_golden_single_depth_export():
    """The bytes a single-depth model exports must not have moved."""
    cases = {
        "classic": (build_stack(), None, None),
        "band_cook": (build_stack(), (4, 8), (4, 8)),
    }
    for name, (stack, layer, band) in cases.items():
        with tempfile.TemporaryDirectory() as d:
            res = export_conditioning(stack, layer, COORDS,
                                      str(Path(d) / "g.safetensors"),
                                      "safetensors", band)
            assert res["layers"] == [12], res["layers"]
            got = tensor_digest(res["path"])
            assert got == GOLDEN[name], f"{name}: {got} != {GOLDEN[name]}"
    print("ok: golden single-depth exports unchanged (classic + patch-and-cook)")


def test_multi_depth_export():
    """A sampler reading several depths gets a tensor for each: the depths
    above the injection are cooked in one resumed pass, the ones at or below
    it are blended, because they sit upstream of the patch."""
    sampler = (4, 8, 12)
    stack = build_stack(sampler=sampler)
    with tempfile.TemporaryDirectory() as d:
        res = export_conditioning(stack, (6, 8), COORDS,
                                  str(Path(d) / "stack.safetensors"),
                                  "safetensors", (6, 8))
        assert res["cooked"] is True and res["band"] == [6, 8]
        assert res["layers"] == [4, 8, 12]
        cond, meta = Conditioning.load(res["path"])
        assert cond.layers == sampler
        assert json.loads(meta["layers"]) == [4, 8, 12]

        # Depth 12 is above the injection, so it is the cooked state; depth 4
        # is below it and must equal the plain blend at that depth.
        session = stack.session((6, 8))
        weights = list(session.blend_weights(COORDS)["weights"].values())
        assert np.allclose(cond[4], session._blend_at([4], weights)[4])
        assert not np.allclose(cond[12], session._blend_at([12], weights)[12])

        # The .cond pickle holds one tensor and has nowhere to say which
        # depth is which, so it refuses a stack instead of guessing.
        try:
            export_conditioning(stack, (6, 8), COORDS,
                                str(Path(d) / "x.cond"), "cond", (6, 8))
            raise AssertionError("expected .cond to refuse a multi-depth stack")
        except ValueError as e:
            assert "safetensors" in str(e)
    print("ok: multi-depth export (cooked above the patch, blended below)")


def test_uncooked_multi_depth_export():
    """With no band there is nothing to cook, and every sampler depth comes
    straight from the landmarks' own states."""
    stack = build_stack(sampler=(4, 8, 12))
    with tempfile.TemporaryDirectory() as d:
        res = export_conditioning(stack, None, COORDS,
                                  str(Path(d) / "flat.safetensors"),
                                  "safetensors", None)
        assert res["cooked"] is False and res["layers"] == [4, 8, 12]
        cond, _ = Conditioning.load(res["path"])
        session = stack.session(None)
        weights = list(session.blend_weights(COORDS)["weights"].values())
        for depth in (4, 8, 12):
            assert np.allclose(cond[depth], session._blend_at([depth], weights)[depth])
    print("ok: uncooked multi-depth export")


def test_export_trim():
    """A model whose sampler is handed the prompt tokens onward gets the
    scaffold dropped, and only at the very end: cooking needs the context."""

    class Trimming(StubEmbedder):
        PREFIX = 2

        def export_trim(self):
            return self.PREFIX

    for band, cooked in ((None, False), ((6, 8), True)):
        stack = LayerStack(Trimming(sampler=(4, 8, 12)))
        stack.add_landmarks(WORDS)
        plain = LayerStack(StubEmbedder(sampler=(4, 8, 12)))
        plain.add_landmarks(WORDS)
        with tempfile.TemporaryDirectory() as d:
            cut = export_conditioning(stack, band, COORDS,
                                      str(Path(d) / "t.safetensors"),
                                      "safetensors", band)
            whole = export_conditioning(plain, band, COORDS,
                                        str(Path(d) / "w.safetensors"),
                                        "safetensors", band)
            assert cut["cooked"] is cooked and whole["cooked"] is cooked
            assert cut["shape"][1] == whole["shape"][1] - Trimming.PREFIX
            a, _ = Conditioning.load(cut["path"])
            b, _ = Conditioning.load(whole["path"])
            # The kept tail is exactly the untrimmed export's tail, which is
            # what "trim last, after cooking" has to mean.
            for depth in a.layers:
                assert np.allclose(a[depth], b[depth][Trimming.PREFIX:]), depth
    print("ok: export trim drops the scaffold, after any cook")


def test_mixed_modality_export():
    """A cursor among phrases and pictures exports one frame: prefix, the
    picture group's slot, the phrase group's slot, the shared tail, each
    group blended only with its own kind. Single-modality universes are
    untouched (the golden test above pins that)."""
    class Mixed(StubEmbedder):
        PREFIX, TAIL = 2, 3

        def template_prefix_tokens(self):
            return self.PREFIX

        def template_tail_tokens(self):
            return self.TAIL

        def conditioning_layers(self, text, layers):
            # Pictures carry a 6-token slot, phrases a 3-token slot, both
            # wrapped in the same 2-token prefix and 3-token tail.
            slot = 6 if text.startswith("image:") else 3
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            base = rng.standard_normal((self.PREFIX + slot + self.TAIL, 64)).astype(np.float32)
            base[: self.PREFIX] = 7.0  # the scaffold prefix is shared verbatim
            return {int(l): base + 0.01 * l for l in layers}

        def mixed_position_ids(self, segments):
            self.last_segments = segments
            return None

    emb = Mixed(dim=64, sampler=(4, 8, 12))
    stack = LayerStack(emb)
    phrases = ["Mario", "Luigi", "Peach"]
    pics = ["image:aaaa", "image:bbbb"]
    stack.add_landmarks(phrases + pics)
    session = stack.session(None)
    blend = session.blend_weights(COORDS)
    w = blend["weights"]
    with tempfile.TemporaryDirectory() as d:
        res = export_conditioning(stack, None, COORDS, str(Path(d) / "m.safetensors"),
                                  "safetensors", None)
        cond, meta = Conditioning.load(res["path"])
        assert json.loads(meta["groups"]) == {"phrase": 3, "image": 2}, meta["groups"]
        frame = json.loads(meta["frame"])
        assert frame == [["prefix", 2], ["image", 6], ["phrase", 3], ["tail", 3]], frame
        assert cond.seq_len == 2 + 6 + 3 + 3
        states = {p: emb.conditioning_layers(p, [8])[8] for p in phrases + pics}
        # The picture block is the pictures' slots blended with their own
        # weights, the phrase block likewise; no cross-modality averaging.
        pic_block = sum(w[p] * states[p][2:8] for p in pics)
        phr_block = sum(w[p] * states[p][2:5] for p in phrases)
        tail_block = sum(w[p] * states[p][-3:] for p in phrases + pics)
        assert np.allclose(cond[8][2:8], pic_block, atol=1e-5)
        assert np.allclose(cond[8][8:11], phr_block, atol=1e-5)
        assert np.allclose(cond[8][11:14], tail_block, atol=1e-5)
        assert np.allclose(cond[8][:2], states["Mario"][:2])  # the shared prefix
        assert abs(json.loads(meta["group_weights"])["image"]
                   - sum(w[p] for p in pics)) < 1e-5

        # A cook over a mixed frame asks the embedder for 3-D positions with
        # the frame's layout.
        res2 = export_conditioning(stack, (6, 8), COORDS, str(Path(d) / "c.safetensors"),
                                   "safetensors", (6, 8))
        assert res2["cooked"] is True
        assert emb.last_segments == [("text", 2), ("image", 6), ("text", 3), ("text", 3)], emb.last_segments
    print("ok: mixed-modality export (blend within, concatenate across)")


def test_comfy_node_fuses_stack():
    """The ComfyUI node hands a multi-depth export to the sampler in the
    layout ComfyUI's own Krea 2 encoder produces: (1, seq, n*dim), depths
    ascending, depth i owning features [i*dim, (i+1)*dim). One depth at
    2560 wide is what the Krea 2 model refuses, so layer=-1 must fuse."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tc_node", Path(__file__).resolve().parent.parent
        / "comfyui_node" / "nodes.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)

    rng = np.random.default_rng(0)
    depths = (2, 5, 35)
    cond = Conditioning({d: rng.standard_normal((7, 16)).astype(np.float32)
                         for d in depths})
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "stack.safetensors")
        cond.save(path, {"layers": json.dumps(list(depths))}, framework="np")

        fused, meta, seen = node.read_conditioning(path, -1)
        assert seen == list(depths)
        assert tuple(fused.shape) == (1, 7, 3 * 16), fused.shape
        for i, depth in enumerate(depths):
            block = fused[0, :, i * 16:(i + 1) * 16].numpy()
            assert np.array_equal(block, cond[depth]), depth
        # Round trip through the DiT's own unpack: reshape(b, seq, n, dim).
        unpacked = fused.reshape(1, 7, 3, 16)
        assert np.array_equal(unpacked[0, :, 2].numpy(), cond[35])

        one, _, _ = node.read_conditioning(path, 5)
        assert tuple(one.shape) == (1, 7, 16)
        assert np.array_equal(one[0].numpy(), cond[5])
        try:
            node.read_conditioning(path, 4)
            raise AssertionError("expected a refusal on a depth not in the file")
        except ValueError as e:
            assert "no layer 4" in str(e), e

        # A single-depth file is untouched by the widget.
        single = str(Path(d) / "one.safetensors")
        Conditioning.single(cond[35], 35).save(single, {}, framework="np")
        t, _, seen = node.read_conditioning(single, -1)
        assert seen == [] and tuple(t.shape) == (1, 7, 16)
    print("ok: ComfyUI node fuses a stack layer-major (Krea 2's frame)")


def test_profiles():
    """The per-model conventions, as verified against ComfyUI on 08222026."""
    import os

    from tokencollider import profiles

    # Hermetic: ignore whatever profiles.yaml this machine happens to have.
    os.environ["TOKENCOLLIDER_PROFILES"] = "/nonexistent/profiles.yaml"
    krea = profiles.get("krea2")
    assert krea.sampler_layers == (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    assert krea.trim_template_prefix is True
    assert krea.config_repo is None  # must be pointed at a local config
    assert krea.template.startswith("<|im_start|>system\n")
    assert krea.template.count("{}") == 1

    z = profiles.get("zimage")
    assert z.sampler_layers is None and z.layer == "35"
    # The viewport opens on the charted layer, so it must sit inside the
    # slider bounds, or the first view is one nobody warmed.
    for prof in (z, krea):
        assert prof.layer_min <= int(prof.layer) <= prof.layer_max, prof.name
    # comfy/text_encoders/z_image.py has no template_end slice.
    assert z.trim_template_prefix is False

    # Every setting is scoped; there is no shared tier to shadow with.
    import os
    for shared in ("TOKENCOLLIDER_MODEL", "TOKENCOLLIDER_LAYER",
                   "TOKENCOLLIDER_TEMPLATE", "TOKENCOLLIDER_CONFIG_DIR"):
        os.environ[shared] = "SHOULD_BE_IGNORED"
    try:
        for name in ("MODEL", "LAYER", "TEMPLATE", "CONFIG_DIR"):
            value, source = profiles.resolve(krea, name, None, "fallback")
            assert value == "fallback", (name, value)
            assert source == "profile:krea2", (name, source)
        os.environ["TOKENCOLLIDER_KREA2_MODEL"] = "scoped"
        assert profiles.resolve(krea, "MODEL", None) == ("scoped", "TOKENCOLLIDER_KREA2_MODEL")
        assert profiles.resolve(krea, "MODEL", "flag") == ("flag", "flag")
        try:
            profiles.require(krea, "MODEL", None, "weights path")
            raise AssertionError("expected a refusal on a missing path")
        except SystemExit as e:
            assert "TOKENCOLLIDER_KREA2_MODEL" in str(e), e
    finally:
        for k in ("TOKENCOLLIDER_MODEL", "TOKENCOLLIDER_LAYER",
                  "TOKENCOLLIDER_TEMPLATE", "TOKENCOLLIDER_CONFIG_DIR",
                  "TOKENCOLLIDER_KREA2_MODEL"):
            os.environ.pop(k, None)

    assert profiles.parse_sampler_layers("8,2,5") == (2, 5, 8)
    assert profiles.parse_sampler_layers("") is None
    try:
        profiles.get("nope")
        raise AssertionError("expected a refusal on an unknown profile")
    except SystemExit as e:
        assert "unknown model profile" in str(e)
    assert krea.origin == "built-in"
    print("ok: model profiles (krea2 trims, zimage does not)")


def test_profiles_yaml():
    """profiles.yaml supplies locations and variants; the built-ins keep the
    conventions. A new profile must extend one, so it cannot invent a dialect
    by accident."""
    import os

    from tokencollider import profiles

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profiles.yaml"
        path.write_text(
            "profiles:\n"
            "  krea2:\n"
            "    model: /base/krea.safetensors\n"
            "    config_dir: /base/cfg\n"
            "  krea2-finetune:\n"
            "    extends: krea2\n"
            "    model: /finetune/dir\n"
            "  krea2-deep:\n"          # extends a profile defined later above
            "    extends: krea2-finetune\n"
            "    layer: 23\n"
            "default: krea2-finetune\n", encoding="utf-8")
        os.environ["TOKENCOLLIDER_PROFILES"] = str(path)
        try:
            finetune = profiles.get("krea2-finetune")
            assert finetune.model_path == "/finetune/dir"
            assert finetune.config_dir == "/base/cfg"          # inherited
            assert finetune.trim_template_prefix is True       # convention survives
            assert finetune.sampler_layers == (2, 5, 8, 11, 14, 17, 20, 23, 26,
                                            29, 32, 35)
            assert finetune.env_prefix == "TOKENCOLLIDER_KREA2_FINETUNE"     # scoped by name
            assert finetune.origin == str(path)

            # Chained extends, resolved regardless of order in the file.
            deep = profiles.get("krea2-deep")
            assert deep.layer == "23" or deep.layer == 23
            assert deep.model_path == "/finetune/dir"

            # The file's `default` is what an unnamed profile resolves to.
            assert profiles.get().name == "krea2-finetune"

            # Built-ins survive alongside, unmodified.
            assert profiles.get("zimage").trim_template_prefix is False

            for bad, expect in (
                ("profiles: {orphan: {model: /x}}", "needs a 'template'"),
                ("profiles: {a: {extends: b}, b: {extends: a}}", "circular"),
                ("profiles: {krea2: {typo: 1}}", "unknown field"),
                ("profiles: {krea2: {}}\ndefault: nope", "not defined"),
            ):
                path.write_text(bad, encoding="utf-8")
                try:
                    profiles.load(path)
                    raise AssertionError(f"expected a refusal: {bad}")
                except SystemExit as e:
                    assert expect in str(e), (bad, str(e))
        finally:
            os.environ.pop("TOKENCOLLIDER_PROFILES", None)
    print("ok: profiles.yaml (extends, default, and four loud failures)")


def test_light_warm_stays_light():
    """A light warm caches pooled vectors and nothing else, including when a
    bulk layer view pulls a depth it did not originally store. On a
    twelve-depth model, falling through to a full warm here is the difference
    between tens of megabytes and tens of gigabytes."""
    from tokencollider.embedder import Embedder
    from tokencollider.store import EmbeddingStore

    class Recorder(Embedder):
        """Real _store_states, fake forward pass."""

        def __init__(self, store, **kw):
            super().__init__(store, model_name="stub", **kw)
            self.n_layers = 13
            self.passes = 0

        def _span(self, text):
            return 1, 3

        def _forward_all_batch(self, texts):
            self.passes += len(texts)
            return [[np.full((4, 8), float(l), dtype=np.float32)
                     for l in range(13)] for _ in texts]

    words = ["alpha", "beta", "gamma"]
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "light.db")
        emb = Recorder(store, layer="10", light=True,
                       sampler_layers=(2, 5, 8, 10))

        emb.embed_many(words)
        conds = store.conn.execute("SELECT COUNT(*) FROM conditionings").fetchone()[0]
        assert conds == 0, conds
        depths = {r[0] for r in store.conn.execute(
            "SELECT DISTINCT layer FROM embeddings")}
        assert depths == {"10"}, depths

        # Scrubbing to a depth the light warm never stored must cache that
        # depth's pooled vectors, not a whole conditioning stack per phrase.
        emb.embed_layers_many(words, 4)
        conds = store.conn.execute("SELECT COUNT(*) FROM conditionings").fetchone()[0]
        assert conds == 0, f"light warm sprouted {conds} conditioning rows"
        depths = {r[0] for r in store.conn.execute(
            "SELECT DISTINCT layer FROM embeddings")}
        assert depths == {"10", "4"}, depths

        # A full (non-light) embedder does store the stack, one row per
        # sampler depth, which is the behaviour light mode is opting out of.
        full = Recorder(store, layer="10", light=False,
                        sampler_layers=(2, 5, 8, 10))
        full.embed_many(["delta"])
        rows = store.conn.execute(
            "SELECT COUNT(*) FROM conditionings WHERE text = 'delta'").fetchone()[0]
        assert rows == 4, rows
        store.close()
    print("ok: light warms stay light across a layer scrub")


def test_cook_capture_range():
    """Depths below the injection cannot be cooked, and saying so beats
    handing back a state the resumed pass never produced.

    Runs against the real Embedder: the guard is hoisted above the model load
    precisely so a bad call is rejected without touching the GPU."""
    from tokencollider.embedder import Embedder
    from tokencollider.store import EmbeddingStore

    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "t.db")
        emb = Embedder(store, model_name="unused", layer="35")
        try:
            emb.cook_layers(np.zeros((3, 8), np.float32), 8, None, 12, capture=[4])
            raise AssertionError("expected a refusal on an out-of-range capture")
        except ValueError as e:
            assert "outside the cooked range" in str(e), e
        assert emb._model is None, "the guard must fire before the model loads"
        store.close()

    # And the stub confirms the capture set that comes back on a good call.
    states = StubEmbedder().cook_layers(
        np.zeros((3, 64), np.float32), 8, None, 12, capture=[8, 10, 12])
    assert sorted(states) == [8, 10, 12]
    print("ok: cook capture range (guarded before the model load)")


def test_layer_setting_is_canonical():
    """'20-20' parses to layer 20, so it must also KEY as 20: the light warm
    did int('20-20') and the vocab looked under a key nothing wrote."""
    from tokencollider.embedder import Embedder, canonical_layer
    from tokencollider.store import EmbeddingStore
    assert [canonical_layer(v) for v in ("last", "20", "20-20", "18-34", 7, (3, 3))] \
        == ["last", "20", "20", "18-34", "7", "3"]
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "c.db")
        assert Embedder(store, model_name="unused", layer="20-20").layer == "20"
        store.close()
    print("ok: layer settings key one way")


def test_profiles_are_data():
    """No model is wired into the code. The built-ins load from
    tokencollider/builtin_profiles.yaml, a new encoder is a profile from scratch, and
    with nothing chosen there is no silent default."""
    import os

    from tokencollider import profiles

    assert "def " not in profiles.BUILTIN_FILE.read_text(encoding="utf-8")
    assert set(profiles.builtins()) >= {"zimage", "krea2"}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profiles.yaml"
        path.write_text(
            "profiles:\n"
            "  other:\n"
            "    template: \"<s>{}</s>\"\n"
            "    sampler_layers: [10, 20]\n"
            "    layer: 20\n"
            "    config_dir: /cfg\n", encoding="utf-8")
        reg = profiles.load(path)
        other = reg["other"]
        assert (other.template, other.sampler_layers, other.env_prefix) == \
            ("<s>{}</s>", (10, 20), "TOKENCOLLIDER_OTHER"), other
        assert other.trainer_arch is None and not other.trim_template_prefix
        saved = {k: os.environ.pop(k, None)
                 for k in ("TOKENCOLLIDER_PROFILE", "TOKENCOLLIDER_PROFILES")}
        os.environ["TOKENCOLLIDER_PROFILES"] = str(path)  # a file with no default:
        try:
            profiles.get()
            raise AssertionError("expected a refusal with no profile chosen")
        except SystemExit as e:
            assert "no model profile chosen" in str(e), e
        finally:
            os.environ.pop("TOKENCOLLIDER_PROFILES", None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
    print("ok: profiles are data, with no built-in default")


def test_comfy_stack_skips_mirrors():
    """A stack is one cursor's positive exports. A mirror (the negative half
    of a polarity pair) that happens to share the stem must stay out, under
    either spelling the server writes, and a `_cookNN.` in a directory name
    must not count."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tc_node", Path(__file__).resolve().parent.parent
        / "comfyui_node" / "nodes.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "run_cook99.x"
        root.mkdir()
        # A taken name gets a 6-hex digest appended (server.py), mirror or not.
        for name in ("a_cook08.safetensors", "a_cook20.safetensors",
                     "a_cook14_3fa9c1.safetensors",
                     "a_cook08_mirror.safetensors", "a_cook08.mirror.safetensors",
                     "a_cook14_mirror_3fa9c1.safetensors",
                     "notes.txt"):
            (root / name).write_bytes(b"")
        found = node.LoadConditioningStack._matches(str(root))
        assert [(n, Path(p).name) for n, p in found] == [
            (8, "a_cook08.safetensors"), (14, "a_cook14_3fa9c1.safetensors"),
            (20, "a_cook20.safetensors")], found
    print("ok: comfy stack loader keeps renamed cooks and leaves mirrors out")

    # One stack per load: a folder of two stacks (or a stack beside a single
    # cooked export) is refused by name; a glob picks one out.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name in ("red_stack1a2b3c_map0_8_cook04.safetensors",
                     "red_stack1a2b3c_map0_8_cook08.safetensors",
                     "red_stackd4e5f6_map0_8_cook04.safetensors",
                     "red_stackd4e5f6_map0_8_cook08.safetensors",
                     "blue_map0_8_cook08.safetensors"):
            (root / name).write_bytes(b"")
        try:
            node.LoadConditioningStack._matches(str(root))
            raise AssertionError("a folder of three stacks was accepted")
        except ValueError as e:
            assert "3 different stacks" in str(e) and "red_stackd4e5f6_map0_8" in str(e), e
        assert node.LoadConditioningStack.IS_CHANGED(str(root), 1, 0.0) != \
            node.LoadConditioningStack.IS_CHANGED(str(root), 1, 0.0)  # NaN: re-run
        found = node.LoadConditioningStack._matches(str(root / "red_stack1a2b3c_*"))
        assert [n for n, _ in found] == [4, 8], found
    print("ok: comfy stack loader refuses a folder of several stacks")


def test_layer_count_comes_from_config():
    """A light warm caches only the charting layer, so the highest cached
    layer says nothing about the top of the model. The count comes from the
    checkpoint's config (blocks + 1); the cache is only a last resort."""
    from transformers import Qwen3Config
    from tokencollider.embedder import Embedder
    from tokencollider.store import EmbeddingStore

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "cfg"
        Qwen3Config(num_hidden_layers=6, hidden_size=32, intermediate_size=64,
                    num_attention_heads=4, num_key_value_heads=2,
                    vocab_size=128).save_pretrained(cfg)
        store = EmbeddingStore(Path(d) / "c.db")
        store.layers_cached = lambda _model: ["2"]  # a light warm at layer 2
        emb = Embedder(store, model_name="unused", config_dir=str(cfg), layer="2")
        assert emb.layer_count() == 7, emb.layer_count()
        # No config anywhere: fall back to the cache's lower bound.
        bare = Embedder(store, model_name=str(Path(d) / "missing"), layer="2")
        assert bare.layer_count() == 3, bare.layer_count()
    print("ok: layer count from the checkpoint config, not a light cache")


def test_prefix_rows_copied_not_scaled():
    """The template prefix is identical for every landmark (causal attention,
    shared scaffold), so a blend whose weights do not sum to 1 (strict, pole
    and world-axes solves) must copy it, not scale it."""
    from tokencollider.embedder import FakeEmbedder
    from tokencollider.layout import LayerStack

    class Prefixed(FakeEmbedder):
        def template_prefix_tokens(self):
            return 2

        def conditioning_layers(self, text, layers):
            out = super().conditioning_layers(text, layers)
            for t in out.values():
                t[:2] = 7.0  # what causal attention guarantees: shared rows
            return out

    stack = LayerStack(Prefixed(dim=16))
    for w in ("red", "blue", "green"):
        stack.add_landmark(w)
    session = stack.session(None)
    layer = stack.embedder.n_layers - 1
    for weights in ([0.5, 0.4, 0.0], [0.6, 0.6, -0.1], [1.0, 0.0, 0.0]):
        out = session._blend_at([layer], weights)[layer]
        assert np.allclose(out[:2], 7.0), (weights, out[:2, :3])
    print("ok: blend copies the shared prefix rows instead of scaling them")


def test_check_frame():
    """docs/export-format.md: a reader that knows its model refuses a file
    from another frame (template, layers, prefix trim) and warns on another
    model or on fields an old file lacks."""
    from tokencollider.conditioning import check_frame

    t = np.zeros((3, 4), dtype=np.float32)
    stack = Conditioning({2: t, 5: t})
    good = {"template": "T{}", "model": "a.safetensors", "template_prefix_tokens": "4",
            "layers": "[2, 5]"}
    assert check_frame(stack, good, template="T{}", layers=(2, 5), prefix_tokens=4,
                       model="a.safetensors") == []
    for field, bad, layers, prefix in (
            ("template", {**good, "template": "U{}"}, (2, 5), 4),
            ("layers", good, (2, 5, 8), 4),
            ("template_prefix_tokens", good, (2, 5), 0)):
        try:
            check_frame(stack, bad, template="T{}", layers=layers, prefix_tokens=prefix)
            raise AssertionError(f"{field} mismatch accepted")
        except ValueError as e:
            assert field in str(e), (field, e)
    notes = check_frame(stack, good, template="T{}", layers=(2, 5), prefix_tokens=4,
                        model="elsewhere/a-finetune")
    assert len(notes) == 1 and "model" in notes[0], notes
    # A single-layer file names its layer in "layer"; an old one may not.
    one = Conditioning.single(t, 35)
    assert check_frame(one, {"template": "T{}", "layer": "35"}, template="T{}",
                       layers=(35,), prefix_tokens=0) == []
    try:
        check_frame(one, {"template": "T{}", "layer": "35"}, template="T{}",
                    layers=(20,), prefix_tokens=0)
        raise AssertionError("single-layer mismatch accepted")
    except ValueError:
        pass
    old = check_frame(Conditioning.single(t, 0), {}, template="T{}", layers=(35,),
                      prefix_tokens=0)
    assert len(old) == 2, old  # no template, no layer: loads, with notes
    print("ok: export readers refuse another frame and warn on another model")


if __name__ == "__main__":
    test_value_type()
    test_file_format()
    test_golden_single_depth_export()
    test_multi_depth_export()
    test_uncooked_multi_depth_export()
    test_export_trim()
    test_mixed_modality_export()
    test_comfy_node_fuses_stack()
    test_comfy_stack_skips_mirrors()
    test_profiles_are_data()
    test_layer_setting_is_canonical()
    test_profiles()
    test_profiles_yaml()
    test_light_warm_stays_light()
    test_layer_count_comes_from_config()
    test_prefix_rows_copied_not_scaled()
    test_check_frame()
    test_cook_capture_range()
    print("all conditioning tests passed")
