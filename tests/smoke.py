"""Model-free smoke test: store roundtrip, universe math on synthetic vectors,
and command-line parsing."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider import provenance
from tokencollider.store import EmbeddingStore
from tokencollider.universe import Universe, cosine


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "test.db")
        vec = np.random.default_rng(0).standard_normal(2560).astype(np.float32)
        tpl = "an image of {}"
        assert store.get("m", "last", "mean", tpl, "Mario") is None
        store.put("m", "last", "mean", tpl, "Mario", vec)
        out = store.get("m", "last", "mean", tpl, "Mario")
        assert np.array_equal(out, vec)
        # Different pooling or template is a different cache entry.
        assert store.get("m", "last", "last", tpl, "Mario") is None
        assert store.get("m", "last", "mean", "{}", "Mario") is None
        # Conditioning table roundtrip.
        tens = np.random.default_rng(1).standard_normal((7, 64)).astype(np.float32)
        assert store.get_conditioning("m", "last", tpl, "Mario") is None
        store.put_conditioning("m", "last", tpl, "Mario", tens)
        assert np.array_equal(store.get_conditioning("m", "last", tpl, "Mario"), tens)
        store.close()
    print("ok: store roundtrip")


def test_store_forget_large_batch():
    """A pattern sweep can match the whole vocabulary; SQLite caps bound
    parameters per statement, so forget() has to chunk."""
    from tokencollider import store as store_mod
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "test.db")
        vec = np.ones(4, dtype=np.float32)
        n = store_mod.FORGET_CHUNK * 2 + 7
        with store.bulk():
            for i in range(n):
                store.put("m", "last", "mean", "{}", f"w{i}", vec)
        store.put("other", "last", "mean", "{}", "w0", vec)
        texts = [f"w{i}" for i in range(n)] + ["w0", "w1"]  # duplicates
        assert store.forget("m", texts) == (n, 0)
        assert store.count_for("m", "last", "mean", "{}") == 0
        assert store.get("other", "last", "mean", "{}", "w0") is not None
        store.close()
    print("ok: forget chunks a batch larger than one statement")


def test_universe_shared_component_removed():
    """Centering must erase a direction shared by every member."""
    rng = np.random.default_rng(42)
    d = 64
    shared = rng.standard_normal(d) * 10.0  # huge common component
    members = np.stack([shared + rng.standard_normal(d) for _ in range(10)]).astype(np.float32)
    phrases = [f"p{i}" for i in range(10)]
    uni = Universe.build(phrases, members, variance=0.99)

    # Raw cosines are all ~1 (dominated by the shared direction) ...
    raw = [cosine(members[0], members[i]) for i in range(1, 10)]
    assert min(raw) > 0.9, raw
    # ... but universe-relative cosines drop toward 0 once the shared
    # direction is centered out (members differ only by random noise).
    rel = [cosine(uni.projected[0], uni.projected[i]) for i in range(1, 10)]
    assert max(rel) < 0.7, rel
    assert min(rel) < 0.0, rel
    print(f"ok: shared component removed (raw spread {max(raw)-min(raw):.3f}, "
          f"relative spread {max(rel)-min(rel):.3f})")


def test_universe_rank_geometry():
    """A query built as 'mostly A, a little B' must rank A first both ways."""
    rng = np.random.default_rng(7)
    d = 64
    members = rng.standard_normal((5, d)).astype(np.float32)
    phrases = ["A", "B", "C", "D", "E"]
    uni = Universe.build(phrases, members, variance=0.999)
    query = (0.8 * members[0] + 0.2 * members[1]).astype(np.float32)

    rows = uni.rank(query)
    assert rows[0]["phrase"] == "A", rows[0]
    by_raw = min(rows, key=lambda r: r["raw_rank"])
    assert by_raw["phrase"] == "A", by_raw
    # Ranks are a permutation of 1..n for both metrics.
    for metric in ("raw_rank", "relative_rank"):
        assert sorted(r[metric] for r in rows) == list(range(1, 6))
    print("ok: rank geometry")


def test_projection_and_residual():
    rng = np.random.default_rng(3)
    members = rng.standard_normal((6, 32)).astype(np.float32)
    uni = Universe.build([f"p{i}" for i in range(6)], members, variance=0.999)
    # A member's own residual is ~0 when the basis keeps all variance.
    assert uni.residual_norm(members[2]) < 1e-3
    # A vector orthogonal to the subspace has residual ~ its centered norm.
    ortho = np.zeros(32, dtype=np.float32)
    ortho_c = ortho - uni.mean
    within = uni.project(ortho) @ uni.components
    expect = float(np.linalg.norm(ortho_c - within))
    assert abs(uni.residual_norm(ortho) - expect) < 1e-4
    print("ok: projection and residual")


def test_exclude_self():
    rng = np.random.default_rng(1)
    members = rng.standard_normal((4, 16)).astype(np.float32)
    uni = Universe.build(["A", "B", "C", "D"], members, variance=0.999)
    rows = uni.rank(members[0], exclude="A")
    assert all(r["phrase"] != "A" for r in rows)
    print("ok: exclude self")


def test_store_vectors_for():
    with tempfile.TemporaryDirectory() as d:
        store = EmbeddingStore(Path(d) / "test.db")
        tpl = "an image of {}"
        rng = np.random.default_rng(5)
        for w in ("alpha", "beta", "gamma"):
            store.put("m", "18-34", "mean", tpl, w, rng.standard_normal(16).astype(np.float32))
        store.put("m", "36", "mean", tpl, "other-layer", rng.standard_normal(16).astype(np.float32))
        assert store.count_for("m", "18-34", "mean", tpl) == 3
        texts, mat = store.vectors_for("m", "18-34", "mean", tpl)
        assert texts == ["alpha", "beta", "gamma"] and mat.shape == (3, 16)
        assert store.vectors_for("m", "0", "mean", tpl) == ([], None)
        # forget: exact-text rows vanish across the model, others survive
        e, c = store.forget("m", ["beta", "never-there"])
        assert (e, c) == (1, 0)
        assert store.count_for("m", "18-34", "mean", tpl) == 2
        store.vacuum()
        assert store.get("m", "18-34", "mean", tpl, "alpha") is not None
        store.close()
    print("ok: store vectors_for slice + forget")


class _StubEmbedder:
    model_name = "stub-model"
    layer = "18-34"
    pooling = "mean"
    template = "an image of {}"


def test_provenance_roundtrip():
    phrases = ["Mario", "Luigi", "Peach"]
    meta = provenance.build_meta(_StubEmbedder(), phrases,
                                 extra={"manual": ["Peach"]})
    # Fingerprint is order-invariant over phrases and config-sensitive.
    fp = meta["fingerprint"]
    assert fp == provenance.fingerprint("stub-model", "18-34", "mean",
                                        "an image of {}", ["Peach", "Mario", "Luigi"])
    assert fp != provenance.fingerprint("stub-model", "18-24", "mean",
                                        "an image of {}", phrases)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "u.txt"
        provenance.write_universe(path, phrases, meta)
        back_phrases, back_meta = provenance.read_universe(path)
        assert back_phrases == phrases
        assert back_meta["fingerprint"] == fp and back_meta["manual"] == ["Peach"]
        # Matching config loads silently; a config mismatch refuses.
        provenance.verify_universe(path, back_phrases, back_meta, _StubEmbedder())
        other = _StubEmbedder()
        other.layer = "35"
        try:
            provenance.verify_universe(path, back_phrases, back_meta, other)
            raise AssertionError("config mismatch should refuse to load")
        except SystemExit:
            pass
        # A different model with the same layer/pooling/template loads the
        # phrases (that is how two models are compared on one neighborhood)
        # and strips what was a coordinate in the old frame.
        swapped = _StubEmbedder()
        swapped.model_name = "other-model"
        stripped = provenance.verify_universe(
            path, back_phrases,
            {**back_meta, "view": {"cursor": [1, 2, 3]},
             "parent": {"fingerprint": "abc", "coords": [0.0] * 6, "path": "/x"}},
            swapped)
        assert "view" not in stripped and stripped["manual"] == ["Peach"]
        assert stripped["parent"] == {"fingerprint": "abc", "path": "/x"}
        # Headerless legacy files verify as a no-op.
        legacy = Path(d) / "legacy.txt"
        legacy.write_text("Mario\nLuigi\n")
        p2, m2 = provenance.read_universe(legacy)
        assert p2 == ["Mario", "Luigi"] and m2 is None
        provenance.verify_universe(legacy, p2, m2, other)
    print("ok: provenance roundtrip + verification")


def test_bridge_cache():
    from tokencollider import bridge
    from tokencollider.embedder import FakeEmbedder

    # trigger injection mirrors ai-toolkit's semantics
    assert bridge.inject_trigger("a photo of [trigger] resting", "sks") == \
        "a photo of sks resting"
    assert bridge.inject_trigger("a red door", "sks") == "sks a red door"
    assert bridge.inject_trigger("sks at dusk", "sks") == "sks at dusk"
    # hash: deterministic, urlsafe, unpadded
    h = bridge.cache_hash("sks a red door", "zimage")
    assert h == bridge.cache_hash("sks a red door", "zimage") and "=" not in h

    from safetensors.numpy import save_file
    with tempfile.TemporaryDirectory() as d:
        dataset = Path(d) / "set"
        dataset.mkdir()
        for name, caption in (("one.png", "a red door"), ("two.jpg", None)):
            (dataset / name).write_bytes(b"fake")
            if caption:
                (dataset / name).with_suffix(".txt").write_text(caption)
        fake = FakeEmbedder(dim=64)
        anchor_path = Path(d) / "anchor.safetensors"
        anchor = np.random.default_rng(0).standard_normal((1, 5, 64)).astype(np.float32)
        save_file({"conditioning": anchor}, str(anchor_path),
                  metadata={"universe": "u" * 8})
        written = bridge.write_cache(dataset, fake, [anchor_path], "zimage",
                                     trigger="sks", default_caption="a photo")
        assert len(written) == 2
        from safetensors import safe_open
        for path in written:
            assert path.parent.name == "_t_e_cache"
            with safe_open(str(path), framework="np") as f:
                emb = f.get_tensor("text_embed")
                meta = f.metadata()
            assert emb.shape[1] == 64
            assert int(meta["anchor_tokens"]) == 5
            assert emb.shape[0] == int(meta["caption_tokens"]) + 5
            assert meta["caption"].startswith("sks")
            assert bridge.cache_hash(meta["caption"], "zimage") in path.name

        # krea2: a twelve-depth (here three) stack, caption trimmed to the
        # trainer's frame, fused layer-major to (L, n*dim) bf16 under the key
        # and class_name AdvancedPromptEmbeds.load dispatches on, filename
        # hashed with the krea2 space version.
        import json

        from tokencollider.conditioning import Conditioning

        class KreaStub(FakeEmbedder):
            PREFIX = 2

            def export_trim(self):
                return self.PREFIX

            def conditioning(self, text):
                base = self.conditioning_tokens(text)
                # Small offsets: the cache is bf16, whose resolution near 35
                # would be 0.25 and swamp the comparison below.
                return Conditioning({d: base + 0.01 * d for d in (2, 5, 35)})

        stub = KreaStub(dim=64)
        anchor2 = Path(d) / "anchor2.safetensors"
        rng = np.random.default_rng(1)
        Conditioning({d: rng.standard_normal((4, 64)).astype(np.float32)
                      for d in (2, 5, 35)}).save(str(anchor2), {"axes": "world"},
                                                 framework="np")
        for p in written:
            p.unlink()
        written = bridge.write_cache(dataset, stub, [anchor2], trigger="sks",
                                     default_caption="a photo", arch="krea2")
        import torch
        for path in written:
            with safe_open(str(path), framework="pt") as f:
                assert list(f.keys()) == ["text_embeds"], list(f.keys())
                emb = f.get_tensor("text_embeds")
                meta = f.metadata()
            assert meta["class_name"] == "AdvancedPromptEmbeds"
            assert emb.dtype == torch.bfloat16 and emb.shape[1] == 3 * 64
            cap = stub.conditioning(meta["caption"])
            assert int(meta["caption_tokens"]) == cap.seq_len - KreaStub.PREFIX
            assert emb.shape[0] == int(meta["caption_tokens"]) + 4
            # Layer-major: depth 35's block is the third 64 features, and its
            # caption rows are the trimmed caption states.
            block = emb[: int(meta["caption_tokens"]), 2 * 64:3 * 64].float().numpy()
            assert np.allclose(block, cap[35][KreaStub.PREFIX:], atol=0.03), "layer order"
            assert bridge.cache_hash(meta["caption"], "krea2") in path.name
            assert bridge.cache_hash(meta["caption"], "zimage") not in path.name
            assert json.loads(meta["anchor_0"])["axes"] == "world"

        # --jumpstart: caption alone at the canonical path (the student),
        # caption + anchor beside it as .anchor.safetensors (the teacher).
        for p in written:
            p.unlink()
        written = bridge.write_cache(dataset, stub, [anchor2], trigger="sks",
                                     default_caption="a photo", arch="krea2",
                                     jumpstart=True)
        for path in written:
            side = path.with_name(path.stem + bridge.ANCHOR_SUFFIX)
            assert side.exists(), side
            with safe_open(str(path), framework="pt") as f:
                student = f.get_tensor("text_embeds"); m1 = f.metadata()
            with safe_open(str(side), framework="pt") as f:
                teacher = f.get_tensor("text_embeds"); m2 = f.metadata()
            assert m1["role"] == "student" and m2["role"] == "teacher"
            assert student.shape[0] == int(m1["caption_tokens"])
            assert teacher.shape[0] == student.shape[0] + 4
            assert m1["anchor_tokens"] == "0" and m2["anchor_tokens"] == "4"
            assert m2["class_name"] == "AdvancedPromptEmbeds"
            # The trainer finds the sidecar from the canonical path alone.
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "jt", Path(__file__).resolve().parent.parent
                / "trainers" / "tokencollider_jumpstart" / "JumpstartTrainer.py")
            src = Path(spec.origin).read_text()
            # The module imports ai-toolkit; exercise the path helper alone.
            ns = {}
            head = "\n".join(l for l in src.split("def trainer_base")[0].splitlines()
                             if not l.startswith(("from extensions_built_in", "from toolkit")))
            exec(head, ns)
            assert ns["anchor_path_for"](str(path)) == str(side)
    print("ok: ai-toolkit bridge cache (zimage single tensor, krea2 fused stack, jumpstart sidecars)")


def test_bridge_dropout_pair():
    """Caption dropout works with a cached text encoder, by a separate path:
    ai-toolkit loads get_blank_text_embedding_path() on a drop. The bridge has
    to write that file AND its anchored sidecar, or the trainer falls back to
    live encoding with no anchor in it."""
    import tempfile

    import numpy as np

    from tokencollider import bridge
    from tokencollider.conditioning import Conditioning
    from tokencollider.embedder import FakeEmbedder

    assert bridge.dropout_caption("sks") == "sks"
    assert bridge.dropout_caption(None) == ""

    emb = FakeEmbedder(dim=32)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        from PIL import Image
        for name in ("a", "b"):
            Image.new("RGB", (8, 8)).save(root / f"{name}.png")
            (root / f"{name}.txt").write_text(f"a photo of {name}")
        anchor = root / "anchor.safetensors"
        Conditioning.single(np.zeros((4, 32), np.float32),
                            emb.n_layers - 1).save(str(anchor), framework="np")
        bridge.write_cache(root, emb, [anchor], "zimage", trigger="sks", jumpstart=True)

        cache = root / "_t_e_cache"
        blank_hash = bridge.cache_hash("sks", "zimage")
        for name in ("a", "b"):
            blank = cache / f"{name}_{blank_hash}.safetensors"
            assert blank.exists(), sorted(p.name for p in cache.iterdir())
            sidecar = blank.with_name(blank.stem + bridge.ANCHOR_SUFFIX)
            assert sidecar.exists(), "dropout teacher missing"
            # Sidecars are written in ai-toolkit's layout (key "text_embed"
            # for zimage), not ours, so read the metadata directly.
            from safetensors import safe_open
            with safe_open(str(sidecar), framework="np") as f:
                meta = dict(f.metadata() or {})
                assert "text_embed" in list(f.keys()), list(f.keys())
            assert meta["caption"] == "sks" and meta["role"] == "teacher"
            assert int(meta["anchor_tokens"]) == 4, meta
            assert meta["dropout"] == "1", meta
    print("ok: bridge writes the dropout student/teacher pair")


def test_bridge_alternate_anchors():
    """With --alternates each anchor gets its OWN teacher sidecar instead of
    being concatenated into one, so the trainer can sample a different anchor
    per step. The two halves live in different files, so this also pins the
    glob pattern the trainer uses against the names the bridge writes."""
    import glob as _glob
    import re
    import tempfile

    import numpy as np
    from PIL import Image
    from safetensors import safe_open

    from tokencollider import bridge
    from tokencollider.conditioning import Conditioning
    from tokencollider.embedder import FakeEmbedder

    emb = FakeEmbedder(dim=32)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        Image.new("RGB", (8, 8)).save(root / "a.png")
        (root / "a.txt").write_text("a photo")
        anchors = []
        for i in range(3):
            a = root / f"anchor{i}.safetensors"
            Conditioning.single(np.full((2 + i, 32), float(i), np.float32),
                                emb.n_layers - 1).save(str(a), framework="np")
            anchors.append(a)
        bridge.write_cache(root, emb, anchors, "zimage", jumpstart=True, alternates=True)

        cache = root / "_t_e_cache"
        stem = str(cache / f"a_{bridge.cache_hash('a photo', 'zimage')}")
        # The trainer globs for this exact shape; keep the two in step.
        src = (Path(__file__).resolve().parent.parent / "trainers" /
               "tokencollider_jumpstart" / "JumpstartTrainer.py").read_text()
        pattern = re.search(r'root \+ "(\.anchor\.[^"]+)"', src).group(1)
        found = sorted(_glob.glob(stem + pattern))
        assert len(found) == 3, (pattern, found, sorted(p.name for p in cache.iterdir()))
        assert not Path(stem + bridge.ANCHOR_SUFFIX).exists(), \
            "alternates should replace the single fixed anchor, not add to it"
        # The dropout caption's teachers follow the same layout, or a dropout
        # step falls back to every anchor concatenated: a different target.
        blank = str(cache / f"a_{bridge.cache_hash(bridge.dropout_caption(None), 'zimage')}")
        assert len(_glob.glob(blank + pattern)) == 3
        assert not Path(blank + bridge.ANCHOR_SUFFIX).exists()
        try:
            bridge.write_cache(root, emb, anchors, "zimage", alternates=True)
            raise AssertionError("alternates without jumpstart should refuse")
        except SystemExit:
            pass
        seen = set()
        for f in found:
            with safe_open(f, framework="np") as fh:
                meta = dict(fh.metadata() or {})
            seen.add(meta["alternate"])
            assert meta["role"] == "teacher"
            # each carries ONE anchor, of its own length (2, 3, 4 tokens)
            assert int(meta["anchor_tokens"]) == 2 + int(meta["alternate"]), meta
        assert seen == {"00", "01", "02"} or seen == {"0", "1", "2"}, seen
    print("ok: bridge writes one teacher sidecar per alternate anchor")


def test_bridge_dialect_comes_from_profile_data():
    """The cache contract for an ai-toolkit arch is the built-in profile
    declaring that trainer_arch, not numbers written into the bridge."""
    from tokencollider import bridge, profiles

    class Session:
        def __init__(self, profile, **over):
            self.layer = str(over.get("layer", profile.layer))
            self.template = over.get("template", profile.template)
            self.trim_template_prefix = profile.trim_template_prefix
            self._sampler = over.get("sampler", profile.sampler_layers)

        def sampler_layers(self):
            return self._sampler

    for name in ("zimage", "krea2"):
        ref = profiles.builtins()[name]
        assert bridge.check_dialect(ref.trainer_arch, Session(ref)) is None, name
    z = profiles.builtins()["zimage"]
    why = bridge.check_dialect("zimage", Session(z, layer="23"))
    assert why and "layers" in why, why
    k = profiles.builtins()["krea2"]
    why = bridge.check_dialect("krea2", Session(k, sampler=(2, 5, 8)))
    assert why and "layers" in why, why
    why = bridge.check_dialect("krea2", Session(k, template="{}"))
    assert why and "template" in why, why
    print("ok: the bridge's dialect check reads the profile data")


def test_cli_parsing():
    from tokencollider.cli import parse_args
    a = parse_args([])
    assert a.command == "view" and a.universe is None and not a.fake
    a = parse_args(["--fake", "-p", "krea2", "u.txt"])
    assert (a.command, a.fake, a.profile, a.universe) == ("view", True, "krea2", "u.txt")
    # common options work before or after the command, and neither clobbers
    a = parse_args(["-p", "krea2", "rank", "u.txt", "Link", "--layer", "18-24"])
    assert (a.command, a.profile, a.layer, a.universe, a.query) == \
        ("rank", "krea2", "18-24", "u.txt", "Link")
    a = parse_args(["compare", "u.txt", "q", "a", "b", "-p", "zimage"])
    assert (a.profile, a.query, a.a, a.b) == ("zimage", "q", "a", "b")
    assert a.db.name == "embeddings.db"
    a = parse_args(["axes", "u.txt", "words.txt"])
    assert (a.wordlist, a.axes, a.min_cos) == ("words.txt", 6, 0.2)
    a = parse_args(["forget", "the", "a", "--file", "stop.txt", "--dry-run"])
    assert (a.texts, a.file, a.dry_run) == (["the", "a"], "stop.txt", True)
    assert parse_args(["profiles"]).command == "profiles"
    print("ok: cli parses the short forms, options either side of the command")



def test_data_paths():
    """A checkout keeps its files beside the repo; an install keeps them in
    the user data folder, never inside site-packages; TOKENCOLLIDER_HOME wins."""
    import os
    from unittest import mock
    from tokencollider import paths
    assert paths.is_checkout()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TOKENCOLLIDER_HOME", None)
        assert paths.home() == paths.REPO
        os.environ["TOKENCOLLIDER_HOME"] = "/srv/tc"
        assert paths.home() == Path("/srv/tc")
    with mock.patch.object(paths, "is_checkout", return_value=False), \
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/x/data"}):
        os.environ.pop("TOKENCOLLIDER_HOME", None)
        with mock.patch.object(paths.sys, "platform", "linux"):
            assert paths.home() == Path("/x/data/tokencollider")
            os.environ["XDG_DATA_HOME"] = "relative/ignored"
            assert paths.home() == Path.home() / ".local/share/tokencollider"
        with mock.patch.object(paths.sys, "platform", "win32"), \
                mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:/Users/u/AppData/Local"}):
            assert paths.home() == Path("C:/Users/u/AppData/Local/TokenCollider")
        with mock.patch.object(paths.sys, "platform", "darwin"):
            assert paths.home() == Path.home() / "Library/Application Support/TokenCollider"
    print("ok: data paths (checkout, installed per platform, TOKENCOLLIDER_HOME)")

if __name__ == "__main__":
    test_store_roundtrip()
    test_store_forget_large_batch()
    test_universe_shared_component_removed()
    test_universe_rank_geometry()
    test_projection_and_residual()
    test_exclude_self()
    test_store_vectors_for()
    test_provenance_roundtrip()
    test_bridge_cache()
    test_bridge_dropout_pair()
    test_bridge_alternate_anchors()
    test_cli_parsing()
    test_data_paths()
    test_bridge_dialect_comes_from_profile_data()
    print("all smoke tests passed")
