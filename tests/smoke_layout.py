"""Model-free smoke tests for the layout session, oklab mapping, and HTTP sidecar.
"""

import json
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider.embedder import FakeEmbedder
from tokencollider.layout import LayerStack, LayoutSession
from tokencollider.oklab import hex_to_zscores, zscores_to_hex
from tokencollider.server import TOKEN_HEADER, make_handler

TOKEN = "test-session-token"
AUTH = {TOKEN_HEADER: TOKEN}

WORDS = ["Mario", "Luigi", "Bowser", "Link", "Zelda", "Kirby", "Samus", "Ridley",
         "Pikachu", "Sonic"]


def build_session(words):
    s = LayoutSession(FakeEmbedder(dim=256))
    for w in words:
        s.add_landmark(w)
    return s


def test_reconstruct_roundtrip():
    s = build_session(WORDS)
    for i in (0, 4, 9):
        vec = s.vectors[i]
        coords = s.coords_of(vec)
        rec = s.reconstruct(coords)
        # Reconstruction recovers the in-subspace part; difference == residual.
        assert abs(np.linalg.norm(vec - rec) - s.residual_norm(vec)) < 1e-3
    # A point synthesized from coords reconstructs to something whose
    # projection is exactly those coords (right-inverse property).
    coords = np.array([1.0, -2.0, 0.5, 0.3, 0.0, -1.0])
    rec = s.reconstruct(coords)
    back = s.coords_of(rec)
    assert np.allclose(back, coords, atol=1e-6), back
    print("ok: reconstruction is a right-inverse")


def test_layout_stability():
    """Adding one landmark must not teleport the existing ones."""
    s = build_session(WORDS[:8])
    before = {e["text"]: np.array(e["coords"]) for e in s.layout()["landmarks"]}
    s.add_landmark(WORDS[8])
    after = {e["text"]: np.array(e["coords"]) for e in s.layout()["landmarks"]}
    moves = [np.linalg.norm(after[w] - before[w]) for w in WORDS[:8]]
    spread = np.mean([np.linalg.norm(c) for c in before.values()])
    assert max(moves) < spread, (max(moves), spread)
    print(f"ok: layout stability (max move {max(moves):.2f} vs spread {spread:.2f})")


def test_procrustes_beats_none():
    """With alignment disabled, at least verify aligned drift <= unaligned drift."""
    from tokencollider import layout as L

    s1 = build_session(WORDS[:8])
    before = np.stack([s1.coords_of(v) for v in s1.vectors])
    s1.add_landmark(WORDS[8])
    aligned_drift = np.linalg.norm(
        np.stack([s1.coords_of(v) for v in s1.vectors[:8]]) - before
    )

    orig = L.procrustes_align
    L.procrustes_align = lambda new, old: new  # disable
    try:
        s2 = build_session(WORDS[:8])
        before2 = np.stack([s2.coords_of(v) for v in s2.vectors])
        s2.add_landmark(WORDS[8])
        raw_drift = np.linalg.norm(
            np.stack([s2.coords_of(v) for v in s2.vectors[:8]]) - before2
        )
    finally:
        L.procrustes_align = orig
    assert aligned_drift <= raw_drift + 1e-6, (aligned_drift, raw_drift)
    print(f"ok: procrustes ({aligned_drift:.2f} aligned vs {raw_drift:.2f} unaligned)")


def test_small_counts():
    s = LayoutSession(FakeEmbedder(dim=64))
    assert s.layout()["n_landmarks"] == 0
    s.add_landmark("alpha")
    lay = s.layout()
    assert lay["n_axes"] == 0 and lay["landmarks"][0]["coords"] == [0.0] * 6
    s.add_landmark("beta")
    assert s.layout()["n_axes"] == 1
    s.remove_landmark("alpha")
    assert s.layout()["n_landmarks"] == 1
    print("ok: degenerate landmark counts")


def test_oklab_roundtrip():
    for z in ([0, 0, 0], [1.5, -1.0, 0.7], [-2.0, 2.0, -2.0]):
        hexcol = zscores_to_hex(*z)
        back = hex_to_zscores(hexcol)
        assert np.allclose(back, np.clip(z, -2, 2), atol=0.15), (z, hexcol, back)
    print("ok: oklab hex roundtrip")


def test_blend_weights():
    s = build_session(WORDS)
    coords = s.coords_of(s.vectors[3])  # Link's own position
    blend = s.blend_weights(coords)
    assert blend["relative_error"] < 0.35, blend["relative_error"]
    top = max(blend["weights"], key=lambda p: blend["weights"][p])
    assert top == "Link", (top, blend["weights"])
    print(f"ok: blend weights (err {blend['relative_error']:.3f}, top={top})")


def test_layer_stack():
    """Per-layer sessions share landmarks, differ in geometry, and adjacent
    layers stay closer than distant ones (the scrub morphs, not teleports)."""
    stack = LayerStack(FakeEmbedder(dim=256))
    for w in WORDS:
        stack.add_landmark(w)

    def coord_map(layer):
        return {e["text"]: np.array(e["coords"][:3])
                for e in stack.session(layer).layout()["landmarks"]}

    top, mid, low = coord_map(None), coord_map(18), coord_map(0)
    assert set(top) == set(mid) == set(low) == set(WORDS)
    # primary IS the top layer for a "last"-configured embedder
    assert stack.session(36) is stack.primary

    def dist(a, b):
        return sum(np.linalg.norm(a[w] - b[w]) for w in WORDS)

    d_adjacent = dist(coord_map(16), mid)
    d_far = dist(low, top)
    assert d_far > 1e-3, "layers should differ in geometry"
    assert d_adjacent < d_far, (d_adjacent, d_far)
    # landmark added after sessions exist propagates everywhere
    stack.add_landmark("Yoshi")
    for layer in (None, 0, 16, 18):
        assert "Yoshi" in [e["text"] for e in stack.session(layer).layout()["landmarks"]]
    stack.remove_landmark("Yoshi")
    assert all("Yoshi" not in stack.session(l).phrases for l in (None, 0, 16, 18))
    print(f"ok: layer stack (adjacent drift {d_adjacent:.1f} < far drift {d_far:.1f})")


def test_trails_stop_at_the_sampler_depth():
    """The post-norm final state is charted only when it is the depth the
    sampler actually reads. It is the one normed state in the stack, so on any
    other config it is a meaningless kink on the end of every trail."""

    class Sampler35(FakeEmbedder):
        def sampler_layers(self):
            return (35,)

    class Krea(FakeEmbedder):
        def sampler_layers(self):
            return (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    for emb, top in ((FakeEmbedder(dim=64, n_layers=37), 36),
                     (Sampler35(dim=64, n_layers=37), 35),
                     (Krea(dim=64, n_layers=37), 35)):
        stack = LayerStack(emb)
        stack.add_landmarks(["a", "b", "c"])
        res = stack.trajectories()
        assert res["depths"][0] == 0
        assert res["depths"][-1] == top, (type(emb).__name__, res["depths"][-1])
        # n_layers is the point count the viewport iterates, and every trail
        # must be exactly that long or the client indexes off the end.
        assert res["n_layers"] == len(res["depths"]) == top + 1
        for points in res["trajectories"].values():
            assert len(points) == res["n_layers"]
    print("ok: trails stop at the sampler depth (36 charted only if sampled)")


def test_band_configured_stack():
    """A band TOKENCOLLIDER_LAYER (e.g. '18-34') must resolve, not int()-crash: the
    primary session's depth is the (lo, hi) pair, and the first /layout of a
    fresh viewport (no layer param) reports it as [lo, hi]."""
    fake = FakeEmbedder(dim=64)
    fake.layer = "18-34"
    stack = LayerStack(fake)
    for w in WORDS[:4]:
        stack.add_landmark(w)
    assert stack.primary_index() == (18, 34)
    assert stack.resolve_layer(None) == [18, 34]
    assert stack.session((18, 34)) is stack.primary
    assert stack.layer_key(None) == "18-34"

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stack, token=TOKEN))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/layout", headers=AUTH)) as r:
        lay = json.loads(r.read())
    assert lay["layer"] == [18, 34] and lay["n_landmarks"] == 4, lay["layer"]
    server.shutdown()
    server.server_close()
    print("ok: band-configured embedder serves")


def test_http_server():
    stack = LayerStack(FakeEmbedder(dim=128))
    tmp_root = tempfile.mkdtemp()
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(stack, export_root=Path(tmp_root), token=TOKEN))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json", **AUTH},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert call("GET", "/health")[1]["ok"] is True
    for w in ["red", "blue", "crimson", "azure", "scarlet", "cobalt", "rust"]:
        status, lay = call("POST", "/landmarks", {"text": w})
        assert status == 200
    assert lay["n_landmarks"] == 7 and lay["n_axes"] == 6
    assert lay["n_layers"] == 37 and lay["layer"] == 36
    # The cook stop rides in the payload, apart from the charted depth: a
    # view topping out below it cooks on export. Fake samples its top state.
    assert lay["cook_stop"] == 36 and lay["sampler_layers"] == [36], lay
    entry = lay["landmarks"][0]
    assert set(entry) == {"text", "coords", "color", "residual", "source", "density", "kind"}
    assert entry["kind"] == "phrase"
    assert entry["source"] == "manual"  # typed in over HTTP = a settlement
    densities = [e["density"] for e in lay["landmarks"]]
    assert all(d is None or 0 <= d <= 9 for d in densities), densities
    assert any(d is not None for d in densities)  # fallback path still scores
    assert entry["color"].startswith("#") and len(entry["color"]) == 7
    assert lay["layer_min"] is None and lay["layer_max"] is None

    # layer scrubbing over HTTP: same members, different geometry, sane errors
    status, low = call("GET", "/layout?layer=4")
    assert status == 200 and low["layer"] == 4 and low["n_landmarks"] == 7
    assert [e["text"] for e in low["landmarks"]] == [e["text"] for e in lay["landmarks"]]
    assert any(
        not np.allclose(a["coords"][:3], b["coords"][:3], atol=1e-3)
        for a, b in zip(low["landmarks"], lay["landmarks"])
    ), "layer 4 should lay out differently from layer 36"
    status, _ = call("GET", "/layout?layer=banana")
    assert status == 400
    status, _ = call("GET", "/layout?layer=999")
    assert status == 400

    # layer ranges: averaged view differs from both endpoints, shares members
    status, rng_lay = call("GET", "/layout?layer=0-8")
    assert status == 200 and rng_lay["layer"] == [0, 8], rng_lay["layer"]
    status, lo_lay = call("GET", "/layout?layer=0")
    assert status == 200
    assert [e["text"] for e in rng_lay["landmarks"]] == [e["text"] for e in lay["landmarks"]]
    assert any(
        not np.allclose(r["coords"][:3], s["coords"][:3], atol=1e-3)
        for r, s in zip(rng_lay["landmarks"], lo_lay["landmarks"])
    ), "0-8 average should differ from layer 0 alone"
    status, body_rng = call("POST", "/blend", {"coords": entry["coords"], "layer": [0, 8]})
    assert status == 200 and "red" in body_rng["weights"]
    status, _ = call("GET", "/layout?layer=8-0")  # reversed bounds normalize
    assert status == 200

    # constellation: one trail per landmark, one point per depth, real motion
    status, traj = call("GET", "/trajectory")
    assert status == 200 and traj["n_layers"] == 37 and traj["extent"] > 0
    assert set(traj["trajectories"]) == {e["text"] for e in lay["landmarks"]}
    for path in traj["trajectories"].values():
        assert len(path) == 37
        for p in path:
            assert len(p["coords"]) == 6
            assert p["color"].startswith("#") and len(p["color"]) == 7
    a_path = traj["trajectories"]["red"]
    assert not np.allclose(a_path[0]["coords"][:3], a_path[36]["coords"][:3], atol=1e-3), \
        "trajectory endpoints should differ across depth"
    # a single-layer view's sphere sits exactly on its trail point
    status, l9 = call("GET", "/layout?layer=9")
    by_text = {e["text"]: e for e in l9["landmarks"]}
    assert np.allclose(
        by_text["red"]["coords"], traj["trajectories"]["red"][9]["coords"], atol=1e-5
    ), "layer-9 layout must match trajectory point 9"

    status, q = call("POST", "/interrogate", {"coords": entry["coords"], "k": 3})
    assert status == 200
    assert q["nearest_landmarks"][0]["text"] == "red", q["nearest_landmarks"]

    status, b = call("POST", "/blend", {"coords": entry["coords"]})
    assert status == 200 and "red" in b["weights"]

    from safetensors import safe_open
    if True:
        # Request paths are RELATIVE to the operator-chosen root; absolute
        # paths from a request are refused (see test_server_refuses_hostile_requests).
        out = str(Path(tmp_root) / "point.safetensors")
        status, exp = call("POST", "/export",
                           {"coords": entry["coords"], "path": "point.safetensors"})
        assert status == 200 and exp["path"] == out, exp
        with safe_open(out, framework="np") as f:
            tensor = f.get_tensor("conditioning")
            meta = f.metadata()
        assert list(tensor.shape) == exp["shape"] and tensor.shape[0] == 1
        assert "red" in json.loads(meta["weights"])
        # provenance rides in the file: config + live universe fingerprint
        from tokencollider import provenance
        assert meta["model"] == "fake" and meta["pooling"] == "mean"
        assert meta["universe"] == provenance.fingerprint(
            "fake", "last", "mean", "{}", [e["text"] for e in lay["landmarks"]])
    # default export name: CLOSEST_NEXT_layersKEY (cursor sits on "red")
    status, exp = call("POST", "/export", {"coords": entry["coords"]})
    assert status == 200
    auto = Path(exp["path"])
    # An auto-named export obeys the operator's export root like a named one.
    assert auto.parent == Path(tmp_root).resolve(), auto
    # "last" resolves to a depth in the key: filenames name the real layer.
    assert auto.name.startswith("red_") and "_layer36" in auto.name, auto.name
    auto.unlink()

    # cook-depth stack: one cursor, several injection depths, distinct files
    status, stk = call("POST", "/export",
                       {"coords": entry["coords"], "layer": [0, 8], "stack": True})
    assert status == 200 and len(stk["stack"]) == len(stk["depths"]) == 4, stk.get("depths")
    assert stk["depths"] == sorted(stk["depths"], reverse=True)
    stack_paths = [r["path"] for r in stk["stack"]]
    assert len(set(stack_paths)) == 4
    # one stack, one tag: every file shares the prefix before _cookNN
    import re as _re
    prefixes = {_re.sub(r"_cook\d+$", "", Path(p).stem) for p in stack_paths}
    assert len(prefixes) == 1 and "_stack" in prefixes.pop(), stack_paths
    # the same stack again rewrites its own files, no per-file digests
    status, again = call("POST", "/export",
                         {"coords": entry["coords"], "layer": [0, 8], "stack": True})
    assert [r["path"] for r in again["stack"]] == stack_paths, again["stack"]
    # a different cursor is a different stack, even in the same neighborhood
    moved = [c + 0.05 for c in entry["coords"]]
    status, other = call("POST", "/export",
                         {"coords": moved, "layer": [0, 8], "stack": True})
    other_paths = [r["path"] for r in other["stack"]]
    assert not set(other_paths) & set(stack_paths), other_paths
    for p in stack_paths + other_paths:
        assert "_cook" in Path(p).name
        Path(p).unlink()
    # stackless view (no band anywhere): degrades to a single export
    status, one = call("POST", "/export",
                       {"coords": entry["coords"], "stack": True})
    assert status == 200 and len(one["stack"]) == 1 and one["depths"] == []
    Path(one["stack"][0]["path"]).unlink()
    # full sweep: one file per depth up to the sampler layer
    status, sweep = call("POST", "/export",
                         {"coords": entry["coords"], "stack": True, "sweep": True})
    assert status == 200 and sweep["depths"] == list(range(35, -1, -1))
    assert len(sweep["stack"]) == 36
    for r in sweep["stack"]:
        # out of glob range, and still under the export root
        assert Path(r["path"]).parent == Path(tmp_root).resolve() / "sweeps"
        Path(r["path"]).unlink()

    # mirror: the negative half is a different tensor, beside the positive
    from safetensors.numpy import load_file
    status, pair = call("POST", "/export", {"coords": entry["coords"], "mirror": True})
    assert status == 200 and pair["mirror"]["role"] == "negative", pair
    pos, neg = load_file(pair["path"]), load_file(pair["mirror"]["path"])
    assert any(not np.allclose(pos[k], neg[k]) for k in pos), "mirror equals positive"
    Path(pair["path"]).unlink()
    Path(pair["mirror"]["path"]).unlink()
    # ...but the centre is its own reflection: refused, and nothing written
    before = set(Path(tmp_root).iterdir())
    status, err = call("POST", "/export",
                       {"coords": entry["coords"], "mirror": True, "center": True})
    assert status == 400 and "centre" in err["error"], (status, err)
    assert set(Path(tmp_root).iterdir()) == before, "a half of the pair was written"

    # universe snapshot: named file, header + phrase-per-line, name sanitized
    from tokencollider import server as srv
    status, uni = call("POST", "/export_universe", {"name": "../evil/smoketest"})
    assert status == 200 and uni["n_phrases"] == 7
    saved = Path(uni["path"])
    assert saved.parent == srv.UNIVERSE_DIR and saved.name == "smoketest.txt", uni
    saved_phrases, saved_meta = provenance.read_universe(saved)
    assert saved_phrases == [e["text"] for e in lay["landmarks"]]
    assert saved_meta["fingerprint"] == uni["fingerprint"]
    assert saved_meta["manual"] == saved_phrases  # all typed in by hand
    saved.unlink()

    # session snapshot: view state rides the header, comes back on load,
    # and never perturbs the fingerprint
    view = {"layer_lo": 4, "layer_hi": 8, "normalized": True,
            "camera_pos": [1, 2, 3], "cursor": [0.5, -1.0, 2.0]}
    status, ses = call("POST", "/export_universe",
                       {"name": "smokesession", "view": view})
    assert status == 200 and ses["fingerprint"] == uni["fingerprint"]
    ses_path = Path(ses["path"])
    ses_phrases, ses_meta = provenance.read_universe(ses_path)
    assert ses_meta["view"] == view and ses_phrases == saved_phrases
    # a fresh stack loading this file serves the view back in /layout
    from tokencollider.layout import LayerStack as LS
    stack2 = LS(FakeEmbedder(dim=128))
    stack2.add_landmarks(ses_phrases, source="preload")
    stack2.provenance = ses_meta
    server2 = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stack2, token=TOKEN))
    port2 = server2.server_address[1]
    threading.Thread(target=server2.serve_forever, daemon=True).start()
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port2}/layout", headers=AUTH)) as r:
        lay2 = json.loads(r.read())
    assert lay2["view"] == view
    server2.shutdown()
    server2.server_close()
    ses_path.unlink()

    # image landmarks: a picture registers by content, lands in the chart
    # with its thumbnail served, and round-trips through a universe file as
    # an @path line.
    from PIL import Image as PILImage
    from tokencollider import images as imgs
    pic_dir = Path(tempfile.mkdtemp())
    for name, rgb in (("sunset.png", (240, 120, 40)), ("sea.jpg", (30, 80, 200))):
        PILImage.new("RGB", (96, 64), rgb).save(pic_dir / name)
    status, lay_i = call("POST", "/landmarks", {"images": [str(pic_dir / "sunset.png")]})
    assert status == 200, lay_i
    pics = [e for e in lay_i["landmarks"] if e["kind"] == "image"]
    assert len(pics) == 1 and pics[0]["label"] == "sunset", pics
    assert pics[0]["text"].startswith("image:") and pics[0]["image"].startswith("/image/")
    status, lay_i = call("POST", "/landmarks", {"text": "@" + str(pic_dir)})  # a directory: both
    assert status == 200
    pics = {e["label"]: e for e in lay_i["landmarks"] if e["kind"] == "image"}
    assert set(pics) == {"sunset", "sea"}, set(pics)
    req = urllib.request.Request(f"http://127.0.0.1:{port}" + pics["sea"]["image"],
                                 headers=AUTH)
    with urllib.request.urlopen(req) as r:
        assert r.headers["Content-Type"] == "image/jpeg"
        thumb = r.read()
    assert thumb[:2] == b"\xff\xd8"  # a JPEG
    status, uni_i = call("POST", "/export_universe", {"name": "smoke_images"})
    lines = Path(uni_i["path"]).read_text(encoding="utf-8").splitlines()
    at_lines = [l for l in lines if l.startswith("@")]
    assert len(at_lines) == 2 and all(Path(l[1:]).exists() for l in at_lines), at_lines
    # ...and back: the same keys, by content, regardless of pooling spelling
    keys = imgs.resolve_entries(at_lines, Path(uni_i["path"]).parent, None)
    assert set(keys) == {pics["sunset"]["text"], pics["sea"]["text"]}
    tails = imgs.resolve_entries(at_lines, Path(uni_i["path"]).parent, None, "tail")
    assert all(t.startswith("image/tail:") for t in tails) and imgs.key_mode(tails[0]) == "tail"
    assert imgs.key_sha(tails[0]) == imgs.key_sha(keys[0]) or imgs.key_sha(tails[0]) == imgs.key_sha(keys[1])
    # A missing or corrupt picture is skipped, never fatal to the load.
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "corrupt.png").write_bytes(b"not a png")
        kept = imgs.resolve_entries(["word", "@corrupt.png", "@gone.png"], Path(d), None)
    assert kept == ["word"], kept
    Path(uni_i["path"]).unlink()
    status, _ = call("POST", "/landmarks/remove", {"texts": [pics["sunset"]["text"], pics["sea"]["text"]]})
    assert status == 200
    status, _ = call("GET", "/image/deadbeef")
    assert status == 404

    # gpu release: FakeEmbedder has no model; endpoint answers gracefully
    status, rel = call("POST", "/gpu/release", {})
    assert status == 200 and rel == {"released": False}, rel

    # colonize: nearest vocab + settlements become a live child universe
    vocab_words = ["ruby", "sapphire", "brick", "ocean", "flame", "steel"]
    fake = stack.embedder
    vocab = (vocab_words, np.stack([fake.embed(w) for w in vocab_words]))
    stack.vocab = lambda layer=None: vocab
    # Before the zoom: no world exists, and asking for world axes is a no-op.
    status, pre = call("GET", "/layout?axes=world")
    assert status == 200 and pre["has_world"] is False and pre["axes"] == "local"
    assert pre["center"]["coords"][:3] == [0.0, 0.0, 0.0]  # own chart: the origin
    # Where the natives sit in the atlas's chart, as the interrogator reports
    # them, is where they must still sit after the zoom in world axes.
    status, q = call("POST", "/interrogate", {"coords": entry["coords"], "k": 6})
    atlas_coords = {e["text"]: e["coords"] for e in q["nearest_vocab"]}
    status, col = call("POST", "/colonize", {"coords": entry["coords"], "k": 3})
    assert status == 200 and col["loaded"] is True, col
    # The zoom arrives in world axes with the beachhead still meaningful.
    lay_w = col["layout"]
    assert lay_w["axes"] == "world" and lay_w["has_world"] is True, lay_w["axes"]
    world_coords = {e["text"]: e["coords"] for e in lay_w["landmarks"]}
    for w in col["natives"]:
        assert np.allclose(world_coords[w], atlas_coords[w], atol=1e-5), w
    # Header: the world is the atlas, the parent's axes are recorded.
    assert col["layout"]["center"]["coords"][:3] != [0.0, 0.0, 0.0]
    # Local axes are a different chart of the same landmarks, centred.
    status, lay_l = call("GET", "/layout?axes=local")
    assert lay_l["axes"] == "local" and lay_l["has_world"] is True
    assert lay_l["center"]["coords"][:3] == [0.0, 0.0, 0.0]
    local_coords = {e["text"]: e["coords"] for e in lay_l["landmarks"]}
    assert not np.allclose(local_coords[col["natives"][0]],
                           world_coords[col["natives"][0]])
    # The centre cursor names the landmark mean itself, so its blend is the
    # uniform recipe in either chart. Lifting the centre's world-axes
    # coords instead would give the mean's projection onto the atlas's six
    # directions, a different point, which is why "center" is a flag.
    n_child = lay_w["n_landmarks"]
    for axes_mode, lay_x in (("world", lay_w), ("local", lay_l)):
        status, bc = call("POST", "/blend", {"coords": lay_x["center"]["coords"],
                                             "axes": axes_mode, "center": True})
        assert status == 200
        assert all(abs(w - 1.0 / n_child) < 1e-3 for w in bc["weights"].values()), (axes_mode, bc["weights"])
    status, lifted = call("POST", "/blend", {"coords": lay_w["center"]["coords"], "axes": "world"})
    assert not all(abs(w - 1.0 / n_child) < 1e-3 for w in lifted["weights"].values())
    assert len(col["natives"]) == 3 and set(col["natives"]) <= set(vocab_words)
    assert set(col["manual"]) == {"red", "blue", "crimson", "azure", "scarlet",
                                  "cobalt", "rust"}
    child = Path(col["path"])
    child_phrases, child_meta = provenance.read_universe(child)
    assert set(child_phrases) == set(col["natives"]) | set(col["manual"])
    assert child_meta["parent"]["coords"] == [float(c) for c in entry["coords"]]
    # parent records the depth the neighborhood was carved at, not "last"
    assert child_meta["parent"]["layer"] == "36"
    assert child_meta["parent"]["axes"] == "local"
    assert child_meta["world"]["fingerprint"] == child_meta["parent"]["fingerprint"]
    # A later session rebuilds the world from the file the header names.
    atlas_path = Path(d_world := tempfile.mkdtemp()) / "atlas.txt"
    from tokencollider import provenance as prov
    prov.write_universe(atlas_path, ["red", "blue", "crimson", "azure",
                                     "scarlet", "cobalt", "rust"],
                        prov.build_meta(stack.embedder, ["red", "blue"]))
    cold = LayerStack(FakeEmbedder(dim=128))
    cold.add_landmarks(child_phrases)
    cold.provenance = {"world": {"path": str(atlas_path)}}
    assert cold.world is None and cold.has_world() is True
    basis = cold.basis(None, "world")
    assert basis is not None and cold.world.source_path == str(atlas_path)
    assert cold.world.phrases[:2] == ["red", "blue"]
    assert cold.session(None).centroid(basis)["coords"][:3] != [0.0, 0.0, 0.0]
    atlas_path.unlink()
    by_src = {e["text"]: e["source"] for e in col["layout"]["landmarks"]}
    assert all(by_src[w] == "preload" for w in col["natives"])
    assert all(by_src[w] == "manual" for w in col["manual"])
    child.unlink()
    lay = col["layout"]
    entry = lay["landmarks"][0]

    status, _ = call("POST", "/landmarks/remove", {"text": "rust"})
    assert status == 200
    # batch removal (box select): one request, one refit, count reported
    status, batch = call("POST", "/landmarks/remove",
                         {"texts": ["red", "blue", "not-a-landmark"]})
    assert status == 200 and batch["removed"] == 2, batch.get("removed")
    assert all(e["text"] not in ("red", "blue") for e in batch["landmarks"])
    status, err = call("POST", "/interrogate", {"coords": "garbage"})
    assert status in (400, 500), (status, err)
    status, _ = call("GET", "/nope")
    assert status == 404

    server.shutdown()
    server.server_close()
    print("ok: http sidecar end-to-end")


def test_server_refuses_hostile_requests():
    """The sidecar binds a loopback port, so any page you visit can POST to
    it and every local process can reach it. Every door, all shut:

    - a cross-origin form POST cannot set Content-Type: application/json
      without a preflight, and this server answers none and sends no CORS
      headers, so requiring that header blocks CSRF;
    - every API request needs the session token, so neither a web page nor
      another local process can drive it without being handed one;
    - an Origin other than the sidecar's own page is refused;
    - a Host other than 127.0.0.1/localhost on this port is refused, which is
      what stops a DNS-rebinding page that the browser thinks is same-origin;
    - the web build is served by exact name only, with no token (it holds no
      secret) but behind the same Host check;
    - `/export` takes a filename from the request body, which unchecked is an
      arbitrary file write.
    """
    stack = LayerStack(FakeEmbedder(dim=64))
    for w in ("alpha", "beta", "gamma"):
        stack.add_landmark(w)
    web = Path(tempfile.mkdtemp())
    (web / "index.html").write_text("<html>viewport</html>", encoding="utf-8")
    (web / "index.wasm").write_bytes(b"\0asm")
    (web / "notes.txt").write_text("not a web build type", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(stack, token=TOKEN, web_dir=web))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get(path, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers["Content-Type"], r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers["Content-Type"], e.read()

    def raw(payload, headers):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/export", data=json.dumps(payload).encode(),
            method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    coords = [0.0] * 6
    JSON = {"Content-Type": "application/json", **AUTH}

    code, _ = raw({"coords": coords, "path": "x.safetensors"},
                  {"Content-Type": "text/plain", **AUTH})
    assert code == 403, f"form-POST CSRF not refused: {code}"

    code, _ = raw({"coords": coords}, {**JSON, "Origin": "https://evil.example"})
    assert code == 403, f"cross-origin request not refused: {code}"

    # The token: required on POST and on every API GET, /health included.
    code, body = raw({"coords": coords}, {"Content-Type": "application/json"})
    assert code == 403 and "token" in body["error"], (code, body)
    code, _ = raw({"coords": coords}, {**JSON, TOKEN_HEADER: "guess"})
    assert code == 403, f"wrong token not refused: {code}"
    for path in ("/health", "/layout", "/trajectory"):
        assert get(path)[0] == 403, f"{path} served without a token"
        assert get(path, AUTH)[0] == 200, f"{path} refused with the token"

    # Host: a rebinding page's requests name its own domain.
    for host in ("evil.example", f"evil.example:{port}", "127.0.0.1:1"):
        assert get("/layout", {**AUTH, "Host": host})[0] == 403, host
        assert get("/", {"Host": host})[0] == 403, f"page served to Host {host}"
    assert get("/layout", {**AUTH, "Host": f"localhost:{port}"})[0] == 200

    # The sidecar's own page may call it; its Origin is this host and port.
    code, _ = raw({"coords": coords}, {**JSON, "Origin": f"http://127.0.0.1:{port}"})
    assert code == 200, f"own page refused: {code}"

    # The web build: no token, right types, nothing outside the listing.
    code, ctype, page = get("/")
    assert code == 200 and ctype.startswith("text/html") and b"viewport" in page
    code, ctype, _ = get("/index.wasm")
    assert code == 200 and ctype == "application/wasm", (code, ctype)
    for path in ("/notes.txt", "/../server.py", "/%2e%2e/server.py", "/index.wasm/x"):
        code, _, _ = get(path)
        assert code in (403, 404), f"{path} served: {code}"

    for bad in ("/tmp/pwned.safetensors", "../../../../tmp/pwn.safetensors"):
        code, body = raw({"coords": coords, "path": bad}, JSON)
        assert code == 400 and "export path" in body["error"], (bad, code, body)

    assert not Path("/tmp/pwned.safetensors").exists()
    assert not Path("/tmp/pwn.safetensors").exists()
    server.shutdown()
    print("ok: server refuses CSRF, missing tokens, foreign Origin and Host, "
          "stray static paths, and escaping export paths")


def test_ragged_phrases_blend_tail_to_tail():
    """Phrases tokenize to different lengths. The scaffold tail sits at the
    end of every one, so a blend must meet tail to tail; padding the whole
    sequence on the right slid a short phrase's tail onto a long one's
    words."""
    TAIL = 2

    class Ragged(FakeEmbedder):
        def template_tail_tokens(self):
            return TAIL

        def conditioning_layers(self, text, layers):
            n = 3 + len(text) % 3          # 3..5 tokens, then the tail
            body = np.full((n, self.dim), 1.0, np.float32)
            tail = np.full((TAIL, self.dim), 100.0, np.float32)
            return {int(l): np.concatenate([body, tail]) for l in layers}

    s = LayoutSession(Ragged(dim=8))
    for w in ("ab", "abc", "abcd"):        # 5, 3, 4 body tokens
        s.add_landmark(w)
    weights = [0.5, 0.3, 0.2]
    out = s._blend_at([7], weights)[7]
    assert out.shape[0] == 5 + TAIL, out.shape
    # Every landmark's tail blends with every other's: 100 * sum(weights).
    assert np.allclose(out[-TAIL:], 100.0), out[-TAIL:, 0]
    # The slot is padded, never fed a tail: body rows stay small.
    assert (out[:-TAIL] < 2.0).all(), out[:-TAIL, 0]
    print("ok: ragged phrases blend tail to tail")


def test_knn_radius_skips_self_on_float32():
    """Cached vectors are float32. A point's distance to itself must still
    read as zero, or it counts as its own neighbour and every density digit
    shifts by one rank."""
    from tokencollider.layout import knn_radius
    rng = np.random.default_rng(3)
    mat = (rng.standard_normal((40, 256)) * 20).astype(np.float32)
    k = 4
    got = knn_radius(mat, mat, k)
    m = mat.astype(np.float64)
    truth = np.sort(np.linalg.norm(m[:, None] - m[None], axis=2), axis=1)[:, k]
    assert np.allclose(got, truth, rtol=1e-6), np.flatnonzero(~np.isclose(got, truth))
    print("ok: knn_radius skips self on float32 input")


def test_density_caches_keep_one_corpus_per_layer():
    stack = LayerStack(FakeEmbedder(dim=8))
    cache = {("20", 10): {"a": 1.0}, ("20", 11): {"a": 1.1}, ("35", 10): {"a": 2.0}}
    stack._evict_stale(cache, ("20", 12))
    assert set(cache) == {("35", 10)}, cache
    print("ok: a grown corpus evicts the layer's stale density entries")


def test_server_serialises_concurrent_requests():
    """ThreadingHTTPServer runs handlers in parallel; the layout's arrays are
    updated in several steps. Adds, removes and layout reads racing each
    other must all succeed."""
    stack = LayerStack(FakeEmbedder(dim=256))
    for w in ("a0", "a1", "a2", "a3", "a4", "a5", "a6"):
        stack.add_landmark(w)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stack, token=TOKEN))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json", **AUTH})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    codes = []
    def churn():
        for i in range(40):
            codes.append(call("POST", "/landmarks", {"text": f"x{i}"}))
            codes.append(call("POST", "/landmarks/remove", {"text": f"x{i}"}))
    def read():
        for _ in range(80):
            codes.append(call("GET", "/layout"))
    threads = [threading.Thread(target=churn)] + [threading.Thread(target=read)
                                                  for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.shutdown()
    server.server_close()
    assert codes and set(codes) == {200}, sorted(set(codes))
    print(f"ok: {len(codes)} concurrent requests, all answered 200")


if __name__ == "__main__":
    test_reconstruct_roundtrip()
    test_layout_stability()
    test_procrustes_beats_none()
    test_small_counts()
    test_oklab_roundtrip()
    test_blend_weights()
    test_layer_stack()
    test_trails_stop_at_the_sampler_depth()
    test_band_configured_stack()
    test_http_server()
    test_server_refuses_hostile_requests()
    test_ragged_phrases_blend_tail_to_tail()
    test_knn_radius_skips_self_on_float32()
    test_density_caches_keep_one_corpus_per_layer()
    test_server_serialises_concurrent_requests()
    print("all layout/server smoke tests passed")
