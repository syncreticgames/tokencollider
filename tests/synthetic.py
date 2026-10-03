"""Property tests over a universe whose geometry was planted, not discovered.

Every assertion here compares against `tokencollider/synthetic.py`'s ground truth, so a
failure means the blend, chart, or inserter behaved differently from what the
geometry guarantees. No GPU, no model, no cache.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokencollider.compose import bend, poles_to_signs, signed_lstsq
from tokencollider.layout import LayoutSession
from tokencollider.synthetic import SyntheticEmbedder

HOT = ["fire", "flame", "ember", "blaze"]
COLD = ["ice", "frost", "snow", "chill"]
TOOL = ["hammer", "wrench", "drill", "chisel"]


def opposed(**kw):
    return SyntheticEmbedder({"hot": HOT, "cold": COLD, "tool": TOOL},
                             opposites=[("hot", "cold")], **kw)


def unopposed(**kw):
    return SyntheticEmbedder({"hot": HOT, "cold": COLD, "tool": TOOL}, **kw)


def session(emb, layer=None):
    # Charts are built at the semantic peak, matching krea2 charting at 20.
    layer = emb.peak_layer() if layer is None else layer
    s = LayoutSession(emb, layer=layer)
    for p in emb.phrases():
        s.add_landmark(p)
    return s


def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def test_clusters_are_planted_and_separable():
    """At a high layer, phrases sit nearer their own cluster than any other."""
    e = opposed()
    peak = e.peak_layer()
    for name, members in e.clusters.items():
        own = np.mean([e.embed_layer(p, peak) for p in members], axis=0)
        for other, others in e.clusters.items():
            if other == name:
                continue
            away = np.mean([e.embed_layer(p, peak) for p in others], axis=0)
            for p in members:
                v = e.embed_layer(p, peak)
                assert cos(v, own) > cos(v, away), (p, name, other)
    print(f"ok: planted clusters separate at the peak layer ({peak})")


def test_the_top_layer_loses_the_concept():
    """The sweep found layer 35 resembles layer 00: both ends are token-shaped
    and the middle is where meaning lives. The synthetic model has to show the
    same, or tests built on it will not catch a chart aimed at a dead layer."""
    e = opposed()
    def margin(l):
        hot = np.mean([e.embed_layer(p, l) for p in HOT], axis=0)
        cold = np.mean([e.embed_layer(p, l) for p in COLD], axis=0)
        return cos(hot, cold)
    peak, top = margin(e.peak_layer()), margin(max(e.sampler_layers()))
    assert peak < top, (peak, top)   # opposed clusters are furthest apart at the peak
    print(f"ok: hot/cold cosine {peak:+.2f} at the peak, {top:+.2f} at the top")


def test_low_layers_group_by_surface_not_cluster():
    """The U shape: low layers group by shared prefix, high layers by cluster.
    Scrubbing has to actually change the grouping or the viewport is lying."""
    e = SyntheticEmbedder({"a": ["firefly", "fireman"], "b": ["fireside", "zebra"]},
                          dim=64)
    lo, hi = 1, e.peak_layer()
    # "firefly"/"fireside" share a prefix but sit in different clusters.
    surface_lo = cos(e.embed_layer("firefly", lo), e.embed_layer("fireside", lo))
    surface_hi = cos(e.embed_layer("firefly", hi), e.embed_layer("fireside", hi))
    cluster_lo = cos(e.embed_layer("firefly", lo), e.embed_layer("fireman", lo))
    cluster_hi = cos(e.embed_layer("firefly", hi), e.embed_layer("fireman", hi))
    assert surface_hi < surface_lo, (surface_lo, surface_hi)
    assert cluster_hi > cluster_lo, (cluster_lo, cluster_hi)
    print("ok: grouping migrates from surface to cluster as layers deepen")


def test_negative_weight_today_is_uniform_re_centering():
    """FINDING (08232026). The red tethers carry no semantic signal.

    Extrapolate past any cluster and unconstrained least squares pays for the
    overshoot by subtracting every other landmark EQUALLY, whether or not a
    genuine opposition axis exists. Planted antonym and planted-independent
    clusters take identical negative mass, and `centroid_alignment` correctly
    reports 1.00 for both, meaning "this is only re-centering".

    So a red line means "the cursor is outside the landmark hull", not "this
    landmark is the opposite of what you asked for". A positive/negative
    neighborhood cannot be read out of the current solve; it has to be put in.
    """
    def masses(opposites):
        e = SyntheticEmbedder({"hot": HOT, "cold": COLD, "tool": TOOL},
                              opposites=opposites)
        s = session(e)
        peak = e.peak_layer()
        hot = np.mean([e.embed_layer(p, peak) for p in HOT], axis=0)
        target = s.mean + 2.0 * (hot - s.mean)
        res = s.blend_weights(s.coords_of(target), None, target)
        w = res["weights"]
        return (sum(w[p] for p in COLD), sum(w[p] for p in TOOL),
                res["negativity"]["centroid_alignment"])

    cold_a, tool_a, align_a = masses([("hot", "cold")])   # planted antonym
    cold_b, tool_b, align_b = masses([])                  # no antonym at all
    assert cold_a < 0 and tool_a < 0
    assert abs(cold_a - tool_a) < 1e-6, (cold_a, tool_a)  # split evenly
    assert abs(cold_a - cold_b) < 1e-6, (cold_a, cold_b)  # antonym changes nothing
    assert align_a > 0.99 and align_b > 0.99, (align_a, align_b)
    print(f"ok: negative mass is uniform re-centering "
          f"(cold {cold_a:+.2f} == tool {tool_a:+.2f}, align {align_a:.2f}); "
          f"opposition is not expressible in the current solve")


def test_a_signed_solve_could_express_opposition():
    """The other half of the finding: the geometry DOES carry the opposition,
    so the information is there and only the solve throws it away. Projecting
    the target onto the planted axis recovers hot and cold with opposite signs,
    while an independent cluster stays near zero."""
    e = SyntheticEmbedder({"hot": HOT, "cold": COLD, "tool": TOOL},
                          opposites=[("hot", "cold")])
    peak = e.peak_layer()
    centre = {g: np.mean([e.embed_layer(p, peak) for p in ps], axis=0)
              for g, ps in (("hot", HOT), ("cold", COLD), ("tool", TOOL))}
    axis = e.axis_of("hot")
    proj = {g: float(v @ axis / (np.linalg.norm(axis) ** 2))
            for g, v in centre.items()}
    assert proj["hot"] > 0 > proj["cold"], proj
    assert abs(proj["tool"]) < 0.5 * min(abs(proj["hot"]), abs(proj["cold"])), proj
    print(f"ok: the opposition survives in the geometry "
          f"(hot {proj['hot']:+.2f}, cold {proj['cold']:+.2f}, "
          f"tool {proj['tool']:+.2f})")


def test_modalities_peak_at_different_layers():
    """Krea 2 reads a vision tower and a text tower through one stack, so there
    is no single layer where all meaning lives. Pictures and words never
    coincide at any layer; this holds the code to it.

    The consequence is the part that matters: a universe mixing pictures and
    words has NO chart layer that serves both. Whatever the viewport charts at
    is a compromise, and which compromise should be a measured choice rather
    than the text answer inherited from Z-Image.
    """
    words = ["fire", "flame", "ember"]
    pics = ["image:harbour.png", "image:kestrel.png", "image:violin.png"]
    e = SyntheticEmbedder({"hot": words, "shot": pics},
                          peaks={"text": 18, "image": 28})
    assert e.modality_of("fire") == "text"
    assert e.modality_of("image:a.png") == "image"
    assert e.peak_layer("text") != e.peak_layer("image")

    def tightness(members, layer):
        vs = [e.embed_layer(m, layer) for m in members]
        c = np.mean(vs, axis=0)
        return float(np.mean([cos(v, c) for v in vs]))

    # Each modality is tightest at its own peak, not at the other's.
    for members, mine, theirs in ((words, 18, 28), (pics, 28, 18)):
        assert tightness(members, mine) > tightness(members, theirs), (mine, theirs)

    # And no layer is best for both at once.
    best_text = max(range(1, 36), key=lambda l: tightness(words, l))
    best_pics = max(range(1, 36), key=lambda l: tightness(pics, l))
    # Both peaks must land near what was planted, not at a low-layer artefact.
    assert abs(best_text - 18) <= 2 and abs(best_pics - 28) <= 2, (best_text, best_pics)
    assert best_text != best_pics, (best_text, best_pics)
    print(f"ok: text peaks at {best_text}, pictures at {best_pics}; "
          f"a mixed universe has no single right chart layer")


def test_export_carries_every_sampler_layer():
    e = opposed()
    s = session(e)
    cond, blend = s.export_conditioning(s.coords_of(e.embed("fire")))
    assert list(cond.layers) == list(e.sampler_layers()), cond.layers
    assert cond.seq_len > 0 and not blend["cooked"]
    print(f"ok: export carries all {len(cond)} sampler layers, uncooked at the top")


def test_low_inserter_reinterprets_more_than_a_high_one():
    """The claim: a low inserter paraphrases and a high one
    transcribes. Measure it: the deeper the placement, the further the top
    layer's state travels from the raw blend it was placed as."""
    e = opposed()
    s = session(e)
    coords = s.coords_of(e.embed("fire"))
    top = max(e.sampler_layers())
    drift = {}
    for hi in (8, 26):
        cond, blend = s.export_conditioning(coords, band=(hi, hi))
        assert blend["cooked"], hi
        raw, cooked = s._blend_at([hi], list(
            s.blend_weights(coords, None, None)["weights"].values()))[hi], cond[top]
        n = min(raw.shape[0], cooked.shape[0])
        drift[hi] = float(np.linalg.norm(cooked[:n] - raw[:n]) /
                          (np.linalg.norm(raw[:n]) or 1.0))
    assert drift[8] > drift[26], drift
    print(f"ok: inserter 8 drifts {drift[8]:.2f} vs inserter 26 at {drift[26]:.2f}")


def test_signed_solve_makes_one_neighborhood_negative():
    """The composition ask, and the answer is `strict`.

    Three solves on one target, pushed past the hot cluster:

    - free: cold and tool both take -0.333. Indistinguishable, error 0.000.
    - poles alone: IDENTICAL to free. Naming cold as the negative pole changes
      nothing, because the free solve already drove it negative. The
      constraint was satisfied before it was applied.
    - poles + strict: tool is held at +0.000 and cold is the only subtractor,
      at an error of 0.164.

    So a negative neighborhood is not a labelling of what the solve already
    did. It is a constraint that has to exclude everyone else, and it costs
    accuracy, which is the honest trade and is reported.
    """
    e = opposed()
    s = session(e)
    peak = e.peak_layer()
    hot = np.mean([e.embed_layer(p, peak) for p in HOT], axis=0)
    target = s.mean + 2.0 * (hot - s.mean)
    coords = s.coords_of(target)

    def mass(**kw):
        r = s.blend_weights(coords, None, target, **kw)
        w = r["weights"]
        return (sum(w[p] for p in HOT), sum(w[p] for p in COLD),
                sum(w[p] for p in TOOL), r["relative_error"], r)

    _, c_free, t_free, e_free, _ = mass()
    _, c_pole, t_pole, e_pole, _ = mass(positive=HOT, negative=COLD)
    h_str, c_str, t_str, e_str, r_str = mass(positive=HOT, negative=COLD,
                                             strict=True)

    assert abs(c_free - t_free) < 1e-6                  # free tells them apart: no
    assert abs(c_pole - c_free) < 1e-6                  # poles alone: no change
    assert abs(t_pole - t_free) < 1e-6
    assert t_str >= -1e-9 and c_str < -1e-3, (t_str, c_str)   # strict: cold alone
    assert h_str > 0
    assert r_str["poles"]["strict"] is True
    assert e_str > e_free                               # and it costs something
    print(f"ok: strict makes cold the only subtractor "
          f"(free cold {c_free:+.2f} == tool {t_free:+.2f}; "
          f"strict cold {c_str:+.2f}, tool {t_str:+.2f}, err {e_free:.3f}->{e_str:.3f})")


def test_signed_solve_costs_accuracy_and_says_so():
    """A constraint cannot improve the fit. The relative error must rise, and
    the number is reported so a bad pole choice is visible rather than silent."""
    e = opposed()
    s = session(e)
    peak = e.peak_layer()
    hot = np.mean([e.embed_layer(p, peak) for p in HOT], axis=0)
    target = s.mean + 2.0 * (hot - s.mean)
    coords = s.coords_of(target)
    free = s.blend_weights(coords, None, target)["relative_error"]
    # Deliberately wrong poles: force the cluster we are aiming AT to subtract.
    wrong = s.blend_weights(coords, None, target, strict=True,
                            positive=COLD, negative=HOT)["relative_error"]
    right = s.blend_weights(coords, None, target, strict=True,
                            positive=HOT, negative=COLD)["relative_error"]
    assert free <= right + 1e-9 <= wrong, (free, right, wrong)
    print(f"ok: error reports the cost of the constraint "
          f"(free {free:.3f}, right poles {right:.3f}, wrong poles {wrong:.3f})")


def test_bending_erases_an_association_and_can_amplify_it():
    """Redirecting an attraction, not just reading it. Bend the target off the
    tool direction and the export stops carrying it; bend with a negative
    amount and it carries more. The cluster being aimed at survives both."""
    e = opposed()
    s = session(e)
    peak = e.peak_layer()
    # A point that genuinely leans on both hot and tool.
    hot = np.mean([e.embed_layer(p, peak) for p in HOT], axis=0)
    tool = np.mean([e.embed_layer(p, peak) for p in TOOL], axis=0)
    target = s.mean + (hot - s.mean) + (tool - s.mean)
    coords = s.coords_of(target)

    # Bend acts on the target's geometry, and it promises exactly one thing:
    # the component along the bent direction moves, and everything orthogonal
    # to it is untouched. Cluster directions are NOT orthogonal to each other
    # (each is a centroid minus a mean that includes the others), so a raw
    # dot-product readout of hot moves as a consequence of tool moving. That
    # is arithmetic, not leakage, and it is why compose.bend orthonormalises
    # its directions in the order given.
    u = s.direction(TOOL)
    u = u / np.linalg.norm(u)

    def parts(amount):
        t = target if amount is None else bend(
            target, [s.direction(TOOL)], [amount], origin=s.mean)
        centred = t - s.mean
        along = float(centred @ u)
        return along, centred - along * u

    base_t, base_orth = parts(None)
    er_t, er_orth = parts(1.0)
    loud_t, _ = parts(-1.0)
    assert abs(er_t) < 0.02 * abs(base_t), (base_t, er_t)
    assert loud_t > base_t > er_t, (loud_t, base_t, er_t)
    assert np.allclose(base_orth, er_orth, atol=1e-6)   # nothing else moved
    # The weights follow, but they settle at the group's SHARE OF THE MEAN
    # rather than at zero: the target still sits near the universe mean, and
    # expressing the mean needs every cluster. With three equal clusters that
    # share is 1/3, and erasing tool lands exactly there. Erasing an
    # association removes the group's PULL, it does not delete the group.
    def tool_mass(bends):
        w = s.blend_weights(coords, None, target, bends=bends)["weights"]
        return sum(w[p] for p in TOOL)
    loud_m, base_m, er_m = (tool_mass([(TOOL, -1.0)]), tool_mass(None),
                            tool_mass([(TOOL, 1.0)]))
    share = 1.0 / len(e.clusters)
    assert loud_m > base_m > er_m, (loud_m, base_m, er_m)
    assert abs(er_m - share) < 0.02, (er_m, share)
    print(f"ok: bend moves one direction only "
          f"(tool {base_t:+.2f} -> {er_t:+.2f} erased, {loud_t:+.2f} amplified; "
          f"orthogonal part bit-identical; mass {base_m:.2f} -> "
          f"{er_m:.2f} = its share of the mean)")


def test_bend_and_poles_reach_the_export():
    """Both have to survive the trip into an actual export, or they are only
    a viewport toy."""
    e = opposed()
    s = session(e)
    coords = s.coords_of(np.mean([e.embed_layer(p, e.peak_layer()) for p in HOT],
                                 axis=0))
    cond, blend = s.export_conditioning(coords, positive=HOT, negative=COLD,
                                        bends=[(TOOL, 1.0)])
    assert list(cond.layers) == list(e.sampler_layers())
    assert blend["poles"]["positive"] == sorted(HOT)
    assert blend["bends"][0]["amount"] == 1.0
    assert all(blend["weights"][p] <= 1e-9 for p in COLD), blend["weights"]
    print("ok: poles and bends ride into the export and its metadata")


def test_compare_finds_a_phrase_token_a_global_cosine_hides():
    """The instrument fix. A Krea 2 export is a couple of phrase tokens plus a
    five-token scaffold tail, so a global cosine is mostly scaffold. Plant a
    change in ONE phrase token and check that the global measure misses it
    while Conditioning.compare reports it."""
    from tokencollider.conditioning import Conditioning

    rng = np.random.default_rng(3)
    base = {l: rng.standard_normal((7, 16)).astype(np.float32) for l in (2, 5, 8)}
    a = Conditioning({l: v.copy() for l, v in base.items()})
    moved = {l: v.copy() for l, v in base.items()}
    moved[8][1] *= 0.60                      # one phrase token, 40% norm drop
    b = Conditioning(moved)

    flat_a = np.concatenate([a[l].ravel() for l in sorted(a.layers)])
    flat_b = np.concatenate([b[l].ravel() for l in sorted(b.layers)])
    global_cos = float(flat_a @ flat_b / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b)))
    assert global_cos > 0.98, global_cos      # the whole-tensor view shrugs

    rep = a.compare(b, tail=5)
    assert rep["worst_phrase_token"]["token"] == 1, rep["worst_phrase_token"]
    assert rep["worst_phrase_token"]["layer"] == 8
    ratios = rep["per_token"][8]["norm_ratio"]
    assert abs(ratios[1] - 0.60) < 1e-5, ratios
    assert all(abs(r - 1.0) < 1e-5 for i, r in enumerate(ratios) if i != 1)
    print(f"ok: global cosine {global_cos:.3f} hides it; compare finds token "
          f"{rep['worst_phrase_token']['token']} at layer "
          f"{rep['worst_phrase_token']['layer']}")


def test_preserve_norm_restores_magnitude_and_keeps_direction():
    """blend_tokens(preserve_norm=True) must change ONLY magnitude. Kept as a
    switch even though it measured as a non-lever on Krea 2 (real shortfall
    0.0 to 0.8%, because the hidden states are anisotropic at cosine ~0.98),
    so it stays correct if a less anisotropic encoder ever needs it."""
    from tokencollider.conditioning import blend_tokens

    rng = np.random.default_rng(11)
    a = rng.standard_normal((4, 32)).astype(np.float32) * 2.0
    b = rng.standard_normal((4, 32)).astype(np.float32) * 2.0
    for w in ([0.5, 0.5], [0.7, 0.3], [1.5, -0.5]):
        lin = blend_tokens([a, b], w)
        pn = blend_tokens([a, b], w, preserve_norm=True)
        want = (abs(w[0]) * np.linalg.norm(a, axis=1)
                + abs(w[1]) * np.linalg.norm(b, axis=1)) / (abs(w[0]) + abs(w[1]))
        assert np.allclose(np.linalg.norm(pn, axis=1), want, rtol=1e-4)
        for u, v in zip(lin, pn):
            c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            assert c > 0.9999, c        # direction untouched
    print("ok: preserve_norm restores magnitude only, direction bit-stable")


def test_path_alignment_keeps_the_scaffold_in_place():
    """Ragged phrases are the real obstacle to a path, not the trigonometry.

    A Krea 2 export is a few phrase tokens plus a fixed five-token scaffold
    tail. Phrases differ in length, and zero-padding on the right slides the
    shorter one's tail inward, so a blend averages `<|im_end|>` against a
    phrase token. align() front-aligns the phrase and end-aligns the tail.
    """
    from tokencollider.path import align, path_tokens, slerp

    a = np.zeros((7, 4), np.float32); a[:2] = 1.0; a[2:] = 9.0   # 2 phrase + 5 tail
    b = np.zeros((8, 4), np.float32); b[:3] = 2.0; b[3:] = 9.0   # 3 phrase + 5 tail
    a2, b2, n = align(a, b, 5)
    assert a2.shape == b2.shape == (8, 4) and n == 3
    assert (a2[n:] == 9.0).all() and (b2[n:] == 9.0).all()   # tails meet tails
    assert (a2[2:n] == 0.0).all()                            # pad inside the phrase
    # A naive right-pad would put a's tail at rows 2..6 against b's phrase at 2.
    naive = np.zeros_like(b2); naive[:7] = a
    assert not np.allclose(naive[n:], a2[n:]), "test would not catch the bug"

    # slerp holds the norm where lerp sags through the middle
    rng = np.random.default_rng(5)
    x, y = rng.standard_normal(48) * 3, rng.standard_normal(48) * 3
    mid_s, mid_l = slerp(x, y, 0.5), 0.5 * x + 0.5 * y
    ends = 0.5 * (np.linalg.norm(x) + np.linalg.norm(y))
    assert np.linalg.norm(mid_s) > np.linalg.norm(mid_l)
    assert abs(np.linalg.norm(mid_s) / ends - 1.0) < 0.06
    # endpoints exact, and a zero endpoint degrades to lerp rather than NaN
    assert np.allclose(slerp(x, y, 0.0), x, atol=1e-4)
    assert np.isfinite(slerp(np.zeros(48), y, 0.5)).all()
    print("ok: path alignment keeps scaffold on scaffold; slerp holds the norm")


def test_cooked_mirror_is_named_as_one():
    """An auto-named mirror carries `_mirror` whether or not it was cooked;
    before, a cooked mirror took the positive's exact name plus a hash."""
    import tempfile

    from tokencollider.layout import LayerStack
    from tokencollider.server import export_conditioning

    e = opposed()
    stack = LayerStack(e)
    for p in e.phrases():
        stack.add_landmark(p)
    coords = [0.3] + [0.0] * 5
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        pos = export_conditioning(stack, (2, 8), coords, None, band=(2, 8),
                                  directory=root)
        neg = export_conditioning(stack, (2, 8), [-c for c in coords], None,
                                  band=(2, 8), directory=root, mirror=True)
    assert pos["cooked"] and neg["cooked"]
    assert Path(neg["path"]).stem.endswith("_cook08_mirror"), neg["path"]
    assert not Path(pos["path"]).stem.endswith("_mirror"), pos["path"]
    print("ok: a cooked mirror is named as a mirror")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("all synthetic tests passed")
