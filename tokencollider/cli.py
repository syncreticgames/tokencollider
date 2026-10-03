"""Command line for TokenCollider. `tokencollider` with no command opens the viewport."""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import paths, profiles, provenance
from .embedder import Embedder, FakeEmbedder, load_universe_file
from .store import EmbeddingStore
from .universe import Universe, cosine

DEFAULT_DB = paths.home() / "embeddings.db"


def load_universe(path: Path, embedder) -> tuple[list[str], dict | None]:
    """Read a universe file and refuse it if its recorded config (model /
    layer / pooling / template) doesn't match this session's — coordinates
    from a different frame must never render as if they were commensurable."""
    from . import images

    entries, meta = provenance.read_universe(path)
    phrases = images.resolve_entries(
        entries, path.parent, getattr(embedder, "store", None),
        getattr(embedder, "image_pooling", "image"))
    meta = provenance.verify_universe(path, phrases, meta, embedder)
    return phrases, meta


def build_universe(args, embedder: Embedder) -> Universe:
    phrases, _meta = load_universe(Path(args.universe), embedder)
    vectors = embedder.embed_many(phrases, verbose=True)
    return Universe.build(phrases, vectors, variance=args.variance)


def cmd_warm(args, embedder: Embedder) -> None:
    universe = build_universe(args, embedder)
    print(
        f"cached {len(universe.phrases)} phrases; "
        f"{universe.n_components} components explain "
        f"{universe.explained.sum():.0%} of universe variance"
    )


def cmd_rank(args, embedder: Embedder) -> None:
    universe = build_universe(args, embedder)
    query_vec = embedder.embed(args.query)
    rows = universe.rank(query_vec, exclude=args.query)
    resid = universe.residual_norm(query_vec)
    within = float((universe.project(query_vec) ** 2).sum()) ** 0.5

    print(f"\nuniverse: {args.universe}  ({len(universe.phrases)} members, "
          f"{universe.n_components} components @ {args.variance:.0%} variance)")
    print(f"query: {args.query!r}   in-subspace {within:.1f} / residual {resid:.1f}")
    print(f"\n{'member':<24} {'relative':>9} {'rk':>3}   {'raw':>7} {'rk':>3}   shift")
    for row in rows[: args.top] if args.top else rows:
        shift = row["raw_rank"] - row["relative_rank"]
        marker = f"{shift:+d}" if shift else ""
        print(
            f"{row['phrase']:<24} {row['relative']:>9.4f} {row['relative_rank']:>3}   "
            f"{row['raw']:>7.4f} {row['raw_rank']:>3}   {marker}"
        )


def cmd_compare(args, embedder: Embedder) -> None:
    universe = build_universe(args, embedder)
    q = embedder.embed(args.query)
    a = embedder.embed(args.a)
    b = embedder.embed(args.b)
    qp, ap, bp = (universe.project(v) for v in (q, a, b))

    raw_a, raw_b = cosine(q, a), cosine(q, b)
    rel_a, rel_b = cosine(qp, ap), cosine(qp, bp)

    print(f"\nIs {args.query!r} closer to {args.a!r} than to {args.b!r}?")
    print(f"  raw (full space):        {args.a!r}={raw_a:.4f}  {args.b!r}={raw_b:.4f}"
          f"  -> {'YES' if raw_a > raw_b else 'NO'}")
    print(f"  relative (this universe): {args.a!r}={rel_a:.4f}  {args.b!r}={rel_b:.4f}"
          f"  -> {'YES' if rel_a > rel_b else 'NO'}")
    if (raw_a > raw_b) != (rel_a > rel_b):
        print("  note: the universe lens inverts the raw answer.")


def cmd_axes(args, embedder: Embedder) -> None:
    """Label each universe axis with the wordlist entries most aligned to it.

    Labels rank by distinctiveness — this axis's |cosine| minus the word's
    mean |cosine| over every universe axis — not raw alignment: hub words
    that score ~0.25 against everything ('glasgow') carry no margin and sink,
    while a word with one tall spike wins its axis. (A margin, not a z-score:
    a perfectly flat hub has near-zero std, which would inflate its z.)
    Candidates below --min-cos are dropped entirely; a pole with no survivors
    is honestly unnameable — that reads as a finding, not a bug.
    """
    universe = build_universe(args, embedder)
    words = load_universe_file(Path(args.wordlist))
    vecs = embedder.embed_many(words, verbose=True)
    centered = vecs - universe.mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    # Cosines against ALL kept components, not just the displayed axes — the
    # per-word distribution is the hub detector, so the wider the better.
    cos = (centered @ universe.components.T) / np.clip(norms, 1e-8, None)
    a = np.abs(cos)
    margin = a - a.mean(axis=1, keepdims=True)

    n_axes = min(args.axes, universe.n_components)
    print(f"\nuniverse: {args.universe}  ({len(universe.phrases)} members, "
          f"{universe.n_components} components)")
    print(f"labels:   {args.wordlist}  ({len(words)} candidates, "
          f"min-cos {args.min_cos})")
    for k in range(n_axes):
        print(f"\naxis {k + 1}  ({universe.explained[k]:.0%} of universe variance)")
        for sign, pole in ((1, "+"), (-1, "-")):
            eligible = np.flatnonzero(sign * cos[:, k] >= args.min_cos)
            order = eligible[np.argsort(margin[eligible, k])[::-1]][: args.top]
            labels = "  ".join(
                f"{words[i]} ({sign * cos[i, k]:.2f}, +{margin[i, k]:.2f})" for i in order
            )
            print(f"  {pole}  {labels if len(order) else '(no label above min-cos)'}")


def cmd_forget(args, embedder) -> None:
    """Back bad data out of the cache: delete every row for the given phrases
    (exact text match, all layers/poolings/templates) under this model.
    --pattern sweeps the whole cached vocabulary against regexes (re.search
    semantics)."""
    import re

    if args.model:
        if args.dry_run:
            print(f"[tokencollider] dry run: every cached row for {embedder.model_name} "
                  "would be forgotten")
            return
        e, c = embedder.store.forget_model(embedder.model_name)
        print(f"[tokencollider] forgot model {embedder.model_name}: {e} embedding rows, "
              f"{c} conditioning rows deleted")
        if args.vacuum:
            embedder.store.vacuum()
        return
    texts = list(args.texts)
    if args.file:
        texts.extend(load_universe_file(Path(args.file)))
    if args.pattern:
        compiled = [re.compile(p) for p in args.pattern]
        vocab = embedder.store.texts_for(embedder.model_name)
        hits = [t for t in vocab if any(rx.search(t) for rx in compiled)]
        print(f"[tokencollider] patterns matched {len(hits)}/{len(vocab)} cached phrases")
        texts.extend(hits)
    texts = sorted(set(texts))
    if not texts:
        raise SystemExit("[tokencollider] nothing to forget: pass phrases, --file, "
                         "and/or --pattern")
    if args.dry_run:
        for t in texts[:200]:
            print(f"  {t!r}")
        if len(texts) > 200:
            print(f"  ... and {len(texts) - 200} more")
        print(f"[tokencollider] dry run: {len(texts)} phrase(s) would be forgotten")
        return
    e, c = embedder.store.forget(embedder.model_name, texts)
    print(f"[tokencollider] forgot {len(texts)} phrase(s): {e} embedding rows, "
          f"{c} conditioning rows deleted")
    if args.vacuum:
        print("[tokencollider] vacuuming (reclaims space and scrubs deleted bytes) ...")
        embedder.store.vacuum()
        print("[tokencollider] vacuum done")
    elif e or c:
        print("[tokencollider] note: deleted bytes linger in the file until --vacuum")


def cmd_bridge(args, embedder) -> None:
    """Pre-write ai-toolkit's text-embedding cache for a training dataset:
    per image, concat(caption embedding, anchor tensors) at the layers the
    sampler reads, in the file layout the profile's `trainer_arch` uses.
    The caption embeddings have to match what ai-toolkit's own encoder pass
    would have produced, so the profile's dialect is checked against the
    arch's; --force overrides."""

    from .bridge import ARCHS, check_dialect, write_cache

    profile = args.profile_obj
    arch = profile.trainer_arch
    if arch not in ARCHS:
        raise SystemExit(
            f"[tokencollider] the {profile.name!r} profile names "
            f"{'no trainer_arch' if arch is None else f'trainer_arch {arch!r}'}"
            f", and the bridge writes caches for {', '.join(ARCHS)}.")
    off = check_dialect(arch, embedder)
    if off and not args.force:
        raise SystemExit(
            f"[tokencollider] {off}. Caption embeddings would not match ai-toolkit's own "
            "encoder pass (--force writes anyway).")
    written = write_cache(
        Path(args.dataset), embedder, [Path(a) for a in args.anchor], arch,
        trigger=args.trigger, caption_ext=args.caption_ext,
        default_caption=args.default_caption,
        jumpstart=args.jumpstart,
        alternates=getattr(args, "alternates", False),
    )
    print(f"[tokencollider] wrote {len(written)} cache files -> {written[0].parent}"
          + (" (+ .anchor sidecars)" if args.jumpstart else ""))
    print(f"[tokencollider] cache layout: arch {arch}"
          + (f", trigger word {args.trigger!r} baked into every caption"
             if args.trigger else ""))


def cmd_profiles(args, _embedder=None) -> None:
    """What profiles exist, where they came from, and whether they are usable."""
    path = profiles.profiles_path()
    registry = profiles.load()
    names = sorted(k for k in registry if not k.startswith("__"))
    print(f"[tokencollider] profiles file: {path}"
          f"{'' if path.exists() else '  (not present; built-ins only)'}")
    incomplete = False
    for name in names:
        prof = registry[name]
        model, model_src = profiles.resolve(prof, "MODEL", None, prof.model_path)
        config, _ = profiles.resolve(prof, "CONFIG_DIR", None, prof.config_dir)
        incomplete = incomplete or not model
        print(f"{' ' if model else '!'} {name:<16} {prof.origin}")
        print(f"    model    {model or '(unset)'}"
              f"{f'  ({model_src})' if model else ''}")
        print(f"    config   {config or prof.config_repo or '(unset)'}")
        print(f"    layer    {prof.layer}   bounds "
              f"{prof.layer_min}..{prof.layer_max}   sampler "
              f"{list(prof.sampler_layers) if prof.sampler_layers else 'one layer'}")
    if incomplete:
        print("\n! = no weights path; add a 'model:' line before using it")


def cmd_serve(args, embedder) -> None:
    from .layout import LayerStack
    from .server import serve

    stack = LayerStack(embedder)
    if args.universe:
        phrases, meta = load_universe(Path(args.universe), embedder)
        stack.add_landmarks(phrases, source="preload")
        for p in (meta or {}).get("manual", []):
            if p in stack.origin:  # settlements stay marked across reloads
                stack.origin[p] = "manual"
        stack.provenance = meta
        stack.source_path = str(Path(args.universe).resolve())
        if stack.has_world():
            print(f"[tokencollider] world axes available from {stack.world_path()}")
        print(f"[tokencollider] preloaded {len(stack.phrases)} landmarks from {args.universe}")
    serve(stack, port=args.port, export_root=args.export_dir,
          layer_bounds=getattr(args, "layer_bounds", (None, None)),
          godot=getattr(args, "godot", None), web=getattr(args, "web", False))


def images_max_edge() -> float:
    from .images import IMAGE_SIZE
    return IMAGE_SIZE


def describe(embedder, profile, sources: dict, bounds=(None, None)) -> None:
    """Say what is actually loaded. A wrong model or a leftover template makes
    plausible-looking output rather than an error, so the session states its
    configuration instead of leaving it to be inferred from the starfield."""
    sampler = embedder.sampler_layers()
    lo, hi = bounds
    print(f"[tokencollider] profile {profile.name}  ({profile.origin})")
    print(f"[tokencollider]   model     {embedder.model_name}  ({sources['model']})")
    # The resolved source, not the setting: a self-contained HF model
    # directory supplies its own config, and saying otherwise would send
    # someone hunting for a config dir that is not being used.
    source = embedder.config_source()
    origin = sources["config"]
    if embedder.config_dir is None and source == embedder.model_name:
        origin = "alongside the weights"
    print(f"[tokencollider]   config    {source}  ({origin})")
    print(f"[tokencollider]   layer     {embedder.layer}  ({sources['layer']})   "
          f"bounds {lo if lo is not None else '-'}..{hi if hi is not None else '-'}"
          f"  ({sources['bounds']})   pooling {embedder.pooling}")
    print(f"[tokencollider]   sampler   "
          f"{list(sampler) if sampler else 'resolved on first pass'}"
          f"  ({sources['sampler']})")
    print(f"[tokencollider]   trim      {embedder.trim_template_prefix} "
          f"(scaffold tokens dropped from exports)")
    print(f"[tokencollider]   template  {embedder.template!r}  ({sources['template']})")
    print(f"[tokencollider]   images    pooled over the {embedder.image_pooling} tokens; "
          f"centre-cropped to {int(images_max_edge())} px square")


def add_common_options(parser, suppress: bool = False) -> None:
    """--profile, --layer and --db, the only model options on the command line.

    Everything else a profile resolves (weights, config dir, template, sampler
    layers, pooling, device) comes from profiles.yaml or the profile's own
    TOKENCOLLIDER_<PROFILE>_<SETTING> variable. Added twice: to the main parser with real
    defaults, and to every subcommand with SUPPRESS defaults so the options
    may come before or after the command without the subcommand's default
    overwriting the earlier value.
    """
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("-p", "--profile", default=default,
                        help="model profile, built in or from profiles.yaml "
                             "(default: $TOKENCOLLIDER_PROFILE, then the file's default)")
    parser.add_argument("--layer", default=default,
                        help="charting layer: 'last', an index, or a band "
                             "'lo-hi' (default: the profile's)")
    parser.add_argument("--db", type=Path,
                        default=default if suppress else DEFAULT_DB,
                        help=f"embedding cache (default: {DEFAULT_DB})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokencollider", description="Explore a text encoder's embedding space. "
                               "With no command, opens the viewport.")
    add_common_options(parser)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    common = argparse.ArgumentParser(add_help=False)
    add_common_options(common, suppress=True)

    def command(name, help):
        return sub.add_parser(name, parents=[common], help=help)

    for name, help in (("view", "sidecar and viewport together"),
                       ("serve", "the HTTP sidecar alone")):
        p = command(name, help)
        p.add_argument("universe", nargs="?", help="phrase file to preload")
        p.add_argument("--fake", action="store_true",
                       help="deterministic fake embeddings, no GPU")
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--export-dir", type=Path, default=None,
                       help="the only directory exports may be written to "
                            "(default: exports/)")
        if name == "view":
            p.add_argument("--desktop", action="store_true",
                           help="open the desktop Godot viewport instead of "
                                "the browser one")

    analysis = []
    p = command("warm", "embed a universe into the cache")
    p.add_argument("universe")
    p.add_argument("--light", action="store_true",
                   help="store only the charting-layer pooled vectors")
    analysis.append(p)

    p = command("rank", "rank universe members against a query")
    p.add_argument("universe")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=None, help="show only the top N")
    analysis.append(p)

    p = command("compare", "is QUERY closer to A than to B?")
    p.add_argument("universe")
    p.add_argument("query")
    p.add_argument("a")
    p.add_argument("b")
    analysis.append(p)

    p = command("axes", "label universe axes with wordlist entries")
    p.add_argument("universe")
    p.add_argument("wordlist")
    p.add_argument("--axes", type=int, default=6, help="axes to label")
    p.add_argument("--top", type=int, default=8, help="labels per pole")
    p.add_argument("--min-cos", type=float, default=0.2,
                   help="alignment floor for a label (0 disables)")
    p.add_argument("--light", action="store_true",
                   help="warm wordlist misses light")
    analysis.append(p)

    for p in analysis:
        p.add_argument("--variance", type=float, default=0.90,
                       help="variance the PCA basis must explain")

    p = command("forget", "delete cached rows for phrases")
    p.add_argument("texts", nargs="*", help="exact phrases")
    p.add_argument("--file", default=None, help="also every phrase in this file")
    p.add_argument("--pattern", action="append", default=None,
                   help="regex swept over every cached phrase (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be deleted, delete nothing")
    p.add_argument("--vacuum", action="store_true",
                   help="reclaim space and scrub the deleted bytes")
    p.add_argument("--model", action="store_true",
                   help="every cached row for this profile's model (after swapping "
                        "its weights)")

    p = command("bridge", "pre-write ai-toolkit's text-embedding cache")
    p.add_argument("dataset", help="image folder with caption files")
    p.add_argument("--anchor", action="append", required=True,
                   help="export to concat after every caption (repeatable)")
    p.add_argument("--trigger", default=None, help="trigger word")
    p.add_argument("--caption-ext", default=".txt")
    p.add_argument("--default-caption", default="",
                   help="caption for images with no caption file")
    p.add_argument("--jumpstart", action="store_true",
                   help="caption alone at the cache path, anchored version "
                        "as a .anchor sidecar")
    p.add_argument("--alternates", action="store_true",
                   help="with --jumpstart, one teacher sidecar per --anchor")
    p.add_argument("--force", action="store_true",
                   help="skip the dialect check")

    command("profiles", "list profiles and what they resolve to")
    return parser


def make_embedder(args) -> Embedder:
    """Resolve the profile's settings, refuse an unfinished one, and say what
    was loaded."""
    profile = profiles.get(args.profile)
    resolved = {}

    def setting(name, flag=None, default=None):
        value, resolved[name] = profiles.resolve(profile, name, flag, default)
        return value

    model = profiles.require(profile, "MODEL",
                             setting("MODEL", None, profile.model_path),
                             "weights path")
    config_dir = setting("CONFIG_DIR", None, profile.config_dir)
    layer = setting("LAYER", args.layer, profile.layer)
    template = setting("TEMPLATE", None, profile.template)
    sampler_layers = profiles.parse_sampler_layers(
        setting("SAMPLER_LAYERS", None, profile.sampler_layers))
    pooling = setting("POOLING", None, "mean")
    image_pooling = setting("IMAGE_POOLING", None, "image")
    device = setting("DEVICE", None, None)
    for name, value, choices in (("POOLING", pooling, ("mean", "last")),
                                 ("IMAGE_POOLING", image_pooling,
                                  ("image", "tail"))):
        if value not in choices:
            raise SystemExit(f"[tokencollider] {profile.env_prefix}_{name}={value!r}: "
                             f"expected one of {', '.join(choices)}")
    layer_min, min_src = profiles.resolve_int(profile, "LAYER_MIN",
                                              profile.layer_min)
    layer_max, max_src = profiles.resolve_int(profile, "LAYER_MAX",
                                              profile.layer_max)
    args.layer_bounds = (layer_min, layer_max)
    args.profile_obj = profile
    # A full HF model directory already carries its own config and tokenizer
    # beside the weights, so demanding a separate one would be busywork.
    self_contained = (Path(model).is_dir()
                      and (Path(model) / "config.json").exists())
    if config_dir is None and profile.config_repo is None and not self_contained:
        raise SystemExit(
            f"[tokencollider] the {profile.name!r} profile has no config source: its "
            "config and tokenizer are not usually in the HF cache, and "
            f"{model} does not carry its own. Set a 'config_dir:' line in "
            f"{profiles.profiles_path()} or {profile.env_prefix}_CONFIG_DIR, "
            "pointing at a local directory with config.json, tokenizer.json, "
            "and tokenizer_config.json."
        )
    embedder = Embedder(
        EmbeddingStore(args.db), model_name=model, layer=layer,
        template=template, pooling=pooling, device=device,
        config_dir=config_dir, light=getattr(args, "light", False),
        sampler_layers=sampler_layers, config_repo=profile.config_repo,
        trim_template_prefix=profile.trim_template_prefix,
        image_pooling=image_pooling,
    )
    # Not for `forget --model`, which is how a swapped model's cache is cleared.
    if not (args.command == "forget" and getattr(args, "model", False)):
        embedder.check_weights()
    describe(embedder, profile, {
        "model": resolved["MODEL"], "config": resolved["CONFIG_DIR"],
        "layer": resolved["LAYER"], "template": resolved["TEMPLATE"],
        "sampler": resolved["SAMPLER_LAYERS"],
        "bounds": min_src if min_src == max_src else f"{min_src}/{max_src}",
    }, args.layer_bounds)
    return embedder


COMMANDS = {"warm": cmd_warm, "rank": cmd_rank, "compare": cmd_compare,
            "axes": cmd_axes, "serve": cmd_serve, "view": cmd_serve,
            "forget": cmd_forget, "bridge": cmd_bridge}


def parse_args(argv: list[str]) -> argparse.Namespace:
    # No command means `view`, with every option it was given: `tokencollider --fake`
    # and `tokencollider -p krea2 universes/materials.txt` both open the viewport.
    if not {"-h", "--help", "profiles", *COMMANDS} & set(argv):
        argv = ["view", *argv]
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.command == "profiles":
        cmd_profiles(args)
        return
    if args.command == "view":
        # Before any model load, so a missing viewport fails in a second
        # rather than after a universe preload. The browser build when it is
        # installed, desktop Godot when asked for or when it is not.
        from .server import find_godot, web_build_available
        args.web = not args.desktop and web_build_available()
        if not args.web:
            args.godot = find_godot()
    if args.command in ("view", "serve") and args.fake:
        cmd_serve(args, FakeEmbedder())
        return
    embedder = make_embedder(args)
    try:
        COMMANDS[args.command](args, embedder)
    finally:
        embedder.store.close()


if __name__ == "__main__":
    main()
