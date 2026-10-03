"""Universe provenance: fingerprints, file headers, and load-time verification.

A universe's geometry is fully determined by (model, layer, pooling, template,
phrases): same inputs, same PCA basis. The fingerprint is a content hash over
exactly those inputs, so it doubles as an identity check — a neighborhood
carries its parent universe's fingerprint, an export carries the fingerprint
of the landmark set it was blended in, and a loader can refuse a file whose
recorded config doesn't match the session's before anything renders in the
wrong frame.

File format stays a plain phrase-per-line text file. Metadata rides in a
single `#tokencollider {json}` comment line, which every existing reader
already skips. Files written before the rename carry `#cx {json}` instead,
and still load.
"""

import hashlib
import json
import time
from pathlib import Path

HEADER_PREFIX = "#tokencollider "
LEGACY_HEADER_PREFIX = "#cx "

# The config fields that change embedding geometry. A mismatch in any of them
# means coordinates from the file's frame are meaningless in the session's.
CONFIG_FIELDS = ("model", "layer", "pooling", "template")


def fingerprint(model: str, layer: str, pooling: str, template: str,
                phrases: list[str]) -> str:
    """Content hash of everything that determines a universe's geometry.
    Phrases are sorted: membership defines the distribution, order doesn't."""
    key = json.dumps(
        {"model": model, "layer": str(layer), "pooling": pooling,
         "template": template, "phrases": sorted(phrases)},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def embedder_config(embedder) -> dict:
    """The geometry-determining config of a live embedder (Fake has no real
    model identity; it reports enough for round-trips in --fake sessions)."""
    return {
        "model": getattr(embedder, "model_name", "fake"),
        "layer": str(getattr(embedder, "layer", "last")),
        "pooling": getattr(embedder, "pooling", "mean"),
        "template": getattr(embedder, "template", "{}"),
    }


def build_meta(embedder, phrases: list[str], extra: dict | None = None) -> dict:
    config = embedder_config(embedder)
    meta = {
        "fingerprint": fingerprint(phrases=phrases, **config),
        **config,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    return meta


def write_universe(path: Path, phrases: list[str], meta: dict, store=None) -> None:
    """Phrases one per line; image landmarks as `@path` lines (see
    tokencollider.images), resolved back through the store."""
    from . import images

    header = HEADER_PREFIX + json.dumps(meta, ensure_ascii=False)
    lines = images.to_entries(phrases, store)
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def read_universe(path: Path) -> tuple[list[str], dict | None]:
    """Entries plus the `#tokencollider` metadata header, if the file has one.
    Headerless (legacy / hand-written) files load with meta=None. Entries
    are phrases, or `@path` image lines that
    tokencollider.images.resolve_entries turns into landmark keys."""
    phrases: list[str] = []
    meta: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        prefix = next((p for p in (HEADER_PREFIX, LEGACY_HEADER_PREFIX)
                       if line.startswith(p)), None)
        if prefix and meta is None:
            meta = json.loads(line[len(prefix):])
        elif line and not line.startswith("#"):
            phrases.append(line)
    return phrases, meta


def verify_universe(path: Path, phrases: list[str], meta: dict | None,
                    embedder) -> dict | None:
    """Refuse to load a universe into a session whose geometry differs.

    Config mismatch on layer, pooling, or template is fatal: the file's
    coordinates belong to a different frame and everything downstream would
    render plausible-looking nonsense. A mismatch on the model alone loads
    the phrases with the frame-dependent fields stripped, since comparing two
    models on one neighborhood needs exactly that. A phrase-set drift
    (hand-edited file) only warns — curating wordlists in a text editor is a
    supported workflow; the fingerprint refreshes on the next save.

    Returns the metadata to use, possibly stripped.
    """
    if meta is None:
        return None
    config = embedder_config(embedder)
    mismatched = [
        f"  {field}: file={meta[field]!r}  session={config[field]!r}"
        for field in CONFIG_FIELDS
        if field in meta and str(meta[field]) != str(config[field])
    ]
    if mismatched and all(m.strip().startswith("model:") for m in mismatched):
        # Same layer, pooling, and template, different weights: the phrase
        # list is reusable, the frame is not. This is how the same
        # neighborhood is viewed under a finetune and its base, so load the
        # phrases and drop everything that was a coordinate in the old frame
        # (the saved view and cursor, the parent's beachhead coords).
        print(f"[tokencollider] note: {path.name} was built under another model:\n"
              + "\n".join(mismatched)
              + "\n      loading its phrases for re-embedding; the saved view "
              "and cursor do not carry across")
        meta = {k: v for k, v in meta.items() if k != "view"}
        if isinstance(meta.get("parent"), dict):
            meta["parent"] = {k: v for k, v in meta["parent"].items()
                              if k != "coords"}
        return meta
    if mismatched:
        raise SystemExit(
            f"[tokencollider] {path} was built under a different embedding config:\n"
            + "\n".join(mismatched)
            + "\nIts coordinates are meaningless in this session's frame. "
            "Match the session config (flags / TOKENCOLLIDER_* env) to the file, or "
            "re-warm the universe under the new config and re-save it."
        )
    if "fingerprint" in meta:
        live = fingerprint(phrases=phrases, **{f: str(meta.get(f, config[f]))
                                               for f in CONFIG_FIELDS})
        if live != meta["fingerprint"]:
            print(f"[tokencollider] note: {path.name} phrases changed since its header "
                  "was written — fingerprint refreshes on next save")
    return meta
