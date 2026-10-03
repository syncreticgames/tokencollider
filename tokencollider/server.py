"""Localhost HTTP sidecar for the Godot viewport.

Every request needs the session token in an X-TokenCollider-Token header and a
Host of 127.0.0.1:<port> or localhost:<port>; see docs/security.md. The web
build's own files (GET / and the files beside index.html) need only the Host.

Contract (all JSON):
  GET  /health                      -> {"ok": true, "n_landmarks": N}
  GET  /layout[?layer=N]            -> full layout (see layout.py), plus
                                       "layer" (resolved layer index) and
                                       "n_layers" (hidden-state count, null
                                       until the model has run once)
  GET  /trajectory                  -> {"n_layers": N, "layers": [int],
                                       "extent": float,
                                       "trajectories": {text: [{"coords":
                                       [6 floats], "color": "#hex"} per layer]}}
                                       — each landmark's constellation path;
                                       colors are that layer's own color axes,
                                       extent the max |spatial coord| anywhere
                                       (for a layer-stable world scale).
                                       Trails stop at the deepest layer the
                                       sampler reads, so the post-norm final
                                       state is charted only when it is
                                       actually the sampler layer; "n_layers"
                                       is the point count, "layers" their
                                       hidden-state indices

Layer values (query or body) may be a single layer ("4"), or an inclusive
range averaged in pooled space ("0-6" in a query, [0, 6] in a JSON body).
An "axes" value (query or body) of "world" reads and writes coordinates in
the parent atlas's chart instead of this stack's own; layouts report which
chart they are in ("axes"), whether a world exists ("has_world"), and the
neighborhood's centre in that chart ("center": coords + color, the origin
in local axes). A body with "center": true names the landmark mean itself
as the target, whatever "coords" says: a chart's six numbers only reach its
own subspace, and the mean of this neighborhood lies in no other chart's.
  POST /landmarks    {"text": str, "layer": int?}  -> layout after adding
  POST /landmarks/remove {"text": str, "layer": int?} -> layout after removing
  POST /interrogate  {"coords": [6 floats], "k": int?, "layer": int?} -> nearest
                                       landmarks, plus "nearest_vocab" — the
                                       nearest phrases in the whole cached
                                       vocabulary under this view's config
  POST /blend        {"coords": [6 floats], "layer": int?} -> least-squares landmark weights
  POST /export       {"coords": [6 floats], "path": str?, "layer": int?,
                      "inserter": int?, "mirror": bool?}
                      "layer" picks the chart, "inserter" sets where the blend
                      is placed; without "inserter" the two move together.
                      "mirror" also writes the centroid reflection as the
                      negative half of a polarity pair
  POST /export_universe {"name": str?}  -> writes the loaded landmark set to
                                       universes/<name|session_TIMESTAMP>.txt
                                       with a #tokencollider provenance header (see
                                       tokencollider.provenance)
  POST /colonize     {"coords": [6 floats], "k": int?, "name": str?,
                      "layer": int?, "load": bool?} -> claim the neighborhood
                                       around the cursor: k nearest cached
                                       vocab phrases + all manual landmarks
                                       become a child universe whose header
                                       records the parent fingerprint and
                                       chart transition; load=true (default)
                                       swaps the live stack to it and returns
                                       the new "layout"

Layout payloads carry "layer_min"/"layer_max" (the profile's slider bounds;
null = unbounded), "sampler_layers" (every layer an export carries) and
"cook_stop" (the deepest of them: the layer a cooked export runs to), and a
per-landmark "source" ("manual" | "preload") so the viewport can render your
own additions apart from the surveyed background.

Layer semantics: omitted/null layer = the embedder's configured charting
layer. A numeric layer or range views/solves in that layer's chart. Export
renders a plain blend of sampler-layer tensors when the view's top edge is the
cook stop; for any view topping out below it, `hi` is the inserter: assembles
the blend from layer-hi hidden states and runs the remaining blocks forward
(easing the placement out over two blocks), so the result is always a valid
sampler conditioning. Sampler layers at or below the inserter stay a direct
blend. The band's low edge shapes the chart blend weights solve in; its
high edge sets where cooking begins. Under the krea2 profile the chart (20)
sits below the cook stop (35), so every viewport export cooks.
                     -> writes the blended per-token conditioning tensor at that
                        point to a safetensors file (key "conditioning", shape
                        (1, seq, dim), weights in metadata). This file is the
                        tool's take-home artifact; load it anywhere a raw
                        conditioning is accepted.

Coordinates are layout-space: dims 1-3 position, dims 4-6 color axes
(z-score to color via tokencollider.oklab; clients may send hex through hex_to_zscores
client-side or just send raw coords). Stdlib-only by design.
"""

import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import images, paths, provenance
from .layout import LayerStack, LayoutSession

EXPORT_DIR = paths.home() / "exports"
UNIVERSE_DIR = paths.home() / "universes"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
# The Godot web export. CI builds it into the wheel; a checkout gets it from
# tools/export_web.sh. Absent, `view` falls back to desktop Godot.
WEB_DIR = Path(__file__).resolve().parent / "web"
TOKEN_HEADER = "X-TokenCollider-Token"
# The web export's file types. .wasm must be application/wasm or the browser
# refuses to stream-compile it.
WEB_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
             ".wasm": "application/wasm", ".pck": "application/octet-stream",
             ".png": "image/png"}


def _universe_extra(stack: LayerStack, extra: dict | None = None) -> dict:
    """Header fields beyond the base config: which phrases are the user's own
    additions, and (for neighborhoods) the parent chart this one zooms into."""
    manual = [p for p in stack.phrases if stack.origin.get(p) == "manual"]
    out = {"manual": manual} if manual else {}
    if extra:
        out.update(extra)
    return out


def safe_export_path(path: str, root: Path | None = None) -> str:
    """Confine a caller-supplied export path to the exports directory.

    `/export` takes a filename from the request body and writes it. Unchecked,
    that is an arbitrary file write reachable by anything that can open a
    socket to the loopback port, which on a desktop is every process on the
    box. Absolute paths, `..`, and symlinks that escape are all refused.
    """
    root = (root or EXPORT_DIR).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("export path must be relative to the exports directory")
    dest = (root / candidate).resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"export path escapes {root}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest)


def export_universe(stack: LayerStack, name: str | None,
                    extra: dict | None = None, prefix: str = "session",
                    view: dict | None = None) -> dict:
    """Snapshot the loaded landmark set as a universe file (with a `#tokencollider`
    provenance header), loadable later by `tokencollider view` or any of the CLI
    commands."""
    phrases = list(stack.phrases)
    if not phrases:
        raise ValueError("no landmarks to export")
    if name is None:
        name = time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
    stem = Path(name).stem  # strip any directory part and extension
    if not stem:
        raise ValueError(f"unusable universe name: {name!r}")
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    path = UNIVERSE_DIR / f"{stem}.txt"
    fields = _universe_extra(stack, extra)
    if view:
        # Presentation state (camera, sliders, cursor...) rides in the header
        # but stays OUT of the fingerprint — build_meta hashes only geometry.
        fields["view"] = view
    meta = provenance.build_meta(stack.embedder, phrases, extra=fields)
    provenance.write_universe(path, phrases, meta,
                              getattr(stack.embedder, "store", None))
    return {"path": str(path), "n_phrases": len(phrases),
            "fingerprint": meta["fingerprint"]}


def _slug(text: str, limit: int = 24) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:limit].strip("-") or "x"


def parse_axes(val) -> str:
    """"world" projects through the parent atlas's chart; anything else is
    the stack's own chart."""
    return "world" if str(val or "local").lower() == "world" else "local"


def _center_target(stack: LayerStack, layer, center) -> "np.ndarray | None":
    """The exact landmark mean when the cursor is the neighborhood centre;
    None otherwise, so coords are lifted through the chart as usual."""
    return stack.session(layer).mean if center else None


def _nearest_names(session: LayoutSession, coords, basis=None, target=None) -> str:
    """The two nearest landmarks to the point — the human-scannable half of
    an export name."""
    import numpy as np
    if target is None:
        target = (basis or session).reconstruct(coords)
    v = session.vectors
    cos = (v @ target) / (np.linalg.norm(v, axis=1) * np.linalg.norm(target) + 1e-12)
    order = np.argsort(cos)[::-1]
    store = getattr(session.embedder, "store", None)
    names = [_slug(images.label(store, session.phrases[i])
                   if images.is_image_key(session.phrases[i])
                   else session.phrases[i]) for i in order[:2]]
    while len(names) < 2:
        names.append("x")
    return f"{names[0]}_{names[1]}"


def export_conditioning(stack: LayerStack, layer, coords, path: str | None,
                        fmt: str = "safetensors", band=None,
                        suffix: str = "", directory: Path | None = None,
                        mirror: bool = False,
                        axes: str = "local", center: bool = False) -> dict:
    session = stack.session(layer)
    basis = stack.basis(layer, axes)
    target = _center_target(stack, layer, center)
    cond, blend = session.export_conditioning(coords, band=band, basis=basis,
                                              target=target)
    if path is None:
        # Name = the two roles the band plays, spelled apart: mapLO_HI is the
        # chart the blend weights were solved in; cookNN / layerNN is the
        # single layer the tensor was actually assembled at.
        dest = directory if directory is not None else EXPORT_DIR
        dest.mkdir(parents=True, exist_ok=True)
        names = _nearest_names(session, coords, basis, target)
        key = stack.layer_key(layer).replace("-", "_")
        if blend.get("cooked"):
            stem = f"{names}_map{key}_cook{blend['band'][1]:02d}"
        elif "_" in key:  # band view, classic blend: assembled at the top edge
            stem = f"{names}_map{key}_layer{key.split('_')[-1]}{suffix}"
        else:
            stem = f"{names}_layer{key}{suffix}"
        if mirror:  # in every branch, so a negative never passes for a positive
            stem += "_mirror"
        out = dest / f"{stem}.{fmt}"
        if out.exists():  # same neighborhood, different point: disambiguate
            blob = b"".join(cond[l].tobytes() for l in cond.layers)
            digest = hashlib.sha256(blob).hexdigest()[:6]
            out = dest / f"{stem}_{digest}.{fmt}"
        path = str(out)
    if fmt == "cond":
        # ComfyUI conditioning-list pickle, as loaded by e.g.
        # comfyui-conditioning-saver's LoadConditioning node. No room for
        # metadata — the file must be exactly the conditioning structure, so
        # a multi-layer stack has nowhere to record which layer is which.
        import torch

        if len(cond) != 1:
            raise ValueError(
                f"the .cond format holds one tensor, but this conditioning "
                f"spans depths {list(cond.layers)} — export it as safetensors"
            )
        torch.save([[torch.from_numpy(cond.tensor)[None, :, :], {}]], path)
    elif fmt == "safetensors":
        # Full provenance rides in the file: the exact frame this point was
        # selected in (config + live universe fingerprint) plus the loaded
        # universe file it descends from, so any render traces back to the
        # distribution that defined its axes.
        config = provenance.embedder_config(stack.embedder)
        metadata = {
            "weights": json.dumps(blend["weights"]),
            "coords": json.dumps([float(c) for c in coords]),
            "relative_error": str(blend["relative_error"]),
            "cooked": str(blend["cooked"]).lower(),
            "band": json.dumps(blend.get("band")),
            # How many scaffold tokens were dropped so this lands in the
            # sampler's frame. 0 means the whole templated sequence is here.
            "template_prefix_tokens": str(blend.get("template_prefix_tokens", 0)),
            **config,
            "view_layer": stack.layer_key(layer),
            # Which chart the coords were read in. "world" means the parent
            # atlas's axes, whose fingerprint then rides along too.
            "axes": "world" if basis is not None else "local",
            # True when the target was the landmark mean itself rather than
            # the lift of `coords`.
            "center": str(bool(center)).lower(),
            # Modalities in the recipe and, for a mixed frame, its layout
            # (prefix / image / phrase / tail token counts).
            "groups": json.dumps(blend.get("groups")),
            "group_weights": json.dumps(blend.get("group_weights")),
            "frame": json.dumps(blend.get("frame")),
            "universe": provenance.fingerprint(phrases=stack.phrases, **config),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if basis is not None and stack.world is not None:
            metadata["world_universe"] = provenance.fingerprint(
                phrases=stack.world.phrases, **config)
        if stack.provenance and "fingerprint" in stack.provenance:
            metadata["origin_universe"] = stack.provenance["fingerprint"]
        # One layer writes the historical single "conditioning" key; a stack
        # writes one "layer_NN" key per layer (see tokencollider.conditioning).
        cond.save(path, metadata)
    else:
        raise ValueError(f"unknown export format: {fmt}")
    # Weight matters and must be visible (a lone hub word can eat a whole
    # blend): sorted by |weight| for the viewport's status line, full table
    # printed by the handler.
    top_weights = sorted(blend["weights"].items(), key=lambda kv: -abs(kv[1]))
    return {
        "path": path,
        "shape": [1, cond.seq_len, cond.dim],
        "layers": list(cond.layers),
        "weights": blend["weights"],
        "top_weights": top_weights,
        "relative_error": blend["relative_error"],
        "cooked": blend["cooked"],
        "band": blend.get("band"),
    }


def colonize(stack: LayerStack, layer, coords, name: str | None,
             k: int = 32, load: bool = True, axes: str = "local",
             center: bool = False) -> dict:
    """Claim the neighborhood around a cursor: the k nearest cached-vocabulary
    phrases plus every manual landmark become a child universe whose header
    records the parent's fingerprint and the chart transition (which view, and
    where in it, the zoom happened). With load=True the stack swaps to the new
    neighborhood in place — no restart — and keeps the chart it was carved
    from as its world, so the beachhead coordinate still means the same
    point afterwards."""
    session = stack.session(layer)
    basis = stack.basis(layer, axes)
    target = _center_target(stack, layer, center)
    if target is None:
        target = (basis or session).reconstruct(coords)
    texts, mat = stack.vocab(layer)
    if mat is None:
        viewed = stack.layer_key(layer)
        warmed = stack.layer_key(None)
        hint = (f"scrub the sliders to {warmed} (the slice `warm` writes)"
                if viewed != warmed else
                "warm a wordlist under this profile first (`tokencollider warm FILE --light`)")
        raise ValueError(
            f"no cached vocabulary under layer key {viewed!r}: {hint}, or warm "
            f"with --layer {viewed} to give this view its own vocabulary"
        )
    import numpy as np
    cos = (mat @ target) / (np.linalg.norm(mat, axis=1)
                            * (np.linalg.norm(target) + 1e-12) + 1e-12)
    order = np.argsort(cos)[::-1]
    manual = [p for p in stack.phrases if stack.origin.get(p) == "manual"]
    settled = set(manual)
    natives = []
    for i in order:
        if texts[i] not in settled:
            settled.add(texts[i])
            natives.append(texts[i])
        if len(natives) >= k:
            break
    phrases = natives + manual
    parent_config = provenance.embedder_config(stack.embedder)
    extra = {
        "parent": {
            "fingerprint": provenance.fingerprint(phrases=stack.phrases,
                                                  **parent_config),
            "coords": [float(c) for c in coords],
            "layer": stack.layer_key(layer),
            "axes": "world" if basis is not None else "local",
        },
    }
    if stack.source_path:
        extra["parent"]["path"] = stack.source_path
    if stack.provenance and "fingerprint" in stack.provenance:
        extra["parent"]["origin_universe"] = stack.provenance["fingerprint"]
    # The world is the atlas at the top of the chain, which is the parent
    # itself on a first zoom. Its file is how a later session, possibly
    # under another model, rebuilds the same axes.
    world = stack.world if stack.world is not None else stack
    extra["world"] = {
        "fingerprint": provenance.fingerprint(phrases=world.phrases,
                                              **parent_config),
    }
    if world.source_path:
        extra["world"]["path"] = world.source_path
    origin = {p: "preload" for p in natives} | {p: "manual" for p in manual}
    result = None
    if load:
        stack.replace_landmarks(phrases, origin)
        result = export_universe(stack, name, extra=extra, prefix="neighborhood")
        stack.provenance = {"fingerprint": result["fingerprint"],
                            "path": result["path"], "parent": extra["parent"],
                            "world": extra["world"]}
        stack.source_path = result["path"]
    else:
        # Write the file without touching the live stack: build the header
        # from a shallow stand-in carrying the would-be neighborhood.
        meta = provenance.build_meta(
            stack.embedder, phrases,
            extra=({"manual": manual} if manual else {}) | extra)
        UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(name or time.strftime("neighborhood_%Y%m%d_%H%M%S")).stem
        path = UNIVERSE_DIR / f"{stem}.txt"
        provenance.write_universe(path, phrases, meta,
                                  getattr(stack.embedder, "store", None))
        result = {"path": str(path), "n_phrases": len(phrases),
                  "fingerprint": meta["fingerprint"]}
    result["loaded"] = load
    result["natives"] = natives
    result["manual"] = manual
    return result


def parse_layer(val):
    """None | int | "4" | "0-6" | [0, 6] -> None | int | (lo, hi)."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        lo, hi = int(val[0]), int(val[1])
    elif isinstance(val, str) and "-" in val:
        lo, hi = (int(part) for part in val.split("-", 1))
    else:
        return int(val)
    if lo > hi:
        lo, hi = hi, lo
    return lo if lo == hi else (lo, hi)


def web_build_available() -> bool:
    return (WEB_DIR / "index.html").is_file()


def make_handler(stack: LayerStack, layer_bounds: tuple[int | None, int | None] = (None, None),
                 export_root: Path | None = None, *, token: str,
                 web_dir: Path | None = None):
    # Where exports may be written. The OPERATOR picks this when launching
    # (--export-dir); a REQUEST only ever names a path relative to it. That is
    # the whole security boundary: the person who started the process is
    # trusted, whatever opened a socket to the loopback port is not.
    root = Path(export_root).resolve() if export_root else EXPORT_DIR
    lock = threading.RLock()
    if not token:
        raise ValueError("the sidecar needs a session token")
    # Static files are served by exact name from a listing taken now, so a
    # request path never touches the filesystem.
    web_dir = WEB_DIR if web_dir is None else web_dir
    web_files = {f.name: f for f in web_dir.iterdir()
                 if f.is_file() and f.suffix in WEB_TYPES} if web_dir.is_dir() else {}

    def layout_payload(layer, axes: str = "local") -> dict:
        session = stack.session(layer)
        basis = stack.basis(layer, axes)
        payload = session.layout(basis)
        # Which chart the coords are in, whether a world exists to switch
        # to, and where this neighborhood's centre sits in that chart.
        payload["axes"] = "world" if basis is not None else "local"
        payload["has_world"] = stack.has_world()
        payload["world_path"] = stack.world_path()
        payload["center"] = session.centroid(basis)
        payload["layer"] = stack.resolve_layer(layer)
        payload["n_layers"] = stack.layer_count()
        # The layer a cooked export runs to: the deepest layer the sampler
        # reads. A view whose top edge sits below it cooks on export. For
        # Z-Image that is the charted layer itself (35); for Krea 2 it is 35
        # while the chart sits at 20, so the viewport needs it sent apart.
        sampler = stack.primary.sampler_layers()
        payload["sampler_layers"] = list(sampler) if sampler else None
        payload["cook_stop"] = max(sampler) if sampler else None
        payload["layer_min"], payload["layer_max"] = layer_bounds
        payload["view"] = (stack.provenance or {}).get("view")
        digits = stack.density_digits(layer)
        store = getattr(stack.embedder, "store", None)
        for entry in payload["landmarks"]:
            entry["source"] = stack.origin.get(entry["text"], "manual")
            entry["density"] = digits.get(entry["text"]) if digits else None
            if images.is_image_key(entry["text"]):
                # The viewport draws the picture and prints the file's name;
                # the key stays the identity everything else uses.
                entry["kind"] = "image"
                entry["label"] = images.label(store, entry["text"])
                entry["image"] = f"/image/{images.key_sha(entry['text'])}"
            else:
                entry["kind"] = "phrase"
        return payload

    def add_landmark_texts(texts: list[str], source: str = "manual") -> list[str]:
        """Typed entries: a leading "@" (or an image path) means a picture,
        a directory means every picture in it."""
        store = getattr(stack.embedder, "store", None)
        pooling = getattr(stack.embedder, "image_pooling", "image")
        keys = []
        for t in texts:
            t = str(t)
            raw = t[1:].strip() if t.startswith("@") else t
            if t.startswith("@") or (images.is_image_path(raw) and Path(raw).expanduser().exists()) \
                    or Path(raw).expanduser().is_dir():
                keys.extend(images.register_many(store, [raw], pooling))
            else:
                keys.append(t)
        stack.add_landmarks(keys, source=source)
        return keys

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _guard(self, need_token: bool = True) -> None:
            """Refuse anything not from this sidecar's own page or client.

            Host: a DNS-rebinding page (evil.example resolved to 127.0.0.1)
            is same-origin in the browser's eyes, but its Host header still
            names its own domain. Origin: a browser marks every cross-origin
            request with it; the only page allowed is the one served here.
            Token: random per session, handed to the client out of band
            (env for desktop Godot, URL fragment for the web build), so
            neither a web page nor another local process can drive the API
            without it."""
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                raise PermissionError("unexpected Host header")
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{h}" for h in hosts}:
                raise PermissionError("cross-origin requests are refused")
            if need_token and not hmac.compare_digest(
                    (self.headers.get(TOKEN_HEADER) or "").encode(), token.encode()):
                raise PermissionError("missing or wrong session token")

        def _body(self) -> dict:
            # A browser on any page you visit can POST to a loopback port.
            # It cannot set Content-Type: application/json cross-origin
            # without a preflight, and this server answers no preflight and
            # sends no CORS headers, so requiring it blocks form-POST CSRF.
            # Both Godot clients already send it. The token and Origin checks
            # in _guard are the other two locks on the same door.
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if ctype.lower() != "application/json":
                raise PermissionError(
                    "POST requires Content-Type: application/json")
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else {}

        def _dispatch(self, handle) -> None:
            """Run one request under the server lock and answer every failure.

            One lock for the whole stack. ThreadingHTTPServer runs handlers
            in parallel, and the layout's arrays (phrases, vectors, the PCA
            basis) are updated in several steps, so a concurrent read saw
            them out of step; the embedder's lazy model load could also run
            twice on a cold start. There is one GPU, so serialising costs
            nothing real."""
            try:
                with lock:
                    handle()
            except PermissionError as e:
                # Refused on principle, not malformed: keep it out of the 400
                # bucket so a real client error stays distinguishable from a
                # request this server will never serve.
                self._send(403, {"error": str(e)})
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except SystemExit as e:
                # e.g. the free-VRAM gate while ComfyUI holds the GPU. A dead
                # handler thread would hang the viewport forever: answer.
                self._send(503, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            url = urlparse(self.path)
            name = "index.html" if url.path == "/" else url.path.lstrip("/")
            try:
                self._guard(need_token=name not in web_files and url.path != "/")
            except PermissionError as e:
                self._send(403, {"error": str(e)})
                return
            if name in web_files:
                # The page holds no secret; the token arrives in its URL
                # fragment, which the browser never sends to a server.
                self._send_bytes(200, web_files[name].read_bytes(),
                                 WEB_TYPES[web_files[name].suffix])
                return
            if url.path == "/":
                self._send(404, {"error": "no web build; run tools/export_web.sh"})
                return
            if url.path == "/health":
                # Outside the lock, so it answers during a long warm.
                self._send(200, {"ok": True, "n_landmarks": len(stack.phrases)})
                return
            self._dispatch(lambda: self._get(url))

        def do_POST(self):
            def handle():
                self._guard()
                self._post()
            self._dispatch(handle)

        def _get(self, url) -> None:
            if url.path == "/layout":
                qs = parse_qs(url.query)
                layer = parse_layer(qs["layer"][0]) if "layer" in qs else None
                axes = parse_axes(qs["axes"][0] if "axes" in qs else None)
                self._send(200, layout_payload(layer, axes))
            elif url.path == "/trajectory":
                self._send(200, stack.trajectories())
            elif url.path.startswith("/image/"):
                hit = images.lookup(getattr(stack.embedder, "store", None),
                                    url.path[len("/image/"):])
                if hit is None:
                    self._send(404, {"error": "unknown image"})
                else:
                    self._send_bytes(200, hit[2], "image/jpeg")
            else:
                self._send(404, {"error": f"unknown path {url.path}"})

        def _post(self) -> None:
            body = self._body()
            layer = parse_layer(body.get("layer"))
            axes = parse_axes(body.get("axes"))
            # The centre cursor names the landmark mean exactly, not its
            # six-number shadow in whichever chart is showing.
            center = bool(body.get("center", False))
            if self.path == "/landmarks":
                # "texts" adds a whole batch under ONE refit, matching
                # /landmarks/remove. Realizing an interrogation's ghosts
                # one request per phrase would pay the O(n^2 d) SVD a
                # dozen times over for a single gesture.
                texts = body.get("texts")
                if texts is None:
                    texts = [body["text"]] if "text" in body else []
                # "images": paths or directories, registered and embedded
                # through the vision tower.
                texts = list(texts) + ["@" + str(p) for p in body.get("images", [])]
                add_landmark_texts(texts)
                self._send(200, layout_payload(layer, axes))
            elif self.path == "/landmarks/remove":
                texts = body.get("texts")
                if texts is None:
                    texts = [body["text"]]
                removed = stack.remove_landmarks([str(t) for t in texts])
                if removed:
                    print(f"[tokencollider] removed {removed} landmark(s)")
                payload = layout_payload(layer, axes)
                payload["removed"] = removed
                self._send(200, payload)
            elif self.path == "/interrogate":
                result = stack.session(layer).interrogate(
                    body["coords"], int(body.get("k", 10)),
                    vocab=stack.vocab(layer), basis=stack.basis(layer, axes),
                    target=_center_target(stack, layer, center))
                near = ", ".join(
                    f"{e['text']} {e['cosine']:.3f}"
                    for e in result["nearest_landmarks"][:5])
                print(f"[tokencollider] interrogate: {near}")
                if result.get("nearest_vocab"):
                    vocab_near = ", ".join(
                        f"{e['text']} {e['cosine']:.3f}"
                        for e in result["nearest_vocab"][:5])
                    print(f"[tokencollider]      vocab: {vocab_near}")
                self._send(200, result)
            elif self.path == "/blend":
                basis = stack.basis(layer, axes)
                target = _center_target(stack, layer, center)
                result = stack.session(layer).blend_weights(
                    body["coords"], basis, target)
                result["void"] = stack.void_at(layer, body["coords"], basis, target)
                self._send(200, result)
            elif self.path == "/export":
                band = (layer, layer) if isinstance(layer, int) else layer
                if body.get("inserter") is not None:
                    # Chart and inserter are separate axes, and collapsing
                    # them is a measurement bug waiting to happen: moving
                    # `layer` alone moves BOTH, so a sweep built that way
                    # cannot attribute a change to either one. The Ctrl+X
                    # sweep path already keeps them apart; this exposes the
                    # same thing to a single export. `layer` still picks the
                    # chart, `inserter` sets where the blend is placed.
                    ins = int(body["inserter"])
                    lo = min(band) if isinstance(band, tuple) else ins
                    band = (min(lo, ins), ins)
                if body.get("stack"):
                    # Inserter stack: ONE cursor, ONE solved recipe,
                    # several inserters across the band. Deep
                    # cooks reinterpret, shallow cooks transcribe. Meant
                    # for timestep-scheduled conditioning downstream.
                    # "sweep": every layer 0..sampler — the synthetic walk
                    # that maps which inserter carries the concept.
                    import numpy as np
                    if body.get("sweep"):
                        sampler = stack.session(layer).sampler_layers()
                        top = max(sampler) if sampler else None
                        if top is None:
                            raise ValueError("model layer count unknown — "
                                             "add a landmark first")
                        depths = list(range(top - 1, -1, -1))
                        lo = 0
                        warm = getattr(stack.embedder,
                                       "warm_all_conditionings", None)
                        if warm:
                            print(f"[tokencollider] sweep: pre-warming conditionings "
                                  f"0..{top - 1} (one forward per phrase) ...")
                            warm(list(stack.phrases), list(range(0, top)))
                    else:
                        base = band if isinstance(band, tuple) else (
                            stack.primary_index()
                            if isinstance(stack.primary_index(), tuple)
                            else None)
                        if base is None:
                            # No band view = nowhere to vary the cook
                            # from; degrade to the single classic export.
                            result = export_conditioning(
                                stack, layer, body["coords"], None,
                                body.get("format", "safetensors"), band,
                                directory=root, axes=axes, center=center)
                            self._send(200, {"stack": [result],
                                             "depths": []})
                            return
                        lo, hi = int(min(base)), int(max(base))
                        count = max(2, int(body.get("stack_count", 4)))
                        depths = sorted(
                            {int(round(x)) for x in np.linspace(lo, hi, count)},
                            reverse=True)
                    # Sweeps are a diagnostic strip, not a performance:
                    # keep them out of the Load Conditioning Stack node's
                    # glob path (and out of stack filename collisions).
                    directory = root / "sweeps" if body.get("sweep") else root
                    print(f"[tokencollider] export stack: inserters {depths} ...")
                    results = [export_conditioning(
                        stack, layer, body["coords"], None,
                        body.get("format", "safetensors"), (min(lo, d), d),
                        suffix=f"_cook{d:02d}", directory=directory,
                        axes=axes, center=center)
                        for d in depths]
                    for r in results:
                        print(f"[tokencollider]   -> {r['path']}")
                    self._send(200, {"stack": results, "depths": depths})
                    return
                print("[tokencollider] export: blending"
                      + (" + cooking" if band is not None else "") + " ...")
                result = export_conditioning(
                    stack, layer, body["coords"],
                    safe_export_path(body["path"], root) if body.get("path") else None,
                    body.get("format", "safetensors"), band,
                    directory=root, axes=axes, center=center)
                if body.get("mirror"):
                    # Polarity guidance. The cursor reflected through the
                    # chart centroid (coords origin IS the universe mean,
                    # PCA being mean-centred) is the negative half of a
                    # pair. CFG's guidance vector is the difference between
                    # the two predictions, so with both endpoints on one
                    # axis it pushes along that axis alone and the guidance
                    # scale becomes a fader on it. Both halves must come
                    # from the same chart and the same inserter or the
                    # difference amplifies off-axis junk, which is why this
                    # reuses `layer`, `band`, `axes` and `center` verbatim.
                    mirrored = [-float(c) for c in body["coords"]]
                    # A pair has to stay together. When the caller named
                    # the positive half, the negative takes that name with
                    # a .mirror infix and lands beside it; only an unnamed
                    # export falls back to the auto-generated name, which
                    # would otherwise scatter the two halves across
                    # directories and leave nothing linking them.
                    mirror_path = None
                    if body.get("path"):
                        base = Path(safe_export_path(body["path"], root))
                        mirror_path = str(base.with_name(
                            f"{base.stem}.mirror{base.suffix}"))
                    result["mirror"] = export_conditioning(
                        stack, layer, mirrored, mirror_path,
                        body.get("format", "safetensors"), band,
                        mirror=True, directory=root, axes=axes,
                        center=center)
                    result["mirror"]["role"] = "negative"
                    print(f"[tokencollider] mirror -> {result['mirror']['path']}")
                print(f"[tokencollider] export -> {result['path']} "
                      f"(cooked={result['cooked']}, "
                      f"err={result['relative_error']})")
                print("[tokencollider] weights: " + "  ".join(
                    f"{p} {w:+.3f}" for p, w in result["top_weights"]))
                self._send(200, result)
            elif self.path == "/export_universe":
                self._send(200, export_universe(stack, body.get("name"),
                                                view=body.get("view")))
            elif self.path == "/gpu/release":
                release = getattr(stack.embedder, "release", None)
                result = release() if release else {"released": False}
                print(f"[tokencollider] gpu release: {result}")
                self._send(200, result)
            elif self.path == "/colonize":
                print("[tokencollider] colonize: gathering neighborhood ...")
                result = colonize(
                    stack, layer, body["coords"], body.get("name"),
                    int(body.get("k", 32)), bool(body.get("load", True)),
                    axes=axes, center=center)
                if result["loaded"]:
                    # Arrive in the world's axes: the same chart the
                    # cursor was dropped in, now with the natives
                    # around it.
                    result["layout"] = layout_payload(layer, "world")
                self._send(200, result)
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def log_message(self, fmt, *args):
            pass  # keep stdout clean; errors still traceback to stderr

    return Handler


def find_godot() -> str:
    """The Godot binary: $TOKENCOLLIDER_GODOT, else godot or godot4 on PATH."""
    godot = (os.environ.get("TOKENCOLLIDER_GODOT") or shutil.which("godot")
             or shutil.which("godot4"))
    if godot is None:
        raise SystemExit("[tokencollider] godot not found on PATH; set "
                         "TOKENCOLLIDER_GODOT, build the browser viewport with "
                         "tools/export_web.sh, or run the sidecar alone "
                         "with `tokencollider serve`")
    if not (FRONTEND_DIR / "project.godot").is_file():
        # A wheel install ships the web build, not the Godot project.
        raise SystemExit("[tokencollider] the desktop viewport needs a source "
                         "checkout; this install has only the browser one")
    return godot


def serve(stack: LayerStack, port: int = 8765, export_root: Path | None = None,
          layer_bounds: tuple[int | None, int | None] = (None, None),
          godot: str | None = None, web: bool = False) -> None:
    """Run the sidecar. With web=True, also open the web build in the default
    browser. Given a Godot binary instead, open the desktop viewport and stop
    the sidecar when it closes. The port is bound before either starts, so the
    viewport never races a sidecar that is still loading."""
    # A fixed TOKENCOLLIDER_TOKEN lets a hand-started Godot editor run talk to
    # a hand-started `serve`; otherwise every session gets a fresh one.
    token = os.environ.get("TOKENCOLLIDER_TOKEN") or secrets.token_urlsafe(32)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port),
                                     make_handler(stack, layer_bounds, export_root,
                                                  token=token))
    except OSError as e:
        raise SystemExit(f"[tokencollider] cannot listen on 127.0.0.1:{port}: {e.strerror}")
    port = server.server_address[1]
    root = Path(export_root).resolve() if export_root else EXPORT_DIR
    print(f"[tokencollider] sidecar listening on http://127.0.0.1:{port} (loopback only)")
    print(f"[tokencollider] exports confined to {root}")
    if godot is None and not web:
        print(f"[tokencollider] session token: {token} (send it as {TOKEN_HEADER})")
    try:
        if web:
            url = f"http://127.0.0.1:{port}/#token={token}"
            print(f"[tokencollider] viewport: {url}")
            webbrowser.open(url)
            server.serve_forever()
        elif godot is None:
            server.serve_forever()
        else:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            subprocess.run([godot, "--path", str(FRONTEND_DIR)],
                           env={**os.environ, "TOKENCOLLIDER_PORT": str(port),
                                "TOKENCOLLIDER_TOKEN": token})
    except KeyboardInterrupt:
        print("\n[tokencollider] shutting down")
    finally:
        if godot is not None:
            server.shutdown()
        server.server_close()
