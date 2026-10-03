"""Image landmarks: identity, thumbnails, and the universe-file syntax.

An image is a landmark like a phrase. It embeds through the same encoder
(Qwen3-VL's vision tower feeds the text tower at `<|image_pad|>` positions
inside the same scaffold a phrase sits in), gets pooled vectors at every
layer, and lands in the chart next to words. The interesting question, which
layer brings images and words closest, is then a layer scrub.

Identity is content: the key is `image:<sha256 prefix>` (or `image/tail:`,
see below), so the cache never depends on where the file lives. The universe
file keeps a path, one `@path` per line, relative to the file or absolute;
the session resolves it to a key and remembers path, name, and a thumbnail
in the store, which is what the viewport draws.

Two poolings, chosen per session, each its own key (their geometry differs,
so they must not share cache rows):

- `image`: mean over the image tokens themselves.
- `tail`: mean over the tokens from `<|vision_end|>` to the end of the
  scaffold, which have attended to the image. A coordinate in text-token
  terms by construction.
"""

import hashlib
import io
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
POOLINGS = ("image", "tail")
PREFIX = "image"
THUMB_PX = 160
# Every picture is centre-cropped to a square and resized to this, so every
# picture is the same token grid (512 / 16 / 2 = 16 per side, 256 tokens)
# and a blend of pictures aligns patch by patch. Part of what an image's
# hidden states are, so it is a constant rather than a flag; change it and
# `tokencollider forget` the images.
IMAGE_SIZE = 512

# Fallback registry for sessions without a store (--fake).
_memory: dict[str, tuple[str, str, bytes]] = {}


def is_image_key(text: str) -> bool:
    return text.startswith(PREFIX + ":") or text.startswith(PREFIX + "/")


def key_mode(key: str) -> str:
    """'image' or 'tail', from the key."""
    head = key.split(":", 1)[0]
    return head.split("/", 1)[1] if "/" in head else "image"


def key_sha(key: str) -> str:
    return key.split(":", 1)[1]


def make_key(sha: str, pooling: str = "image") -> str:
    if pooling not in POOLINGS:
        raise ValueError(f"image pooling must be one of {POOLINGS}, not {pooling!r}")
    return f"{PREFIX}:{sha}" if pooling == "image" else f"{PREFIX}/{pooling}:{sha}"


def is_image_path(text: str) -> bool:
    return Path(text).suffix.lower() in IMAGE_EXTS


def image_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def prepare(path: Path, size: int = IMAGE_SIZE):
    """The picture as the vision tower sees it: upright, RGB, centre-cropped
    to a square, resized to `size`. Returns a PIL image."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        return im.resize((size, size), Image.LANCZOS)


def thumbnail(path: Path, px: int = THUMB_PX) -> bytes:
    """A small JPEG of the prepared picture, so the sprite shows the crop the
    model saw rather than the file's full frame."""
    im = prepare(path, px)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _put(store, sha: str, path: Path, name: str, thumb: bytes) -> None:
    if store is not None:
        store.put_image(sha, str(path), name, thumb)
    else:
        _memory[sha] = (str(path), name, thumb)


def lookup(store, sha: str) -> tuple[str, str, bytes] | None:
    """(path, name, thumbnail jpeg) for a registered image, or None."""
    if store is not None:
        hit = store.get_image(sha)
        if hit is not None:
            return hit
    return _memory.get(sha)


def register(store, path: Path, pooling: str = "image") -> str:
    """Register one image file and return its landmark key."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    if not is_image_path(path.name):
        raise ValueError(f"not an image by extension: {path.name}")
    sha = image_sha(path)
    hit = lookup(store, sha)
    # Same content under a new path (a moved or copied file): keep the key,
    # refresh the path when the recorded one no longer exists.
    if hit is None or not Path(hit[0]).exists():
        _put(store, sha, path, path.stem, thumbnail(path))
    return make_key(sha, pooling)


def register_many(store, paths, pooling: str = "image") -> list[str]:
    """Files and directories (non-recursive) to keys, in order, deduplicated."""
    keys, seen = [], set()
    for p in paths:
        p = Path(p).expanduser()
        files = (sorted(q for q in p.iterdir() if is_image_path(q.name))
                 if p.is_dir() else [p])
        for f in files:
            k = register(store, f, pooling)
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def label(store, key: str) -> str:
    """What the viewport prints under an image landmark: the file's stem."""
    hit = lookup(store, key_sha(key))
    return hit[1] if hit else key


def resolve_entries(entries: list[str], base_dir: Path, store,
                    pooling: str = "image") -> list[str]:
    """Universe-file lines to landmark keys: `@path` becomes an image key
    (paths relative to the file's directory), everything else is a phrase.
    Missing or unreadable image files are reported and skipped, not fatal: a
    universe still loads when a picture moved or is corrupt."""
    out = []
    for line in entries:
        if not line.startswith("@"):
            out.append(line)
            continue
        raw = line[1:].strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(base_dir) / path)
        try:
            out.append(register(store, path, pooling))
        except (OSError, ValueError) as e:  # PIL's decode errors are OSErrors
            print(f"[tokencollider] skipping image landmark: {e}")
    return out


def to_entries(keys: list[str], store) -> list[str]:
    """Landmark keys back to universe-file lines."""
    out = []
    for k in keys:
        if is_image_key(k):
            hit = lookup(store, key_sha(k))
            out.append("@" + (hit[0] if hit else k))
        else:
            out.append(k)
    return out
