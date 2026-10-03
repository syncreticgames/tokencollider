# Design

How TokenCollider is put together, and what's planned.

[← README](../README.md)

## Two halves

- **The Python side** (`tokencollider/`) runs the encoder, keeps a SQLite
  cache of embeddings, and serves an HTTP API on 127.0.0.1.
- **The viewport** (`frontend/`) is a Godot 4 project. It runs in the browser
  as a web export the Python side serves, or as a desktop app.

They talk over that local API only. The viewport's code is described in
[frontend/ARCHITECTURE.md](../frontend/ARCHITECTURE.md), and what the server
accepts in [security.md](security.md).

## Decisions

- **One forward pass fills both caches.** Pooled vectors (one per phrase)
  place phrases in the layout. Full per-token tensors are what exports are
  made of. Every layer is kept, since keeping them costs little more than
  pooling them.
- **The 3D view is a projection.** Selecting and querying always happen in the
  full embedding space. The six layout axes are the universe's top principal
  components (`explained` in `GET /layout` gives each one's share of the
  variance). What's left over is each point's `residual`.
- **A conditioning is a map from layer to tensor.** A model that reads one
  layer is the one-entry case of the same code (`tokencollider/conditioning.py`).
- **Model knowledge is data.** Templates and sampler layers live in
  `builtin_profiles.yaml` and `profiles.yaml`. The code names no model, and
  there's no default model.
- **Mismatched files are refused.** A saved universe or an export made under a
  different model setup (template, layers, pooling) won't load into a session
  where it would give wrong positions.
- **Private data stays private.** The cache, exports and saved universes hold
  every phrase in plain text. They live in the user data folder when
  installed, and are gitignored in a checkout.

## Integrations

These are written for particular tools:

- The ComfyUI nodes combine a multi-layer export the way ComfyUI's Krea 2
  encoder does.
- The ai-toolkit bridge and trainer write ai-toolkit's `zimage` and `krea2`
  cache formats.
- `tools/pack_encoder.py` packs Qwen3-VL-4B for ComfyUI.

## Planned

- **Multi-layer editing in the viewport.** Set a Krea 2 export's position
  separately at each layer it carries.
- **Hiding and filtering landmarks**, to thin out dense universes.
- **`tokencollider diff`.** Compare two related models phrase by phrase: how
  far each phrase moves, which neighbors change, and at which layers.
- **Exporting all vectors** for outside analysis. Until then, the cache is a
  plain SQLite file.
- **Smaller cache for Krea 2.** 34 of a Krea 2 conditioning's 42 tokens are the
  same for every phrase at a given layer. Storing them once would make the
  cache about five times smaller.
