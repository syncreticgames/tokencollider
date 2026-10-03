# Design

Design notes, what is planned, how the two halves fit.

[← TokenCollider README](../README.md)

## Design notes

- Standalone by design: Godot is the frontend, the sidecar the backend,
  nothing else in the loop. Two outputs only: **words** and **tensors**.
- Pooled vectors are the **navigation** representation; full per-token
  tensors are the **export** representation. One forward pass feeds both
  caches — at every layer (the marginal cost of keeping all layers is
  pooling, so they're never thrown away).
- The 3D GUI is a *view*; ground truth stays full-dimensional. Selection and
  interrogation always happen in the real space.
- Axis significance is ordered: PCA sorts axes by variance explained (per-axis
  numbers in `GET /layout`'s `explained`); everything past dim 6 lives in
  each point's `residual`.
- A conditioning is an ordered map from hidden-state index to a `(seq, dim)`
  array (`tokencollider/conditioning.py`). A single-layer model is the size-one case,
  not a separate code path, and its file format is unchanged.
- Per-model knowledge is data, not control flow (`tokencollider/builtin_profiles.yaml`,
  overlaid by `profiles.yaml`). No model name appears in the code's logic,
  and there is no default model. A finding about a model that turns out
  wrong is an edit to a data file.
- Anything whose geometry depends on model, layer, pooling, template, or
  phrase set refuses to load under a different config rather than render
  plausible nonsense.
- Privacy: the cache, exports and saved universes carry every embedded
  phrase in plaintext (and vectors decode back to text). Installed, they live
  in the user data folder; in a checkout, `.gitignore` covers them. Keep both
  true.

## Planned

- **Multi-layer authoring in the viewport**: layer stops being a view and
  becomes a coordinate, so a Krea 2 export can be pinned per layer rather than
  cooked from one chart.
- **Selection and filtering controls** in the viewport: fade/hide landmarks
  by selection actions (box-select and friends), so dense universes and
  constellations can be pared down to the phrases under investigation.
- **Finetune diffing** (`tokencollider diff`): per-phrase drift magnitudes between two
  same-family models, neighbor-rank shifts ("base puts 'trees' nearest
  'forest'; finetune says 'jungle'"), and drift-by-layer profiles to localize
  which layers a finetune rebinds.
- **Export all vectors**: bulk dump of the cached embeddings for external
  analysis (the DB is already queryable SQLite in the meantime).
- **Scaffold dedup in the cache**: 34 of a Krea 2 conditioning's 42 tokens
  are byte-identical across phrases at a given layer; storing them once per
  layer would cut cache size about fivefold.

## Core and integrations

- The core (`tokencollider/`, the viewport) names no model. What it knows about one
  comes from a profile, and what it assumes is a kind of model: a
  decoder-only language model transformers can build, the usual decoder
  layout for exports that run a blend forward, Qwen-VL style image tokens
  for image landmarks. The free-VRAM check is sized from the checkpoint.
- The integrations are written for particular targets and say so: the
  ComfyUI node fuses stacks the way Krea 2's encoder does, the ai-toolkit
  bridge and trainer write ai-toolkit's `zimage` and `krea2` cache formats,
  and `tools/pack_encoder.py` packs Qwen3-VL-4B for ComfyUI.

## Shape of the project

Two halves: a Python sidecar (the `tokencollider` package) that owns the encoder, the
cache and the HTTP API, and a Godot frontend (`frontend/`) that owns the
viewport. They talk over loopback only. Model files are read from wherever
`profiles.yaml` points; exports are written for ComfyUI on the same machine.
