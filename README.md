# TokenCollider

A viewer for the embedding space of a language-model text encoder: the
decoder-only LLM that a growing number of diffusion models read their
conditioning from, often at several hidden layers at once. Phrases become
**landmarks** in a 3D viewport in the browser, grouped into **universes**, and the
encoder's layers can be scrubbed to see how meaning gets built. A point in the
viewport can be read back as words (the nearest cached phrases) or written out
as a conditioning tensor in the diffusion model's own frame.

No model is wired into the core. A diffusion model's conventions are a
profile, which is data: the prompt template the phrase is wrapped in, the
hidden layers the sampler reads, and whether the sampler drops the template's
leading tokens. What the core does assume is a kind of model:

- **A decoder-only language model** that transformers can build from the
  checkpoint's own config (Qwen, Llama, Mistral and similar). Encoders like
  T5 or CLIP are a different architecture and are not supported.
- **The usual decoder layout** (`layers`, `rotary_emb`, `norm`) for exports
  that run a blend forward through the encoder's later layers. A model laid
  out differently still embeds and exports, and refuses only that step.
- **Qwen-VL style image tokens** for image landmarks. Text needs none.
- **Llama-style key names** in a single-file checkpoint, the ComfyUI layout.
  A Hugging Face model directory loads whatever its config names.

The free-VRAM check is sized from the checkpoint's own weights.

Two profiles are built in, in `tokencollider/builtin_profiles.yaml`, and they are the
two the tool has been run against:

- `zimage`: Qwen3-4B, the text encoder for [Z-Image](https://github.com/Tongyi-MAI/Z-Image). One layer.
- `krea2`: Qwen3-VL-4B, the text encoder for [Krea 2](https://www.krea.ai). Twelve layers, and images as
  well as text.

A *universe* is a list of phrases that defines a local metric. Its embeddings
are centred on their mean, which removes whatever all members share. The top
principal components are the directions the universe actually varies along,
and similarity is measured inside that subspace. So a question like "is Link
closer to Mario than to Donkey Kong?" has two answers:

- **raw**: cosine similarity in the full embedding space.
- **relative**: cosine similarity in the universe's own coordinates.

Where the two disagree, the universe is changing the answer.

## Install and run

TokenCollider needs Python 3.13 or newer and PyTorch. On Linux, PyTorch from
PyPI already includes CUDA. On Windows, PyPI's PyTorch runs on the CPU only, so
install the CUDA build first with the command from
[PyTorch's install selector](https://pytorch.org/get-started/locally/).

From a release, install the wheel attached to it. It includes the browser
viewport:

```
pip install tokencollider-<version>-py3-none-any.whl
tokencollider --fake
```

`--fake` uses deterministic fake embeddings, so it runs with no GPU and no
model. To use a real encoder, copy
[`profiles.example.yaml`](profiles.example.yaml) to `profiles.yaml` in
TokenCollider's home and point it at your model files.

TokenCollider's home holds the embedding cache, `profiles.yaml`, exports and
saved universes. The cache and exports contain every phrase you embed, in
plain text.

- Installed: `~/.local/share/tokencollider` on Linux,
  `%LOCALAPPDATA%\TokenCollider` on Windows, and
  `~/Library/Application Support/TokenCollider` on macOS.
- Source checkout: the repo itself. The cache, `profiles.yaml` and exports
  are gitignored, and saved universes go to `universes/saved/`, also
  gitignored, apart from the sample lists.
- `TOKENCOLLIDER_HOME` overrides both. `TOKENCOLLIDER_PROFILES` points at a
  `profiles.yaml` somewhere else.

`tokencollider view` (the default command) starts the sidecar and opens the
viewport in your browser. To add an image as a landmark, press Tab and type
`@` and the image or folder path, then Enter. Dropping files on the window
works only in the desktop viewport. `tokencollider view --desktop` opens the desktop
Godot viewport instead; it needs a source checkout and
[Godot 4](https://godotengine.org/download) on your PATH.

From a source checkout, `uv sync` installs the dependencies, and
`tools/export_web.sh` builds the browser viewport. That needs Godot 4.7 and its
web export templates. Without it, `view` falls back to desktop Godot.

## Layout

The core. Its code names no model; what it knows about one comes from a
profile:

- `tokencollider/`: the Python side. It owns the encoder, the SQLite embedding cache and a
  loopback-only HTTP sidecar, and provides the `tokencollider` command.
- `frontend/`: the Godot 4 viewport. It runs on the desktop, or in the
  browser as a web export the sidecar serves. Its structure is in
  [frontend/ARCHITECTURE.md](frontend/ARCHITECTURE.md).
- `tools/export_web.sh`: builds the web export. When a GitHub release is
  published, `.github/workflows/release.yml` runs it, builds the wheel with it
  inside, and attaches the wheel to the release.
- `universes/`: sample phrase lists.
- `tests/`: model-free test suites and a metric baseline, plus
  `tests/real_model.py`, which runs the two built-in profiles on real weights.

Integrations, each written for particular models or tools:

- `comfyui_node/`: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes
  that load exported conditionings. A single-layer export loads as is. A
  multi-layer stack is fused the way ComfyUI's Krea 2 encoder fuses it, the
  only multi-layer layout the node knows.
- `tokencollider bridge` (`tokencollider/bridge.py`) and `trainers/`: pre-write
  [ai-toolkit](https://github.com/ostris/ai-toolkit)'s text-embedding cache
  and distil an export into a LoRA. They write ai-toolkit's `zimage` and
  `krea2` cache formats, so they work for profiles that declare one of those
  as their `trainer_arch`.
- `tools/pack_encoder.py`: packs a sharded Qwen3-VL-4B into the single-file
  layout ComfyUI's Krea 2 loader reads.

Further reading:

- [docs/design.md](docs/design.md): design notes and how the two halves fit.
- [docs/vocabulary.md](docs/vocabulary.md): the project's terms.
- [docs/security.md](docs/security.md): what the sidecar exposes and refuses.
- [docs/export-format.md](docs/export-format.md): the export file format, for
  programs that read or write it.
- [docs/credits.md](docs/credits.md): licences and what each dependency asks.

## Built on

- **[Godot Engine](https://godotengine.org)** (MIT) is the viewport.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous
  (GPL-3.0) is the renderer the exports are built for.
- **[ai-toolkit](https://github.com/ostris/ai-toolkit)** by
  [Ostris](https://github.com/ostris) (MIT). The trainer extension is built on
  its caching and its `SDTrainer`.
- **[Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** and
  **[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)** by Alibaba's Qwen team
  are the encoders the built-in profiles load.
- **Krea 2** by Krea and **Z-Image** by Tongyi MAI are the diffusion models the
  built-in profiles target.
- **[PyTorch](https://pytorch.org)**,
  **[transformers](https://github.com/huggingface/transformers)**,
  **[safetensors](https://github.com/huggingface/safetensors)**,
  **[numpy](https://numpy.org)**, **[Pillow](https://python-pillow.org)** and
  **[PyYAML](https://pyyaml.org)**.

## Licence

MIT. See [LICENSE](LICENSE).
