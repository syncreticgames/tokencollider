# TokenCollider

TokenCollider shows how a text encoder arranges phrases, and turns points in
that arrangement into conditioning files for diffusion models.

Some image models, such as [Krea 2](https://www.krea.ai) and
[Z-Image](https://github.com/Tongyi-MAI/Z-Image), read their prompt through a
language model. TokenCollider embeds phrases with that same encoder and lays
them out in 3D. You can see which phrases sit near each other, and how that
changes from one encoder layer to the next. Then you can pick any point and:

- read it back as the nearest phrases it knows, or
- export it as a conditioning file, which a diffusion model samples from like
  a prompt.

## Quick start

You need Python 3.13 or newer and PyTorch. On Windows, PyPI's PyTorch runs
on the CPU only, so install the CUDA build first with the command from
[PyTorch's install selector](https://pytorch.org/get-started/locally/). On
Linux, the PyPI build already includes CUDA.

Install the wheel attached to a [release](https://github.com/syncreticgames/tokencollider/releases),
then try it with fake embeddings (no GPU, no model):

```
pip install tokencollider-<version>-py3-none-any.whl
tokencollider --fake
```

This opens the viewport in your browser. To use a real encoder:

1. Run `tokencollider profiles --init`. It writes an example `profiles.yaml`
   into TokenCollider's home folder (see below).
2. Set the `model:` path for the profile you want.
3. Run `tokencollider -p krea2` (or `-p zimage`).

## Using it

- **Landmarks** are the phrases you add. Press Tab, type a phrase, press
  Enter. To add an image, type `@` and the image or folder path.
- **Universes** are lists of phrases, one per line, loaded at startup:
  `tokencollider -p krea2 phrases.txt`. A source checkout has a few samples in
  `universes/`. The layout is built from
  the universe: its embeddings are centered, and the axes are the directions
  its phrases vary along most.
- **Layers.** The sliders at the bottom choose which encoder layer, or range
  of layers, you're looking at.
- **The cursor.** Press C to drop it, I to list the phrases nearest to it, and
  X to export it. X writes a small set of files, each entering the encoder at
  a different layer; Shift+X writes one. The viewport's key help lists every key, and M opens the
  menu.

Other commands:

| Command | What it does |
|---|---|
| `view` | Start the server and open the viewport (the default) |
| `serve` | Start the server alone |
| `warm` | Embed a universe into the cache ahead of time |
| `rank` | Rank a universe's phrases by similarity to a query |
| `compare` | Say whether a phrase is closer to A or to B |
| `axes` | Label a universe's axes with words from a word list |
| `forget` | Delete phrases, or a whole model, from the cache |
| `bridge` | Write ai-toolkit's text-embedding cache for a dataset |
| `profiles` | List the profiles and what they resolve to |

`rank` and `compare` give two answers: similarity in the encoder's full space,
and similarity inside the universe's own axes. When they differ, the universe
is changing the answer.

## Models

Two profiles are built in (`tokencollider/builtin_profiles.yaml`):

- `zimage`: Qwen3-4B, Z-Image's encoder. Exports one layer.
- `krea2`: Qwen3-VL-4B, Krea 2's encoder. Exports twelve layers, and takes
  images as well as text.

A profile describes a model as data: its prompt template and the layers its
sampler reads. You can add your own in `profiles.yaml`. The encoder has to be
a decoder-only language model that transformers can load, such as Qwen,
Llama or Mistral. Image landmarks need a Qwen-VL style model. T5 and CLIP
encoders aren't supported.

## Where files go

TokenCollider's home holds the embedding cache, `profiles.yaml`, exports and
saved universes. The cache and exports contain every phrase you embed, in
plain text.

- Installed: `~/.local/share/tokencollider` on Linux,
  `%LOCALAPPDATA%\TokenCollider` on Windows, and
  `~/Library/Application Support/TokenCollider` on macOS.
- Source checkout: the repo itself. These files are gitignored, and saved
  universes go to `universes/saved/`.
- `TOKENCOLLIDER_HOME` sets another folder. `TOKENCOLLIDER_PROFILES` points at
  a `profiles.yaml` elsewhere.

## Working from source

`uv sync` installs the dependencies. `tools/export_web.sh` builds the browser
viewport, which needs Godot 4.7 and its web export templates. Without that
build, `tokencollider view` opens the desktop viewport instead, which needs
[Godot 4](https://godotengine.org/download) on your PATH. `view --desktop`
always does.

Run the tests with `uv run python tests/run.py`.

## Integrations

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes**
  (`comfyui_node/`) load exports as conditioning. Install steps are in
  [its README](comfyui_node/README.md).
- **[ai-toolkit](https://github.com/ostris/ai-toolkit) training.**
  `tokencollider bridge` writes ai-toolkit's text-embedding cache, and the
  trainer in `trainers/tokencollider_jumpstart` trains a LoRA towards an
  export. To install the trainer, copy or symlink that folder into
  ai-toolkit's `extensions/` folder, then start from
  `trainers/jumpstart.example.yaml`.
- `tools/pack_encoder.py` packs a sharded Qwen3-VL-4B into the single file
  ComfyUI's Krea 2 loader reads.

## Docs

- [docs/design.md](docs/design.md): how it's put together, and what's planned.
- [docs/vocabulary.md](docs/vocabulary.md): the terms the code and UI use.
- [docs/export-format.md](docs/export-format.md): the export file format.
- [docs/security.md](docs/security.md): what the local server accepts.
- [docs/credits.md](docs/credits.md): licenses and credits.
- [frontend/ARCHITECTURE.md](frontend/ARCHITECTURE.md): the viewport's code.

## Credits

Built on [Godot](https://godotengine.org), [PyTorch](https://pytorch.org),
[transformers](https://github.com/huggingface/transformers),
[safetensors](https://github.com/huggingface/safetensors),
[numpy](https://numpy.org), [Pillow](https://python-pillow.org) and
[PyYAML](https://pyyaml.org). The built-in profiles load Alibaba's
[Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), the encoders for Krea 2
(by Krea) and Z-Image (by Tongyi MAI). More in
[docs/credits.md](docs/credits.md).

## License

MIT. See [LICENSE](LICENSE).
