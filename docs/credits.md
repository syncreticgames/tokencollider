# Licenses and credits

TokenCollider is MIT licensed (see [LICENSE](../LICENSE)). This page covers
what its dependencies ask of anyone who redistributes it. It's a practical
summary, not legal advice.

[← README](../README.md)

## The source repository

The repository includes none of its dependencies' code: Godot, the Python
packages, the model weights and ComfyUI are all installed separately. So the
source carries only its own MIT notice.

## The release wheel and Godot

The release wheel includes the browser viewport, which is a Godot web export,
so every release ships a compiled copy of Godot. Godot is MIT licensed, and
its notice has to go with it, along with the notices of the libraries it
bundles (FreeType, zlib and others).

The viewport's Credits screen (press M, then Credits) prints both, using
`Engine.get_license_text()` and `Engine.get_copyright_info()`. That's how
Godot's guide to
[complying with licenses](https://docs.godotengine.org/en/stable/tutorials/legal/complying_with_licenses.html)
recommends doing it. The Credits screen has to stay reachable for that reason.

## Bundling Python

A build that bundles the Python side (with PyInstaller, Nuitka or similar)
redistributes every package, and each one's notice has to be included:

| Package | License |
|---|---|
| PyTorch | BSD 3-Clause |
| transformers | Apache 2.0 |
| safetensors | Apache 2.0 |
| numpy | BSD 3-Clause |
| Pillow | MIT-CMU |
| PyYAML | MIT |

Apache 2.0 also asks for any `NOTICE` file to be included. Generate the
bundle with a tool such as `pip-licenses`, since this table will drift.

## Model weights

TokenCollider reads model weights from your disk. It never downloads or
redistributes them, so their licenses don't apply to the project. You get
the weights yourself, under their publishers' terms.

## ComfyUI

ComfyUI is GPL-3.0. The ComfyUI nodes (`comfyui_node/nodes.py`) import only
the Python standard library, `safetensors` (Apache 2.0) and `torch`
(BSD 3-Clause). They import, subclass and copy nothing from ComfyUI. ComfyUI
loads and calls them like any plugin, so they remain MIT. A node that
imported from ComfyUI would change that.

## ai-toolkit

The trainer in `trainers/tokencollider_jumpstart/` subclasses ai-toolkit's
`SDTrainer`. ai-toolkit is MIT (Copyright (c) 2024 Ostris, LLC). The project
includes none of its code, only an extension that imports it.

## Credits

- **[Godot Engine](https://godotengine.org)** (MIT): the viewport.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** (GPL-3.0): what
  the exports are made for.
- **[ai-toolkit](https://github.com/ostris/ai-toolkit)** by Ostris (MIT): the
  LoRA trainer the bridge and jumpstart trainer target.
- **Qwen3-VL-4B** and **Qwen3-4B** by Alibaba: the encoders the built-in
  profiles load.
- **Krea 2** and **Z-Image**: the diffusion models the exports are for.
- **[Oklab](https://bottosson.github.io/posts/oklab/)** by Björn Ottosson: the
  color space the viewport colors points in (`tokencollider/oklab.py`,
  `frontend/lib/Oklab.gd`).
- PyTorch, transformers, safetensors, numpy, Pillow and PyYAML.
