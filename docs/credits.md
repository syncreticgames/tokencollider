# Licence and credits

TokenCollider is MIT licensed (see `LICENSE`). What follows is what the
project depends on and what each dependency asks of you. It is a practical
summary, not legal advice.

[← TokenCollider README](../README.md)

## The rule that decides everything

**Almost every obligation triggers on DISTRIBUTION, not on use.** Nothing here
is vendored into this repository: the engine, the Python packages, the model
weights, and ComfyUI are all things a user installs themselves. So the
source-only repository carries no obligation beyond its own MIT notice.

That changes the moment you ship a **binary**.

## If you ship an exported Godot binary

Godot is MIT licensed, and an export embeds the engine, so the engine's
copyright notice must travel with it. Godot makes this easy: it exposes the
full text at runtime through `Engine.get_license_text()` and
`Engine.get_copyright_info()`, and the recommended pattern is an in-app
credits screen that prints them. See Godot's own
[Complying with licenses](https://docs.godotengine.org/en/stable/tutorials/legal/complying_with_licenses.html).

The engine also bundles third-party components (FreeType, zlib and others)
with their own notices, which is exactly what `get_copyright_info()` returns.
Printing it wholesale is both the easiest and the most correct option.

## If you ship a bundled Python runtime

Freezing the sidecar (PyInstaller, Nuitka, or similar) redistributes every
dependency, and each one's notice then has to be included:

| Package | Licence |
|---|---|
| PyTorch | BSD 3-Clause |
| transformers | Apache 2.0 |
| safetensors | Apache 2.0 |
| numpy | BSD 3-Clause |
| Pillow | MIT-CMU |
| PyYAML | MIT |

Apache 2.0 additionally wants any `NOTICE` file carried along. `pip-licenses`
or `uv pip licenses` can generate the bundle for you; do not hand-maintain
this table for a shipped build, because it will drift.

## Model weights: not yours to ship, and you do not need to

TokenCollider reads whichever encoder a profile points at from local disk
(for the built-in profiles, Qwen3-VL-4B for Krea 2 or Qwen3-4B for Z-Image). It never copies or redistributes them, so their terms do not attach to
this project. Users obtain the weights themselves under whatever licence the
publisher offers.

If you ever ship a build that downloads weights automatically, that is
distribution and the model licence applies. Right now nothing is downloaded:
the loader is offline by construction.

Credit them anyway, in the README, because it is accurate and courteous:
the encoders are Alibaba's Qwen models, and Krea 2 and Z-Image are their
respective authors' work.

## ComfyUI is GPL-3.0, and this project stays clear of it

Worth stating precisely, because it is the one licence here that could reach
into your code.

`comfyui_node/nodes.py` imports only the Python standard library,
`safetensors` (Apache 2.0) and `torch` (BSD 3-Clause), which ComfyUI itself
provides. It touches no ComfyUI module, subclasses no ComfyUI class, and
copies no ComfyUI code. ComfyUI loads it and calls it, which is ordinary
plugin use, and the node is independently useful as a safetensors reader. On
that basis it is not a derivative work of ComfyUI and stays MIT.

If a future node imports from `comfy.*` or `nodes.py`, that reasoning no
longer holds and the question becomes genuinely contested. Keep the boundary
where it is.

## ai-toolkit is MIT

`trainers/tokencollider_jumpstart/` subclasses `SDTrainer` and imports from
`toolkit.*`, so it is plainly built on ai-toolkit. ai-toolkit is MIT
(Copyright (c) 2024 Ostris, LLC), which permits that and asks only that the
notice travel with any substantial portion. We ship none of their code, only
an extension that imports it, so there is nothing to carry; the credit below
is courtesy.

## Credits

- **[Godot Engine](https://godotengine.org)** (MIT). The viewport.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** (GPL-3.0). The
  renderer the exports are built for.
- **[ai-toolkit](https://github.com/ostris/ai-toolkit)** by Ostris (MIT). The
  LoRA trainer the bridge and jumpstart target.
- **Qwen3-VL-4B / Qwen3-4B** by Alibaba. The text encoders whose
  hidden states this tool explores.
- **Krea 2** and **Z-Image**. The diffusion models the exports condition.
- PyTorch, transformers, safetensors, numpy, Pillow, PyYAML.
