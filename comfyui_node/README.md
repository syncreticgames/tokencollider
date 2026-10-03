# ComfyUI-TokenCollider

Three [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for TokenCollider conditioning exports.

- **Load Conditioning (safetensors)**: an export's path in, a standard
  `CONDITIONING` out, plus its metadata as a string. Re-runs when the file's
  content changes, so a re-export to a fixed path retriggers downstream.
- **Load Conditioning Stack (cook depths)**: one stack export (one cursor at
  several inserters), loaded as one conditioning. Give a glob such as
  `exports/<name>_cook*`, or a directory holding only that stack; a directory
  holding several stacks is refused with their names listed. Each
  entry carries its own start and end percent, so each is active over its own
  slice of the denoising schedule.
- **Conditioning Info**: pass-through inspector. Reports token count,
  effective (non-zero) tokens, and dim.

An export is per-token hidden states saved as safetensors, one tensor per
layer the diffusion model reads. A single-layer export loads as is. A
multi-layer stack is fused layer-major into the feature axis, `(1, seq,
n*dim)`, which is how ComfyUI's own Krea 2 encoder hands its twelve layers to
the diffusion model. That is the only multi-layer layout these nodes know; a
model that combines layers differently needs its own fusion.

Beyond the standard library it needs `safetensors` and `torch`, both of which
ComfyUI already provides.

## Licence

MIT, Copyright (c) 2026 Syncretic Games LLC. This package imports no ComfyUI
code, so it is not a derivative work of ComfyUI (GPL-3.0).
