# ComfyUI-TokenCollider

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes that load
TokenCollider exports.

- **Load Conditioning (safetensors)** loads one export as `CONDITIONING`, and
  outputs its metadata as text. It re-runs when the file changes.
- **Load Conditioning Stack (inserters)** loads a set of exports made from one
  point at different inserter layers, as one conditioning. Each file is
  active for its own share of the denoising steps. Give it a glob such as
  `exports/<name>_cook*`, or a folder holding only that set. A folder with
  more than one set is refused, and the error lists them.
- **Conditioning Info** passes a conditioning through and reports its token
  count, non-zero tokens and width.

A single-layer export loads as it is. A multi-layer export, such as Krea 2's
twelve layers, is joined into one tensor the way ComfyUI's own Krea 2 encoder
does it. That's the only multi-layer arrangement these nodes support. The file
format is described in TokenCollider's
[docs/export-format.md](../docs/export-format.md).

The nodes need `safetensors` and `torch`, which ComfyUI already has.

## Install

ComfyUI-Manager can't install from a folder inside another repo, so install
them by hand, as in ComfyUI's guide to
[installing a custom node manually](https://docs.comfy.org/installation/install_custom_node):

1. Copy or symlink this `comfyui_node` folder into `ComfyUI/custom_nodes/`,
   for example as `ComfyUI/custom_nodes/comfyui-tokencollider`. A symlink
   stays up to date with the repo.
2. Restart ComfyUI. The nodes are in the `tokencollider` category.

## License

MIT, Copyright (c) 2026 Syncretic Games LLC. The nodes import no ComfyUI code
(see TokenCollider's [docs/credits.md](../docs/credits.md)).
