# Vocabulary

The terms the code, the viewport and the file names use.

[← README](../README.md)

| Term | Meaning |
|---|---|
| **layer** | One of the encoder's passes, numbered from 0. Both built-in encoders have 36. |
| **landmark** | A phrase or image placed in the viewport. |
| **universe** | A set of landmarks. The layout is built from it. |
| **chart** | The universe's own coordinates, built from its embeddings at one layer or an averaged range of layers. |
| **atlas** | The parent chart a smaller neighborhood was cut from. Its axes are kept aligned to the parent's. |
| **blend** | The weights over landmarks that a point in the chart works out to. |
| **inserter** | The layer an export's blend enters the encoder at (see below). |
| **export** | The file a point is saved as: every layer the model's sampler reads. See [export-format.md](export-format.md). |
| **window** | The part of the denoising schedule an export is active for, widened by **overlap**. |
| **concat** | A caption and an export joined into one sequence of tokens. |

Two settings are easy to mix up:

- **Layer** is about the encoder: which of its passes.
- **Window** is about generation: which denoising steps.

They're independent. An export is placed at a layer and scheduled over a
window.

## The inserter

Krea 2's encoder has 36 layers, and its diffusion model reads twelve of them:
layers 2, 5, 8 and so on up to 35. An export holds all twelve.

The inserter is the layer where the blend goes in. The encoder then runs the
remaining layers on it, so every layer after the inserter looks like the
encoder made it. Layers at or before the inserter come straight from the
blend.

So an inserter at 8 gives nine layers the encoder ran and three taken from the
blend. An inserter at 26 gives the reverse. Same blend, different file. An
earlier inserter (a lower layer number) leaves the encoder more of its own
passes to reinterpret the blend.

## Concat and CFG

In a concat, the caption and the export sit in one sequence, and the
diffusion model attends to both. They were encoded separately, though.
Averaging the two instead gives neither, and tends to produce grids.

CFG (classifier-free guidance) is how far past the prompt the sampler pushes,
away from what it would make with no prompt. At 1.0 it follows the
conditioning as given. Higher values exaggerate whatever sets it apart from
the model's default, which is why high CFG gives harsh contrast.

## In file names

Export files use older short forms: `cookNN` is the inserter layer, and
`mapLO_HI` is the chart's layer range.
