# Vocabulary

One word per thing, and the two axes everything else hangs off.

[← TokenCollider README](../README.md)

The rule throughout: technical terms name things, musical terms explain them. The metaphor never
becomes the label.

## The two axes

Almost every confusion in this project came from one word covering both of
these. They are independent, and they are set by different knobs.

- **Layer.** Where in the encoder's sequential passes something happens (36
  in both built-in encoders). The inserter lives here. Nothing on this axis
  is about time.
- **Window.** When, across a generation's denoising steps, a conditioning is
  active. The Load Conditioning Stack node's start and end percent live here.

An export is *placed* on the layer axis and *scheduled* on the window axis.
Never describe a layer with a time word.

## Names

These are the terms. They appear in code, filenames, and node widgets.

| Term | What it is |
|---|---|
| **layer** | one of the encoder's sequential passes, numbered from 0 |
| **universe** | the landmark set a chart is built from |
| **chart** | the local coordinate system the cursor moves in, built by pooling the universe over a layer band |
| **atlas** | the parent chart a neighborhood was carved from and aligns its axes to |
| **lift** | mapping a cursor's coordinates out of the chart and back into full embedding space |
| **blend** | the signed weights over universe members that a lifted point solves to |
| **inserter** | the layer the blend is placed at (`hi`, the band's high edge) |
| **export** | the file: every layer the sampler reads, written as one safetensors |
| **window** | the sigma range an export is active over, widened by **overlap** |
| **volume** | how hard an export pulls on the image (`repeat` on a model that normalizes its context, such as Z-Image) |
| **CFG** | how far past the charted reading the sampler pushes |
| **concat** | caption tokens and export tokens in one sequence, each keeping its own positions |
| **image embeddings** / **text embeddings** | the two kinds of landmark a blend mixes within but never across |

Coordinates on the layer axis stay numbers. Layer 20, inserter at 8, taps at
2 through 35. No word replaces the number.

## Explanations, using Krea 2

The explanations below use Krea 2's numbers as the worked example: 36 passes,
of which its diffusion model reads twelve. Another model reads a different
set; the profile says which.

These build intuition. They are never used as labels.

- **Rehearsal.** The encoder runs once, before anything renders. Thirty-six
  passes, each reworking what the last one produced.
- **Takes.** Krea 2's diffusion model (a DiT, diffusion transformer) does not
  read the final pass. It reads twelve of them, at layers 2, 5, 8 and on up
  to 35, and uses all twelve at once. One export is twelve takes of one
  passage. Z-Image reads a single take, which is where "one layer per file"
  came from.
- **Score.** What an export is to the sampler: written instructions, fixed
  before the performance starts.
- **Voice.** Each part in a concat. The caption and the export sit in one
  sequence and the DiT attends over both, but the caption was encoded without
  ever seeing the export. Two voices on the stand, neither having rehearsed
  with the other. Averaging them gives you neither, which is why
  `ConditioningAverage` produces grids.
- **The performance.** One generation, from cacophony toward coherence, one
  step per moment.

The inserter in these terms: it decides which rehearsal you walk into and hand
the players your part. Everything after it is rehearsed normally, so those
takes come out sounding like the encoder made them. Takes recorded *before*
the inserter cannot contain your part, so those are filled by hand from the
raw blend. Inserter at 8 leaves nine takes rehearsed and three filled by hand.
Inserter at 26 reverses that. Same blend, same twelve tensors, different file.

On CFG: the sampler computes what it would produce with the prompt and what
it would produce with no prompt at all, then pushes past the first in the
direction away from the second. CFG 1.0 plays the score as written. Above 1.0
overplays it, leaning into whatever makes this score different from the
model's default, which is why high CFG blows out contrast and goes rigid.

## Retired

Old words, and what they became.

| Retired | Now |
|---|---|
| cooking (as a verb), patch-and-cook | placing the blend at the inserter |
| injection depth, cook depth, band top | inserter |
| tap, depth (as a synonym for layer) | layer |
| takes (as a verdict: "the depth that takes") | the inserter that renders best |
| geometry (when it meant the chart), map, view layer, mode, palette | chart |
| attend, plumb, ingest | read |
| library | universe |
| sections | image embeddings, text embeddings |

`cookNN` and `mapLO_HI` stay in export filenames, and `blend`, `export`, and
`chart` stay as identifiers and endpoints. Renaming files or code was never
the point.

"Geometry" survives where it means the actual shape of a point cloud, as in a
universe's geometry or the vision tower's patch geometry. It is retired only
where it stood in for the chart.
