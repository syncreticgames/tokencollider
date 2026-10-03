# Conditioning export format

The file TokenCollider writes when it exports a point: per-token hidden states
from the text encoder, laid out the way the diffusion model's sampler reads
them. Programs that read or write these files should follow this page.

[← README](../README.md)

**Format version: 1.**

## The file

A [safetensors](https://github.com/huggingface/safetensors) file. Two shapes,
told apart by their keys:

- **Single layer.** One key, `conditioning`, shape `(1, seq, dim)`. Written
  when the profile's sampler reads one hidden state (Z-Image reads index 35).
- **Multi-layer.** One key per hidden state the sampler reads, named
  `layer_NN`: the hidden-state index, zero-padded to two digits (`layer_02`,
  `layer_35`). Each is `(1, seq, dim)`, and every layer has the same `seq` and
  `dim`. The metadata field `layers` lists the indices as a JSON array in
  ascending order.

Shared by both:

- **dtype:** float32.
- **Hidden-state index:** index 0 is the embedding table's output, index `i`
  the output of decoder block `i`. That is the numbering of transformers'
  `output_hidden_states`, so a 36-block encoder has indices 0 to 36.
- **Key order inside the file is not meaningful.** Readers sort `layer_NN` by
  index.

### Krea 2

The built-in `krea2` profile writes multi-layer files with exactly these
twelve indices, all twelve every time:

`2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35`

They are the profile's `sampler_layers` (`tokencollider/builtin_profiles.yaml`),
so a profile that changes that list changes which keys its files carry. A
reader for a particular model checks the `layers` field against what that
model reads, and refuses a file that doesn't match.

## The token frame

`seq` is the templated sequence the sampler is handed, which depends on the
profile:

- **Profiles that trim** (`trim_template_prefix: true`, such as `krea2`): the
  scaffold before the prompt is dropped. The file starts at the phrase's own
  tokens and keeps the template's tail after them.
- **Profiles that don't** (such as `zimage`): the whole templated sequence,
  prefix, phrase and tail.

The metadata field `template_prefix_tokens` says how many tokens were dropped
(`0` when nothing was).

### Pairing with a caption

The training cache (`tokencollider bridge`) pairs an export with a caption by
concatenating along the token axis, layer by layer:

1. Embed the caption under the same profile, as a conditioning at the same
   layers, and trim it the same way (`template_prefix_tokens`).
2. Append the export's tokens after the caption's: caption first, export
   after. Layer `i` of the result is the caption's layer `i` followed by the
   export's layer `i`.

The export takes the caption's layer indices, since both come from the same
profile. Pairing a caption with an export from a different profile gives a
tensor no sampler could produce.

## Metadata

Safetensors metadata is a map of strings; values marked JSON are JSON-encoded
strings.

These fields decide whether the tensors fit a given model:

| Field | Value |
|---|---|
| `model` | The encoder's weights path or name |
| `layer` | The profile's charting layer (`"20"`, `"18-34"`, `"last"`) |
| `pooling` | `mean`, or another pooling mode |
| `template` | The prompt template, with `{}` where the phrase goes |
| `template_prefix_tokens` | Tokens dropped from the front (see above) |
| `layers` | JSON array of hidden-state indices; multi-layer files only |
| `sampler_layer` | The hidden-state index of a single-layer file's tensor; single-layer files only, and absent from files written before 10/03/2026 |

These record how the point was made, and aren't needed to load the file:

| Field | Value |
|---|---|
| `weights` | JSON map of landmark to blend weight |
| `coords` | JSON array, the six chart coordinates |
| `relative_error` | How far the blend misses the target, as a fraction |
| `cooked` | `true` when later encoder layers were run on the blend |
| `band` | JSON `[lo, hi]` of the layer band, or `null` |
| `view_layer` | The chart the blend was solved in |
| `axes` | `local` or `world` |
| `center` | `true` when the target was the landmark mean |
| `groups`, `group_weights`, `frame` | JSON: modalities in the blend and, for a text-and-image blend, its token layout |
| `universe` | Fingerprint of the landmark set |
| `world_universe`, `origin_universe` | Fingerprints of the parent atlas and loaded file, when there are any |
| `created` | Local timestamp |

Readers must ignore fields they don't know.

### Which fields a reader checks

A reader that knows which model it feeds must check the file against it
before using the tensors:

- **Refuse** a file whose `template`, layers or `template_prefix_tokens`
  differ from the reader's. Its tokens sit in positions the sampler never
  produces. The layers are the `layer_NN` keys, or for a single-layer file the
  index from `sampler_layer` (see below for older files).
- **Warn** when `model` differs, and load the file. The field is a weights
  path, which changes between machines and between packagings of one model,
  and comparing a finetune with its base is a supported use. TokenCollider
  treats universe files made under another model the same way.
- **Warn** when a file is too old to carry `template` or `layer`, and load it.

`check_frame` in `tokencollider/conditioning.py` implements this, and
`tokencollider bridge` runs it on every anchor. A reader that can't know its
model, such as the ComfyUI node, which isn't told which encoder feeds the
workflow, loads the file as its keys say.

## Reading files already on disk

The canonical reader is `Conditioning.load` in `tokencollider/conditioning.py`.
Any reader must accept what it accepts:

- A file with `conditioning` is single-layer, whatever else it holds. Its
  layer index comes from `sampler_layer`. Older files lack it, so then from
  `layer` (the top of a band: `"18-34"` means 34; `layer` is the profile's
  charting setting, which is the sampler's layer for every built-in
  single-layer profile), then `view_layer`. When none names a number it's 0,
  which for the oldest files is only a label.
- Otherwise, every `layer_NN` key is a layer. A file with neither is not a
  conditioning.
- Single-layer files have no `layers` field. Older files may have no metadata
  at all.

The `.cond` export option writes something else: a pickled ComfyUI
conditioning list for one layer, with no metadata. This page doesn't cover
it. Loading a pickle runs code, so don't pass those files between programs.

## Versioning

- **Version 2** is any change a version 1 reader would misread: key names,
  tensor shape or dtype, the hidden-state numbering, the token frame, or the
  meaning of a frame field.
- **Adding a metadata field doesn't change the version.** Readers already
  ignore fields they don't know.
- Version 1 files carry no version field. Version 2 files will carry a
  `format_version` field, so a file without one is version 1.
