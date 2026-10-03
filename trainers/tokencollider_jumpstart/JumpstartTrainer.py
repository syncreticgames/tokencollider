"""Distil a TokenCollider export into a LoRA, with no captions to learn.

The idea: an export is extra context tokens, and the DiT already renders the
concept when they are present. Nothing maps those tokens onto weights in
closed form (extra context changes every token's attention), but the
mapping can be learned from the model's own behaviour:

- teacher: the base DiT, conditioned on `caption + export` (the anchored
  embedding the bridge wrote as a sidecar);
- student: the base DiT plus the LoRA, conditioned on `caption` alone (the
  regular cache file);
- loss: the student's velocity matches the teacher's on noised latents of
  the dataset's own images, at random timesteps.

The LoRA that results behaves with the anchor folded in: the model does
with the LoRA what it did with the export in context. Resume ordinary image
training from it and the search phase is gone.

How it rides on ai-toolkit's own machinery (SDTrainer, read at main on
08222026):

- `do_prior_prediction = True` makes the train loop call
  `get_prior_prediction()` with the network switched off, before the
  student pass. That is the teacher pass; this class overrides the method
  to substitute the anchored embeddings for the batch.
- In the default loss branch, `calculate_loss()` takes `target = prior_pred`
  whenever a prior prediction is present and neither
  `diff_output_preservation` nor `blank_prompt_preservation` is on
  (SDTrainer: `elif prior_pred is not None and not do_prior_divergence`).
  So the distillation loss is the trainer's normal loss with the teacher
  as target; nothing else is patched.
- `correct_pred_norm`, `inverted_mask_prior`, and `do_prior_divergence`
  would route the target elsewhere, so they are refused.

Cache layout the bridge writes with `--jumpstart`:

    <dataset>/_t_e_cache/<stem>_<hash>.safetensors          caption only
    <dataset>/_t_e_cache/<stem>_<hash>.anchor.safetensors   caption + export
"""

import glob
import os
import random

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype

ANCHOR_SUFFIX = ".anchor.safetensors"


def anchor_path_for(text_embedding_path: str) -> str:
    """The anchored sidecar beside a regular cache file."""
    root, ext = os.path.splitext(text_embedding_path)
    return root + ANCHOR_SUFFIX


def alternate_paths_for(text_embedding_path: str) -> list[str]:
    """Numbered sidecars beside a cache file, written by `tokencollider bridge
    --jumpstart --alternates`. Empty when the bridge wrote a single fixed
    anchor, which is the default."""
    root, ext = os.path.splitext(text_embedding_path)
    # Escaped: an image named `photo [1].jpg` puts brackets in `root`, which a
    # glob reads as a character class, and then nothing matches.
    return sorted(glob.glob(glob.escape(root) + ".anchor.[0-9][0-9].safetensors"))


def trainer_base():
    """SDTrainer from the CLI; DiffusionTrainer (SDTrainer plus the status,
    loss, and stop/save/sample reporting the web UI reads from its database)
    when the UI launched us, which it signals with AITK_JOB_ID. The UI
    trainer overrides none of the methods the jumpstart touches."""
    if os.environ.get("AITK_JOB_ID"):
        try:
            from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

            return DiffusionTrainer
        except ImportError:
            pass
    return SDTrainer


class JumpstartTrainer(trainer_base()):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tc = self.train_config
        for flag in ("diff_output_preservation", "blank_prompt_preservation",
                     "correct_pred_norm", "inverted_mask_prior",
                     "do_prior_divergence"):
            if getattr(tc, flag, False):
                raise ValueError(
                    f"tokencollider_jumpstart: `{flag}` reroutes the loss target "
                    "away from the teacher prediction; leave it off.")
        # The train loop runs the network-off teacher pass whenever this is
        # set, and the default loss branch then targets it.
        self.do_prior_prediction = True
        self._anchor_cache: dict[str, PromptEmbeds] = {}
        self._alt_cache: dict[str, list[str]] = {}
        conf = self.get_conf("jumpstart", {}) or {}
        # >1 pushes the target PAST the anchored prediction, along the
        # direction away from the caption-only one. Same shape as CFG, and for
        # the same reason: the difference between the two predictions IS the
        # export's contribution, so scaling it sharpens what the LoRA is asked
        # to learn without adding steps. Costs one extra no-grad forward.
        self.anchor_guidance = float(conf.get("anchor_guidance", 1.0))
        if self.anchor_guidance < 1.0:
            raise ValueError("jumpstart.anchor_guidance below 1.0 would pull the "
                             "target back toward the caption-only prediction, "
                             "which is the thing the LoRA already does.")

    def hook_before_train_loop(self):
        # The network is built in run(), after __init__ and the model load,
        # so this is the first hook where it exists.
        super().hook_before_train_loop()
        if self.network is None:
            raise ValueError("tokencollider_jumpstart trains a LoRA; set `network`.")
        self.do_prior_prediction = True
        print("[jumpstart] loss target = base model conditioned on caption + export; "
              "student = LoRA conditioned on caption")
        if self.anchor_guidance != 1.0:
            print(f"[jumpstart] anchor guidance {self.anchor_guidance}: target is "
                  "extrapolated past the anchored prediction")

    def _anchored_embeds(self, batch, device, dtype) -> PromptEmbeds:
        """The batch's anchored embeddings, in batch order, from the sidecar
        files the bridge wrote beside each item's regular cache file.

        Keyed off the path the item ACTUALLY loaded, not the one its caption
        hashes to. Caption dropout does apply with a cached text encoder:
        `load_prompt_embedding` rolls per item and swaps in the blank
        embedding, recording the choice in `_loaded_text_embedding_path`.
        Reading `get_text_embedding_path()` instead would pair a blank student
        against the anchored teacher of the UNDROPPED caption on those steps,
        which trains the LoRA to render the caption's content from an empty
        prompt: the baked-look failure, taught deliberately."""
        items = []
        for file_item in batch.file_items:
            if not getattr(file_item, "is_text_embedding_cached", False):
                raise ValueError(
                    "tokencollider_jumpstart needs `cache_text_embeddings: true` "
                    "and a bridge-written cache (`tokencollider bridge --jumpstart`).")
            loaded = getattr(file_item, "_loaded_text_embedding_path", None)
            base = loaded or file_item.get_text_embedding_path()
            # Alternates, when the bridge wrote them: one anchor per step
            # instead of one anchor for the whole run. A LoRA distilled from a
            # single coordinate can only learn that coordinate; sampling from a
            # neighbourhood teaches the concept's local manifold.
            if base not in self._alt_cache:
                self._alt_cache[base] = alternate_paths_for(base)
            alts = self._alt_cache[base]
            path = random.choice(alts) if alts else anchor_path_for(base)
            if path not in self._anchor_cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"no anchored embedding beside the cache file: {path}. "
                        "The cache was not written by `tokencollider bridge --jumpstart`, "
                        "or predates its dropout pair.")
                self._anchor_cache[path] = PromptEmbeds.load(path)
            items.append(self._anchor_cache[path].clone().detach())
        return concat_prompt_embeds(items).to(device, dtype=dtype)

    def get_prior_prediction(self, noisy_latents, conditional_embeds, *args,
                             batch=None, **kwargs):
        # Same teacher pass SDTrainer makes (network off, no grad), on the
        # anchored conditioning instead of the student's. The call site
        # passes everything by keyword, so the rest rides through kwargs.
        teacher_embeds = self._anchored_embeds(
            batch, self.device_torch, get_torch_dtype(self.train_config.dtype))
        anchored = super().get_prior_prediction(
            noisy_latents, teacher_embeds, *args, batch=batch, **kwargs)
        if self.anchor_guidance == 1.0 or anchored is None:
            return anchored
        # The caption-only prediction from the SAME base model, so the
        # difference is the export's contribution and nothing else.
        plain = super().get_prior_prediction(
            noisy_latents, conditional_embeds, *args, batch=batch, **kwargs)
        if plain is None:
            return anchored
        return plain + self.anchor_guidance * (anchored - plain)
