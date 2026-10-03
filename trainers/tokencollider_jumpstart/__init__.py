"""ai-toolkit extension: distil a TokenCollider export into a LoRA."""

from toolkit.extension import Extension


class JumpstartExtension(Extension):
    uid = "tokencollider_jumpstart"
    name = "TokenCollider Jumpstart"

    @classmethod
    def get_process(cls):
        from .JumpstartTrainer import JumpstartTrainer

        return JumpstartTrainer


AI_TOOLKIT_EXTENSIONS = [JumpstartExtension]
