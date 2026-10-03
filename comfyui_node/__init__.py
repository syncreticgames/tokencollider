"""ComfyUI custom node: load TokenCollider conditioning exports.

Self-contained on purpose. This package imports only the standard library,
`safetensors` and `torch`; it touches no ComfyUI module and subclasses no ComfyUI class,
which is what keeps it independent of ComfyUI's GPL-3.0.
Keep it that way.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
