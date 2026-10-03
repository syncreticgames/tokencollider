extends Control
## Credits, and the engine's license obligation discharged in code.
##
## Godot is MIT, and an exported binary EMBEDS the engine, so its copyright
## notice has to ship with the build. Godot hands you the whole thing at
## runtime, including the notices of every third-party component it bundles
## (FreeType, zlib and the rest), so printing what the engine reports is both
## the least work and the most correct answer. Do not replace this with a
## hand-written list; it will drift and it will be wrong.
##
## See docs/credits.md for what each dependency asks of us.

signal closed

const PROJECT_CREDITS := """[b]TokenCollider[/b]
MIT licensed. Copyright (c) 2026 Syncretic Games LLC.

[b]Built on[/b]
• Godot Engine (MIT) — the viewport
• ComfyUI by comfyanonymous (GPL-3.0) — renders every export
• ai-toolkit by Ostris (MIT) — trains the LoRAs
• Qwen3-VL-4B / Qwen3-4B by Alibaba — the text encoders
• Krea 2, Z-Image — the diffusion models
• Oklab by Björn Ottosson — the color space points are drawn in
• PyTorch, transformers, safetensors, numpy, Pillow, PyYAML

Full license notes: docs/credits.md
"""


func _ready() -> void:
	var body: RichTextLabel = $Margin/Rows/Scroll/Body
	body.bbcode_enabled = true
	body.text = PROJECT_CREDITS + "\n" + engine_notices()
	$Margin/Rows/Close.pressed.connect(func(): closed.emit())


static func engine_notices() -> String:
	## Everything Godot says about itself and what it bundles.
	var out := "[b]Godot Engine[/b]\n" + Engine.get_license_text() + "\n"
	for entry in Engine.get_copyright_info():
		out += "\n[b]%s[/b]\n" % entry.get("name", "")
		for part in entry.get("parts", []):
			for c in part.get("copyright", []):
				out += "  %s\n" % c
			out += "  License: %s\n" % part.get("license", "")
	return out
