extends Control
## Settings, backed by lib/Config.gd.
##
## Owns no state of its own: every control reads its value from Config on show
## and writes it back on change. A panel that keeps a private copy of a setting
## is a panel that disagrees with the rest of the app the first time anything
## else writes one.

signal closed
signal applied  ## something changed that a live session should react to

var settings: RefCounted = null  # lib/Config.gd, injected by the caller


func _ready() -> void:
	$Margin/Rows/Close.pressed.connect(func(): closed.emit())
	$Margin/Rows/Reset.pressed.connect(_on_reset)
	$Margin/Rows/Port/Value.value_changed.connect(_on_port_changed)
	$Margin/Rows/Normalized.toggled.connect(_on_normalized_toggled)
	refresh()


func refresh() -> void:
	## Pull every control from Config. Called on show, so reopening the panel
	## can never display a stale value.
	if settings == null:
		return
	$Margin/Rows/Port/Value.value = settings.get_value("sidecar/port")
	$Margin/Rows/Normalized.button_pressed = settings.get_value("view/normalized")


func _on_port_changed(v: float) -> void:
	if settings == null:
		return
	settings.set_value("sidecar/port", int(v))
	settings.save_settings()
	# The running sidecar keeps the port it was started on; this takes effect
	# next launch. Say so rather than letting the user think it hot-swapped.
	$Margin/Rows/Note.text = "Port applies on next launch."
	applied.emit()


func _on_normalized_toggled(on: bool) -> void:
	if settings == null:
		return
	settings.set_value("view/normalized", on)
	settings.save_settings()
	applied.emit()


func _on_reset() -> void:
	if settings == null:
		return
	settings.reset()
	refresh()
	$Margin/Rows/Note.text = "Settings reset to defaults."
	applied.emit()
