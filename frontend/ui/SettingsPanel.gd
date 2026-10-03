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

## The saved port only matters to a Godot started by hand: `tokencollider
## view` hands the desktop viewport its port, and the browser viewport talks
## to the server it was loaded from. In a browser the setting can't apply at
## all, so it isn't shown there.
var show_port := not OS.has_feature("web")
const PORT_NOTE := "Only for a viewport started by hand; tokencollider view sets its own port."


func _ready() -> void:
	$Margin/Rows/Port.visible = show_port
	$Margin/Rows/Port.tooltip_text = PORT_NOTE
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
	# Say what this actually changes, rather than letting the user think
	# it moved the running sidecar or the next `tokencollider view`.
	$Margin/Rows/Note.text = PORT_NOTE
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
