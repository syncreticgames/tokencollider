extends CanvasLayer
## The front door, reachable from inside the viewport.
##
## Owns navigation between the three UI screens and nothing else. Main.gd
## toggles it and listens for `resumed`; it never learns which screen is
## showing or how any of them work. Adding a fourth screen is a case in
## `_show()` plus a signal, not an edit to the viewport.
##
## An overlay rather than a boot scene on purpose: the viewport keeps its own
## launch path, so `tokencollider view` behaves exactly as it always has, and a broken
## menu can never stop someone reaching the tool. Promote it to the boot
## scene once it has been used in anger.

signal resumed        ## the user wants the viewport back
signal quit_requested

const MENU := preload("res://ui/MainMenu.tscn")
const SETTINGS := preload("res://ui/SettingsPanel.tscn")
const CREDITS := preload("res://ui/CreditsPanel.tscn")

var settings: RefCounted = null   ## lib/Config.gd, passed to the settings screen
var _screen: Control = null
var _scrim: ColorRect = null


func _ready() -> void:
	layer = 100          # above the viewport's own UI
	hide()


func is_open() -> bool:
	return visible


func open() -> void:
	if _scrim == null:
		_scrim = ColorRect.new()
		_scrim.color = Color(0, 0, 0, 0.82)
		_scrim.anchor_right = 1.0
		_scrim.anchor_bottom = 1.0
		_scrim.mouse_filter = Control.MOUSE_FILTER_STOP  # swallow clicks
		add_child(_scrim)
	_show(MENU)
	show()


func close() -> void:
	_drop_screen()
	hide()
	resumed.emit()


func _drop_screen() -> void:
	if _screen != null:
		_screen.queue_free()
		_screen = null


func _show(packed: PackedScene) -> void:
	_drop_screen()
	_screen = packed.instantiate()
	# Settings needs the config object; the others need nothing. Checked
	# rather than assumed, so a screen without the field is not a crash.
	if "settings" in _screen:
		_screen.settings = settings
	if _screen.has_signal("explore_requested"):
		_screen.explore_requested.connect(close)
		_screen.settings_requested.connect(func(): _show(SETTINGS))
		_screen.credits_requested.connect(func(): _show(CREDITS))
		_screen.quit_requested.connect(func(): quit_requested.emit())
	if _screen.has_signal("closed"):
		_screen.closed.connect(func(): _show(MENU))
	add_child(_screen)
