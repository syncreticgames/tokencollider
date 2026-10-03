extends Control
## The front door. Owns navigation and nothing else.
##
## It does not know how to embed, how to talk to the sidecar, or what a
## universe is. It emits what the user asked for and lets the app decide, which
## is what keeps adding a screen from meaning editing the viewport.

signal explore_requested
signal settings_requested
signal credits_requested
signal quit_requested


func _ready() -> void:
	$Margin/Rows/Explore.pressed.connect(func(): explore_requested.emit())
	$Margin/Rows/Settings.pressed.connect(func(): settings_requested.emit())
	$Margin/Rows/Credits.pressed.connect(func(): credits_requested.emit())
	$Margin/Rows/Quit.pressed.connect(func(): quit_requested.emit())
	$Margin/Rows/Explore.grab_focus()
