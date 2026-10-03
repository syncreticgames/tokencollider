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

## In a browser, quitting stops the engine and leaves a dead page, and closing
## the tab is how anyone leaves a web page anyway. So there is no Quit there.
var can_quit := not OS.has_feature("web")


func _ready() -> void:
	$Margin/Rows/Quit.visible = can_quit
	$Margin/Rows/Explore.pressed.connect(func(): explore_requested.emit())
	$Margin/Rows/Settings.pressed.connect(func(): settings_requested.emit())
	$Margin/Rows/Credits.pressed.connect(func(): credits_requested.emit())
	$Margin/Rows/Quit.pressed.connect(func(): quit_requested.emit())
	$Margin/Rows/Explore.grab_focus()
