extends RefCounted
## Navigation between the three screens. The overlay must route every signal
## a screen emits, because an unconnected button is a dead end the user finds
## before any test does.
const MenuOverlay := preload("res://ui/MenuOverlay.gd")
const Config := preload("res://lib/Config.gd")

var harness

func test_starts_closed() -> void:
	var o = MenuOverlay.new()
	o._ready()
	harness.ok(not o.is_open(), "hidden until asked for")
	harness.eq(o.layer, 100, "sits above the viewport UI")
	o.free()

func test_open_shows_the_menu_and_close_emits_resumed() -> void:
	var o = MenuOverlay.new()
	o._ready()
	var resumed := [0]
	o.resumed.connect(func(): resumed[0] += 1)
	o.open()
	harness.ok(o.is_open(), "open() shows it")
	harness.ok(o._screen != null, "a screen is mounted")
	o.close()
	harness.ok(not o.is_open(), "close() hides it")
	harness.eq(resumed[0], 1, "close emits resumed exactly once")
	harness.ok(o._screen == null, "screen dropped on close")
	o.free()

func test_every_menu_signal_is_routed() -> void:
	## The regression this guards: a screen emitting a signal nobody connected
	## looks like a button that does nothing.
	var o = MenuOverlay.new()
	o._ready()
	o.open()
	var menu = o._screen
	for sig in ["explore_requested", "settings_requested", "credits_requested",
			"quit_requested"]:
		harness.ok(menu.get_signal_connection_list(sig).size() >= 1,
			"%s is connected" % sig)
	o.free()

func test_settings_receives_the_config_object() -> void:
	var o = MenuOverlay.new()
	o._ready()
	o.settings = Config.new()
	o.open()
	o._show(MenuOverlay.SETTINGS)
	harness.ok(o._screen.settings != null, "settings screen got the config")
	harness.ok(o._screen.has_signal("closed"), "and can navigate back")
	o.free()
