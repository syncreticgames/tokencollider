extends RefCounted
## Every scene must instantiate headlessly, and every node path a script
## reaches for must exist. A typo in a .tscn node name is invisible until the
## screen is opened, which for a menu means "in front of a user".
var harness

const SCENES := {
	"res://ui/MainMenu.tscn": ["Margin/Rows/Explore", "Margin/Rows/Settings",
		"Margin/Rows/Credits", "Margin/Rows/Quit"],
	"res://ui/SettingsPanel.tscn": ["Margin/Rows/Close", "Margin/Rows/Reset",
		"Margin/Rows/Port/Value", "Margin/Rows/Normalized", "Margin/Rows/Note"],
	"res://ui/CreditsPanel.tscn": ["Margin/Rows/Scroll/Body", "Margin/Rows/Close"],
}

func test_scenes_instantiate_and_have_their_nodes() -> void:
	for path in SCENES:
		var packed = load(path)
		harness.ok(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var inst = packed.instantiate()
		harness.ok(inst != null, "%s instantiates" % path)
		for np in SCENES[path]:
			harness.ok(inst.get_node_or_null(np) != null,
				"%s has %s" % [path, np])
		inst.free()

func test_menu_emits_navigation_not_action() -> void:
	## The menu must not know how to do anything, only to ask.
	var inst = load("res://ui/MainMenu.tscn").instantiate()
	for sig in ["explore_requested", "settings_requested", "credits_requested",
			"quit_requested"]:
		harness.ok(inst.has_signal(sig), "menu emits %s" % sig)
	inst.free()

func test_credits_include_the_engine_notice() -> void:
	## Godot is MIT and an export embeds it, so the notice has to be reachable
	## from the build. If this ever returns nothing, the obligation is unmet.
	var text: String = load("res://ui/CreditsPanel.gd").engine_notices()
	harness.ok(text.length() > 500, "engine notice is substantial")
	harness.ok(text.contains("Godot"), "names the engine")
	harness.ok(text.to_lower().contains("permission"), "carries the MIT grant")
