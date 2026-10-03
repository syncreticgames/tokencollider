extends RefCounted
## Settings persistence. The interesting cases are the broken ones: a user
## whose config was truncated by a bad shutdown, or who hand-edited a value to
## the wrong type, should still get a working app.
const Config := preload("res://lib/Config.gd")

var harness

func _fresh() -> RefCounted:
	var c = Config.new()
	c.reset()
	return c

func test_defaults_when_nothing_saved() -> void:
	var c = _fresh()
	harness.eq(c.get_value("sidecar/port"), 8765, "default port")
	harness.eq(c.get_value("view/normalized"), true, "default normalized")
	harness.eq(c.get_value("session/profile"), "", "default profile")

func test_roundtrip_survives_a_new_instance() -> void:
	var c = _fresh()
	c.set_value("sidecar/port", 8799)
	c.set_value("session/profile", "krea2")
	harness.ok(c.save_settings(), "save reports success")
	var d = Config.new()
	harness.eq(d.get_value("sidecar/port"), 8799, "port persisted")
	harness.eq(d.get_value("session/profile"), "krea2", "profile persisted")
	d.reset()

func test_unknown_key_is_refused_not_invented() -> void:
	var c = _fresh()
	c.set_value("nonsense/key", 1)      # pushes an error, stores nothing
	harness.eq(c.get_value("nonsense/key"), null, "unknown key reads null")

func test_wrong_type_on_disk_falls_back() -> void:
	## Someone hand-edits settings.cfg and writes a quoted port. Without the
	## type check that String flows into integer arithmetic somewhere else.
	var raw := ConfigFile.new()
	raw.set_value("sidecar", "port", "8765")
	raw.save(Config.PATH)
	var c = Config.new()
	harness.eq(c.get_value("sidecar/port"), 8765, "string port falls back to int default")
	c.reset()

func test_corrupt_file_still_yields_defaults() -> void:
	var f := FileAccess.open(Config.PATH, FileAccess.WRITE)
	f.store_string("this is not a config file [[[ = = ]]] \n\n garbage")
	f.close()
	var c = Config.new()
	harness.eq(c.get_value("window/width"), 1280, "corrupt file yields defaults")
	c.reset()
