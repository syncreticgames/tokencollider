extends RefCounted
## Durable settings, in `user://settings.cfg`.
##
## Nothing in this app used to persist: profile, port, window size and view
## preferences all died with the process, and every launch started from the
## defaults baked into the scene. This is the one place that changes.
##
## Design rules, because a settings layer that grows organically becomes the
## worst file in a project:
##
## - Every key has a DEFAULT declared in DEFAULTS. `get_value` never invents
##   one at the call site, so the set of settings is readable in one place.
## - A key not in DEFAULTS is a programming error and says so, rather than
##   silently returning null and surfacing three layers away.
## - Reads never fail. A missing, unreadable or partially corrupt file gives
##   defaults, because a user whose config got truncated by a bad shutdown
##   should get a working app, not a stack trace.
## - Types are checked on read. A hand-edited file with `port = "8765"` gives
##   the default rather than propagating a String into integer arithmetic.

const DEFAULT_PATH := "user://settings.cfg"

## Where settings live. Only the test runner changes it, so the suites never
## read or overwrite the user's real settings (they share `user://`).
static var path := DEFAULT_PATH

const DEFAULTS := {
	"sidecar/port": 8765,
	"sidecar/host": "127.0.0.1",
	"session/profile": "",          # empty = let the sidecar decide
	"session/last_universe": "",
	"view/normalized": true,
	"view/world_axes": false,
	"window/width": 1280,
	"window/height": 720,
	"window/maximized": false,
}

var _cfg := ConfigFile.new()
var _loaded := false


func _split(key: String) -> Array:
	var parts := key.split("/", true, 1)
	return [parts[0], parts[1]] if parts.size() == 2 else ["general", key]


func load_settings() -> void:
	## Errors are deliberately swallowed. Any failure leaves an empty
	## ConfigFile, which yields defaults for every key.
	_cfg = ConfigFile.new()
	_cfg.load(path)
	_loaded = true


func get_value(key: String):
	if not DEFAULTS.has(key):
		push_error("Config: unknown key %s (add it to DEFAULTS)" % key)
		return null
	if not _loaded:
		load_settings()
	var sec: Array = _split(key)
	var fallback = DEFAULTS[key]
	var got = _cfg.get_value(sec[0], sec[1], fallback)
	# A hand-edited file can hold anything; a wrong type is worse than absent.
	if typeof(got) != typeof(fallback):
		return fallback
	return got


func set_value(key: String, value) -> void:
	if not DEFAULTS.has(key):
		push_error("Config: unknown key %s (add it to DEFAULTS)" % key)
		return
	if not _loaded:
		load_settings()
	var sec: Array = _split(key)
	_cfg.set_value(sec[0], sec[1], value)


func save_settings() -> bool:
	return _cfg.save(path) == OK


func reset() -> void:
	## Back to shipped defaults, on disk as well as in memory, so "reset
	## settings" in the UI does not leave a stale file to resurrect later.
	_cfg = ConfigFile.new()
	_loaded = true
	_cfg.save(path)
