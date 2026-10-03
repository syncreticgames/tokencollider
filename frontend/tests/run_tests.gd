extends SceneTree
## Headless GDScript test runner.
##
##   godot --headless --path frontend --script tests/run_tests.gd
##
## Every file in tests/ named test_*.gd is loaded, instantiated, and every
## method starting with `test_` is called. A test fails by calling fail();
## anything else is a pass. Exit code is non-zero if any test failed, so this
## drops straight into the Python gate (tests/run.py) alongside the rest.

var failures: Array[String] = []
var checks := 0

func ok(cond: bool, what: String) -> void:
	checks += 1
	if not cond:
		failures.append(what)

func eq(a, b, what: String) -> void:
	checks += 1
	if a != b:
		failures.append("%s: %s != %s" % [what, a, b])

func close(a: float, b: float, tol: float, what: String) -> void:
	checks += 1
	if absf(a - b) > tol:
		failures.append("%s: %f != %f (tol %f)" % [what, a, b, tol])

func _init() -> void:
	var dir := DirAccess.open("res://tests")
	if dir == null:
		push_error("no tests directory")
		quit(1)
		return
	var names := dir.get_files()
	names.sort()
	var ran := 0
	for f in names:
		if not f.begins_with("test_") or not f.ends_with(".gd"):
			continue
		var script = load("res://tests/" + f)
		# A script that failed to compile still loads as a GDScript but cannot
		# be instantiated. Report that as a failure rather than crashing the
		# runner, or a broken suite looks like a suite that does not exist.
		if script == null or not script.can_instantiate():
			failures.append("%s: script failed to compile" % f)
			continue
		var suite = script.new()
		suite.harness = self
		for m in suite.get_method_list():
			var name: String = m["name"]
			if name.begins_with("test_"):
				suite.call(name)
				ran += 1
	if failures.is_empty():
		print("ok: %d gdscript checks in %d tests" % [checks, ran])
		quit(0)
	else:
		for f in failures:
			print("FAIL ", f)
		print("%d/%d checks failed" % [failures.size(), checks])
		quit(1)
