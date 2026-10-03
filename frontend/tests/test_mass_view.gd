extends RefCounted
## Mass view scales landmarks by weight and tethers the heavy ones. The rule
## worth pinning is the normalization: against the uniform share, so the look
## is the same at 30 landmarks or 2000.
const MassView := preload("res://viewport/MassView.gd")

var harness

func _landmark() -> Node3D:
	var n := Node3D.new()
	var shape := MeshInstance3D.new()
	shape.name = "Shape"
	n.add_child(shape)
	return n

func _field(names: Array) -> Dictionary:
	var d := {}
	for n in names:
		d[n] = _landmark()
	return d

func test_uniform_weights_leave_everything_baseline() -> void:
	var mv = MassView.new()
	var lm := _field(["a", "b", "c"])
	mv.show_blend({"a": 1.0, "b": 1.0, "c": 1.0}, lm, Vector3.ZERO)
	for k in lm:
		var s: Vector3 = lm[k].get_node("Shape").scale
		harness.close(s.x, 0.9, 0.01, "uniform weight -> baseline scale")
	mv.free()
	for k in lm:
		lm[k].free()

func test_heavy_landmark_swells_and_light_one_shrinks() -> void:
	var mv = MassView.new()
	var lm := _field(["hub", "x", "y", "z"])
	mv.show_blend({"hub": 9.0, "x": 0.1, "y": 0.1, "z": 0.1}, lm, Vector3.ZERO)
	var hub: float = lm["hub"].get_node("Shape").scale.x
	var small: float = lm["x"].get_node("Shape").scale.x
	harness.ok(hub > small, "hub swells past a light node (%f > %f)" % [hub, small])
	harness.ok(hub <= 3.0, "scale is clamped, no runaway")
	harness.ok(small >= 0.4, "shrink is clamped too")
	mv.free()
	for k in lm:
		lm[k].free()

func test_scale_is_free_of_universe_size() -> void:
	## The same relative weight must look the same in a small and a large
	## universe, which is the whole reason for normalizing by uniform share.
	var small_lm := _field(["a", "b", "c"])
	var big_names := ["a"]
	for i in 100:
		big_names.append("n%d" % i)
	var big_lm := _field(big_names)
	var w_small := {"a": 3.0, "b": 1.0, "c": 1.0}
	var w_big := {"a": 3.0}
	for i in 100:
		w_big["n%d" % i] = 1.0
	var mv1 = MassView.new()
	mv1.show_blend(w_small, small_lm, Vector3.ZERO)
	var mv2 = MassView.new()
	mv2.show_blend(w_big, big_lm, Vector3.ZERO)
	var s1: float = small_lm["a"].get_node("Shape").scale.x
	var s2: float = big_lm["a"].get_node("Shape").scale.x
	# Ratio to the mean share differs slightly, but both must sit in the same
	# visual band rather than one ballooning with universe size.
	harness.ok(absf(s1 - s2) < 0.6, "same look at 3 and 101 landmarks (%f vs %f)" % [s1, s2])
	mv1.free()
	mv2.free()
	for k in small_lm:
		small_lm[k].free()
	for k in big_lm:
		big_lm[k].free()

func test_clear_restores_every_scale() -> void:
	var mv = MassView.new()
	var lm := _field(["a", "b"])
	mv.show_blend({"a": 5.0, "b": 0.2}, lm, Vector3.ZERO)
	harness.ok(mv.is_showing(), "showing after a blend")
	mv.clear(lm)
	harness.ok(not mv.is_showing(), "cleared")
	for k in lm:
		harness.close(lm[k].get_node("Shape").scale.x, 1.0, 1e-6, "scale restored")
	mv.free()
	for k in lm:
		lm[k].free()

func test_clear_tethers_keeps_the_masses() -> void:
	## A layout refresh moves landmarks, so the lines go but the scaled
	## balls ride their nodes and stay.
	var mv = MassView.new()
	var lm := _field(["hub", "x", "y", "z"])
	mv.show_blend({"hub": 9.0, "x": 0.1, "y": 0.1, "z": 0.1}, lm, Vector3.ZERO)
	harness.ok(mv.get_child_count() == 1, "tethers drawn for the hub")
	var hub: float = lm["hub"].get_node("Shape").scale.x
	mv.clear_tethers()
	harness.ok(mv.is_showing(), "still showing masses")
	harness.close(lm["hub"].get_node("Shape").scale.x, hub, 1e-6, "hub keeps its mass")
	mv.free()
	for k in lm:
		lm[k].free()
