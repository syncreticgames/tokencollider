extends RefCounted
## The view signature decides whether a dropped cursor is still valid. If two
## genuinely different views ever produce the same signature, the app will
## export coordinates the user did not select and nothing will notice.
const ViewState := preload("res://lib/ViewState.gd")

var harness

func _at(lo: int, hi: int, world := false, norm := true) -> RefCounted:
	var v = ViewState.new()
	v.layer_lo = lo
	v.layer_hi = hi
	v.world_axes = world
	v.normalized = norm
	return v

func test_single_layer_vs_band_encoding() -> void:
	harness.eq(_at(20, 20).layer_param(), "20", "single layer")
	harness.eq(_at(8, 20).layer_param(), "8-20", "band")
	harness.eq(_at(20, 20).layer_body_value(), 20, "body: int for a single layer")
	harness.eq(_at(8, 20).layer_body_value(), [8, 20], "body: array for a band")
	harness.ok(not _at(20, 20).is_band(), "single is not a band")
	harness.ok(_at(8, 20).is_band(), "band is a band")

func test_every_distinct_view_has_a_distinct_signature() -> void:
	## Exhaustive over the axes that define a view. Any collision here is a
	## silent wrong-coordinates bug.
	var seen := {}
	var views := []
	for lo in [8, 20]:
		for hi in [20, 35]:
			for world in [false, true]:
				for norm in [false, true]:
					views.append(_at(lo, hi, world, norm))
	for v in views:
		var sig: String = v.signature()
		harness.ok(not seen.has(sig), "signature unique: %s" % sig)
		seen[sig] = true
	harness.eq(seen.size(), views.size(), "no signature collisions")

func test_signature_changes_with_each_axis_independently() -> void:
	var base: String = _at(8, 20, false, true).signature()
	harness.ok(_at(8, 21, false, true).signature() != base, "layer matters")
	harness.ok(_at(8, 20, true, true).signature() != base, "axes chart matters")
	harness.ok(_at(8, 20, false, false).signature() != base, "scaling matters")

func test_layout_path_omits_layer_until_known() -> void:
	var cold = ViewState.new()          # -1, sidecar has not reported yet
	harness.ok(not cold.is_ready(), "cold view is not ready")
	harness.eq(cold.layout_path(), "/layout?axes=local", "no layer before it is known")
	harness.eq(_at(8, 20).layout_path(), "/layout?axes=local&layer=8-20", "layer once known")

func test_roundtrip_and_forward_compatible_restore() -> void:
	var v = _at(8, 20, true, false)
	var w = ViewState.new()
	w.from_dict(v.to_dict())
	harness.eq(w.signature(), v.signature(), "roundtrip preserves the view")
	## A session file written before a field existed must still load.
	var partial = _at(5, 5, false, true)
	partial.from_dict({"layer_hi": 30})
	harness.eq(partial.layer_hi, 30, "present key applied")
	harness.eq(partial.layer_lo, 5, "absent key keeps current value")

func test_matches_layout_rejects_replies_for_another_view() -> void:
	## Out-of-order replies: only the one for the view on screen applies.
	var single = _at(30, 30)
	harness.ok(single.matches_layout({"layer": 30, "axes": "local"}), "same layer applies")
	harness.ok(single.matches_layout({"layer": 30.0, "axes": "local"}), "JSON floats compare as ints")
	harness.ok(not single.matches_layout({"layer": 20, "axes": "local"}), "older layer is stale")
	harness.ok(not single.matches_layout({"layer": [20, 30], "axes": "local"}), "band reply for a single view is stale")
	harness.ok(not single.matches_layout({"axes": "local"}), "reply without a layer is stale once the layer is known")
	var band = _at(8, 20)
	harness.ok(band.matches_layout({"layer": [8, 20], "axes": "local"}), "same band applies")
	harness.ok(not band.matches_layout({"layer": [8, 21], "axes": "local"}), "other band is stale")
	harness.ok(not band.matches_layout({"layer": 20, "axes": "local"}), "single reply for a band view is stale")
	harness.ok(ViewState.new().matches_layout({"layer": 36, "axes": "local"}), "cold view takes any reply")

func test_matches_layout_axes() -> void:
	var world = _at(20, 20, true)
	harness.ok(world.matches_layout({"layer": 20, "axes": "world", "has_world": true}), "world reply for world view")
	harness.ok(world.matches_layout({"layer": 20, "axes": "local", "has_world": false}),
		"local reply with no atlas is the fallback, not stale")
	harness.ok(not world.matches_layout({"layer": 20, "axes": "local", "has_world": true}),
		"local reply while an atlas exists was sent before the switch")
	harness.ok(not _at(20, 20, false).matches_layout({"layer": 20, "axes": "world", "has_world": true}),
		"world reply for a local view is stale")
