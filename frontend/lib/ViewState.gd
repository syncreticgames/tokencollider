extends RefCounted
## What the user is currently looking at: layer band, axes chart, scaling.
##
## Small, pure, and worth its own file because one thing here is subtle and
## load-bearing: the VIEW SIGNATURE. A cursor dropped in one view is
## meaningless in another. Layer bands are different subspaces, and world and
## local axes are different charts of the same layer, so reprojecting a point
## across either is silently wrong rather than approximate. Everything that
## caches a position against a view compares this string.
##
## Getting that comparison wrong produces an export from coordinates the user
## never selected, and nothing downstream can detect it. That is exactly the
## kind of rule that should live in one tested place rather than be re-derived
## at each call site.

var layer_lo := -1     # -1 until the sidecar reports the model's layer count
var layer_hi := -1
var world_axes := false
var normalized := true


func is_ready() -> bool:
	return layer_hi >= 0


func is_band() -> bool:
	return layer_lo != layer_hi


func layer_param() -> String:
	## The `layer` query parameter: a single index, or "lo-hi" for a band.
	if layer_lo == layer_hi:
		return "%d" % layer_hi
	return "%d-%d" % [layer_lo, layer_hi]


func layer_body_value() -> Variant:
	## The same thing for a JSON body, where the sidecar accepts an int or a
	## two-element array.
	return layer_hi if layer_lo == layer_hi else [layer_lo, layer_hi]


func axes_value() -> String:
	return "world" if world_axes else "local"


func signature() -> String:
	## The identity of the space on screen. Compare with `==`; never try to
	## translate a position from one signature to another.
	return layer_param() + ("/norm" if normalized else "/abs") + "/" + axes_value()


func layout_path() -> String:
	var path := "/layout?axes=" + axes_value()
	if is_ready():
		path += "&layer=" + layer_param()
	return path


func to_dict() -> Dictionary:
	return {"layer_lo": layer_lo, "layer_hi": layer_hi,
			"world_axes": world_axes, "normalized": normalized}


func from_dict(d: Dictionary) -> void:
	## Restoring a saved session. Missing keys keep the current value, so an
	## older session file that predates a field still loads.
	layer_lo = int(d.get("layer_lo", layer_lo))
	layer_hi = int(d.get("layer_hi", layer_hi))
	world_axes = bool(d.get("world_axes", world_axes))
	normalized = bool(d.get("normalized", normalized))
