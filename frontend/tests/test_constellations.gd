extends RefCounted
## Trails. The bug worth guarding: the component must publish the scales it
## computes to the shared Context, because the layout placer reads them there.
const Constellations := preload("res://viewport/Constellations.gd")
const Context := preload("res://viewport/Context.gd")
const ViewState := preload("res://lib/ViewState.gd")

var harness

func _rig() -> Array:
	var ctx = Context.new()
	ctx.view = ViewState.new()
	var c = Constellations.new()
	c.ctx = ctx
	return [c, ctx]

func test_starts_hidden_and_clears_the_locked_scale() -> void:
	var rig := _rig()
	var c = rig[0]
	var ctx = rig[1]
	harness.ok(not c.is_showing(), "no trails before drawing")
	ctx.locked_scale = 5.0
	c.clear()
	harness.eq(ctx.locked_scale, 0.0, "clear() releases the locked scale")
	harness.ok(not c.is_showing(), "still hidden after clear")
	c.free()

func test_locked_scale_is_published_not_kept_private() -> void:
	## The regression: the component used to keep its own locked_scale, so the
	## layout placer never saw it and absolute mode fell back to a different
	## scale. Assert the field it writes is the shared one.
	var rig := _rig()
	var c = rig[0]
	harness.ok(not ("locked_scale" in c), "component keeps no private copy")
	harness.ok("locked_scale" in rig[1], "the Context owns it")
	c.free()

class Pending extends RefCounted:
	## A sidecar reply the test releases by hand, so a trace can be in flight.
	signal landed(res)

func _trajectory() -> Dictionary:
	return {"extent": 1.0, "n_layers": 2, "layers": [0, 1],
		"trajectories": {"red": [{"coords": [0, 0, 0, 0, 0, 0], "color": "#ff0000"},
			{"coords": [1, 1, 1, 0, 0, 0], "color": "#ff0000"}]}}

func test_t_cancels_a_trace_in_flight() -> void:
	## T during a slow trace used to start a second draw, and T to turn trails
	## off before the reply landed was undone when it did.
	var rig := _rig()
	var c = rig[0]
	var ctx = rig[1]
	var pending := Pending.new()
	var calls := [0]
	ctx.api = func(_path, _body = null):
		calls[0] += 1
		return pending.landed
	c.toggle()                     # on: trace goes out
	c.toggle()                     # off, before it lands
	harness.eq(calls[0], 1, "the second T cancels; it doesn't trace again")
	pending.landed.emit(_trajectory())
	harness.ok(not c.is_showing(), "the late reply doesn't bring trails back")
	c.toggle()                     # on again
	pending.landed.emit(_trajectory())
	harness.ok(c.is_showing(), "a trace that is still wanted draws")
	c.free()
