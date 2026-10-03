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
