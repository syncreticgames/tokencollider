extends RefCounted
## The Context is the seam the whole viewport split rests on. What matters is
## that a component can be given one in a test: no scene, no sidecar, no Label.
const Context := preload("res://viewport/Context.gd")
const ViewState := preload("res://lib/ViewState.gd")

var harness

func test_defaults_are_safe_to_call() -> void:
	## A component constructed before anything is injected must not crash.
	## Defaults are inert, not null.
	var ctx = Context.new()
	harness.eq(ctx.api.call("/anything"), null, "default api returns null")
	ctx.report.call("hello")            # must not throw
	harness.eq(ctx.signature(), "", "no view yet, empty signature")
	harness.eq(ctx.world_scale, 1.0, "default scale")
	harness.ok(ctx.landmarks.is_empty(), "no landmarks yet")

func test_injection_is_observable() -> void:
	var ctx = Context.new()
	var heard: Array[String] = []
	ctx.report = func(what: String) -> void: heard.append(what)
	ctx.api = func(path: String, _body = null): return {"echo": path}
	ctx.report.call("status one")
	harness.eq(heard.size(), 1, "report reaches the injected sink")
	var got: Dictionary = ctx.api.call("/layout")
	harness.eq(got["echo"], "/layout", "api reaches the injected transport")

func test_view_is_shared_not_copied() -> void:
	## Components compare signatures against the live view; a copy would let
	## a stale component think its cached positions were still valid.
	var ctx = Context.new()
	var v = ViewState.new()
	v.layer_lo = 8
	v.layer_hi = 20
	ctx.view = v
	harness.eq(ctx.signature(), "8-20/norm/local", "signature from the view")
	v.layer_hi = 35
	harness.eq(ctx.signature(), "8-35/norm/local", "follows the live view")

func test_scales_have_exactly_one_home() -> void:
	## REGRESSION (08232026). world_scale and locked_scale existed on BOTH
	## Main.gd and the Context after the split, so Constellations wrote one
	## copy while the layout placer read the other. Trails rendered at a scale
	## nothing else used (tiny curves), and the depth-stable branch that makes
	## spheres land on their trail lines was dead, so toggling normalized and
	## absolute changed nothing.
	##
	## Nothing in a test can stop someone re-declaring a field in Main.gd, so
	## this asserts the shape the fix relies on: both scales live here, and a
	## component writing one is observed by anyone reading it.
	var ctx = Context.new()
	harness.eq(ctx.world_scale, 1.0, "world_scale starts at 1")
	harness.eq(ctx.locked_scale, 0.0, "locked_scale starts at 0, meaning no trails")
	# A component locks a scale; every other reader must see it immediately.
	ctx.locked_scale = ctx.WORLD_TARGET / 8.0
	ctx.world_scale = ctx.locked_scale
	harness.close(ctx.world_scale, ctx.WORLD_TARGET / 8.0, 1e-6,
		"a locked scale is visible to every reader")
	harness.ok(ctx.locked_scale > 0.0,
		"locked_scale > 0 is what selects the depth-stable branch")
