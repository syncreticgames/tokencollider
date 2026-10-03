extends RefCounted
## Ghosts belong to the view they were queried in. The marker group and the
## view it belongs to live in one place, so a rescale moves the markers that
## are actually on screen and realize() sees the view they came from.
const Ghosts := preload("res://viewport/Ghosts.gd")
const Context := preload("res://viewport/Context.gd")
const ViewState := preload("res://lib/ViewState.gd")

var harness

func _setup():
	var ctx = Context.new()
	ctx.view = ViewState.new()
	var g = Ghosts.new()
	g.ctx = ctx
	return g

func test_begin_records_the_view_realize_checks() -> void:
	var g = _setup()
	g.begin(g.ctx.signature())
	g.make_ghost("ember", [1.0, 0.0, 0.0], 0.9)
	harness.eq(g.view, g.ctx.signature(), "queried view recorded")
	harness.eq(g.ghosts.get_child_count(), 1, "marker in the group")
	g.free()

func test_rescale_moves_the_markers_on_screen() -> void:
	var g = _setup()
	g.begin(g.ctx.signature())
	g.make_ghost("ember", [1.0, 2.0, 0.0], 0.9)
	g.ctx.world_scale = 3.0
	g.follow_layout(g.ctx.signature())
	var pos: Vector3 = g.ghosts.get_child(0).position
	harness.close(pos.y, 6.0, 1e-5, "marker follows the new scale")
	g.free()

func test_another_view_clears_the_markers() -> void:
	var g = _setup()
	g.begin("20/norm/local")
	g.make_ghost("ember", [1.0, 0.0, 0.0], 0.9)
	g.follow_layout("8/norm/local")
	harness.ok(g.ghosts == null, "markers from another view are dropped")
	g.free()
