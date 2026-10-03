extends Node3D
## Trails: each landmark's path across the encoder's layers, as a polyline.
## Watching clusters tear apart and re-form while scrubbing is the point.
##
## Owns its trail geometry. The scales it computes live on the Context because
## the layout placer reads them too; keeping a private copy is what broke this
## the first time. It never
## reaches into the viewport: everything it needs arrives through a Context,
## and everything it wants done to the world is a signal. That is what makes
## it movable, testable, and safe to enhance without reading Main.gd.

signal needs_refresh   ## re-place landmarks under the freshly locked scale
signal trails_stale    ## the view changed; these trails no longer mean anything

var ctx: RefCounted = null   ## viewport/Context.gd
var _trails: Node3D = null


func is_showing() -> bool:
	return _trails != null


func clear() -> void:
	## Drop trails outright rather than redrawing: a redraw would warm every
	## landmark at every layer, and an old universe's trails are moot anyway.
	if _trails != null:
		_trails.queue_free()
		_trails = null
	ctx.locked_scale = 0.0


func toggle() -> void:
	if _trails != null:
		_trails.queue_free()
		_trails = null
		ctx.report.call("constellation off")
		return
	await draw()

func draw() -> void:
	ctx.report.call("tracing constellation ...")
	var res = await ctx.api.call("/trajectory")
	if res == null:
		return
	if float(res["extent"]) > 0.0:
		ctx.locked_scale = ctx.WORLD_TARGET / float(res["extent"])
		if not ctx.view.normalized:
			ctx.world_scale = ctx.locked_scale  # trails below and spheres share it at once
	# Per-depth draw scale: constant in absolute mode; in normalized mode each
	# depth is rescaled by its own RMS spread — the same statistic the layout
	# view uses, so spheres stay glued to their trails in both modes.
	var n := int(res["n_layers"])
	var depth_scales: Array[float] = []
	for d in n:
		if ctx.view.normalized:
			var slice := []
			for text in res["trajectories"]:
				slice.append(res["trajectories"][text][d])
			var rms: float = ctx.rms_spread(slice)
			depth_scales.append(ctx.WORLD_TARGET / rms if rms > 0.0 else 1.0)
		else:
			depth_scales.append(ctx.world_scale)
	if _trails != null:
		_trails.queue_free()
	_trails = Node3D.new()
	add_child(_trails)
	for text in res["trajectories"]:
		trace_trail(res["trajectories"][text], depth_scales)
	var mode := "normalized" if ctx.view.normalized else "absolute"
	ctx.report.call("constellation (%s): %d trails x %d layers (faint end = layer 0, segment color = that depth's color axes) — T to hide" % [
		mode, res["trajectories"].size(), n])
	needs_refresh.emit()  # re-place spheres under the freshly chosen scale

func mark_stale() -> void:
	# Landmark set changed: extent and every trail are outdated.
	if _trails != null:
		draw()
	else:
		ctx.locked_scale = 0.0

func trace_trail(points: Array, depth_scales: Array[float]) -> void:
	if points.size() < 2:
		return
	var mesh := ImmediateMesh.new()
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mi.material_override = mat
	mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	var last := points.size() - 1
	for i in points.size():
		# Hue/chroma = that depth's own color axes (dims 4-6): the segment
		# shows the entanglement at that step. Alpha still ramps with depth
		# so direction stays legible.
		var c := Color.html(points[i]["color"])
		c.a = lerpf(0.2, 1.0, float(i) / float(last))
		mesh.surface_set_color(c)
		var p: Array = points[i]["coords"]
		mesh.surface_add_vertex(Vector3(p[0], p[1], p[2]) * depth_scales[i])
	mesh.surface_end()
	_trails.add_child(mi)

