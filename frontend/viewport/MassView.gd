extends Node3D
## The blend, made visible: landmark balls swell by weight, tethers draw pull.
##
## Owns its own child nodes and nothing else. It is handed the solved weights
## and the landmark nodes; it does not know how to ask the sidecar for them,
## which is what lets it be reused (a settings preview, a second viewport)
## and what stops Main.gd growing another API call.
##
## Green is positive weight, red is negative. A red tether does not mean
## "opposite": in an unconstrained solve it only means the cursor sits outside
## the landmark hull.

const POSITIVE := Color(0.4, 1.0, 0.5)
const NEGATIVE := Color(1.0, 0.35, 0.3)
const TETHER_MIN_RATIO := 2.0  ## only tether the clearly-above-average pullers

var _tethers: Node3D = null
var _massed := false


func show_blend(weights: Dictionary, landmarks: Dictionary, origin: Vector3) -> void:
	## Normalised against the UNIFORM SHARE (total / n), not max or a
	## percentile. Max flattens everything when one hub dominates; a percentile
	## inflates hundreds of average nodes at universe scale and the whole scene
	## balloons, which reads as a view reset. Average contributors stay
	## baseline size and only genuinely heavy nodes swell. Scale-free: the same
	## look at 30 landmarks or 2000.
	clear(landmarks)
	var total := 0.0
	for t in weights:
		total += absf(float(weights[t]))
	if weights.is_empty() or total <= 0.0:
		return
	var mean_share := total / weights.size()
	_tethers = Node3D.new()
	add_child(_tethers)
	_massed = true
	for t in weights:
		if not landmarks.has(t):
			continue
		var w := float(weights[t])
		var ratio := absf(w) / mean_share
		var node: Node3D = landmarks[t]
		(node.get_node("Shape") as MeshInstance3D).scale = \
			Vector3.ONE * clampf(0.9 * pow(ratio, 0.4), 0.4, 3.0)
		if ratio >= TETHER_MIN_RATIO:
			_draw_tether(origin, node.position, w >= 0.0,
				clampf(ratio / 10.0, 0.1, 1.0))


func _draw_tether(from: Vector3, to: Vector3, positive: bool, frac: float) -> void:
	var mesh := ImmediateMesh.new()
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mi.material_override = mat
	var color := POSITIVE if positive else NEGATIVE
	color.a = 0.15 + 0.75 * frac
	mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	mesh.surface_set_color(color)
	mesh.surface_add_vertex(from)
	mesh.surface_set_color(color)
	mesh.surface_add_vertex(to)
	mesh.surface_end()
	_tethers.add_child(mi)


func clear_tethers() -> void:
	## Lines only. Landmarks are about to move, so line endpoints would lie;
	## the masses ride their nodes and stay.
	if _tethers != null:
		_tethers.queue_free()
		_tethers = null


func clear(landmarks: Dictionary) -> void:
	clear_tethers()
	if _massed:
		for node in landmarks.values():
			(node.get_node("Shape") as MeshInstance3D).scale = Vector3.ONE
		_massed = false


func is_showing() -> bool:
	return _massed
