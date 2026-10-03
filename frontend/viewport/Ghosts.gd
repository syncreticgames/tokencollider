extends Node3D
## Interrogation results as in-world markers.
##
## A ghost is a word the sidecar found near the cursor, shown where it would
## sit if it were a landmark. Ghosts belong to the view they were queried in
## and go stale when it changes: a position from another chart is WRONG, not
## approximate, which is the rule lib/ViewState.gd exists to enforce.

signal applied_layout(layout: Dictionary)  ## realized ghosts became landmarks
signal trails_stale                        ## the landmark set changed

var ctx: RefCounted = null   ## viewport/Context.gd
var view := ""               ## the view signature these were queried in
var ghosts: Node3D = null    ## the marker group; built on first use, freed on clear


func _ensure_group() -> void:
	if ghosts == null:
		ghosts = Node3D.new()
		ghosts.name = "GhostGroup"
		add_child(ghosts)


func make_ghost(text: String, coords: Array, cosine: float) -> void:
	var node := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(0.25, 0.25, 0.25)
	node.mesh = box
	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	# Hotter = closer: cosine drives glow and opacity, so the local gradient
	# is readable at a glance without printed numbers.
	var heat := clampf(inverse_lerp(0.5, 1.0, cosine), 0.0, 1.0)
	mat.albedo_color = Color(1.0, 0.8, 0.4, 0.25 + 0.5 * heat)
	mat.emission_enabled = true
	mat.emission = Color(1.0, 0.7, 0.3)
	mat.emission_energy_multiplier = 0.2 + 0.8 * heat
	node.material_override = mat
	var lc := Vector3(coords[0], coords[1], coords[2])
	node.set_meta("lcoords", lc)
	node.set_meta("text", text)
	node.position = lc * ctx.world_scale
	var tag := Label3D.new()
	tag.text = text
	tag.position.y = 0.4
	tag.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	tag.font_size = 32
	tag.pixel_size = 0.008
	tag.modulate = Color(1.0, 0.85, 0.6, 0.5 + 0.5 * heat)
	node.add_child(tag)
	_ensure_group()
	ghosts.add_child(node)

func realize() -> void:
	## Promote an interrogation's vocab ghosts into real landmarks. They are
	## already cached (interrogate only ever reads the cache), so this costs
	## one refit and no forward pass.
	if ghosts == null or ghosts.get_child_count() == 0:
		ctx.report.call("no ghosts to realize — interrogate first (I)")
		return
	if view != ctx.signature():
		clear_ghosts()
		ctx.report.call("those ghosts belonged to another view — interrogate again")
		return
	var texts := []
	for g in ghosts.get_children():
		texts.append(str(g.get_meta("text")))
	ctx.report.call("realizing %d ghosts ..." % texts.size())
	var body := {"texts": texts}
	if ctx.view.layer_hi >= 0:
		body["layer"] = ctx.view.layer_body_value()
	body["axes"] = ctx.view.axes_value()
	var layout = await ctx.api.call("/landmarks", body)
	if layout == null:
		return
	clear_ghosts()
	applied_layout.emit(layout)
	ctx.report.call("realized %d — %d landmarks, %d axes" % [
		texts.size(), layout["n_landmarks"], layout["n_axes"]])
	trails_stale.emit()

func begin(view_sig: String) -> void:
	## A fresh interrogation: drop the old markers and remember which view
	## the new ones are queried in.
	clear_ghosts()
	view = view_sig


func follow_layout(view_sig: String) -> void:
	## After a layout refresh. Markers from another view are wrong, so they
	## go; in the same view they follow a rescale.
	if ghosts == null:
		return
	if view != view_sig:
		clear_ghosts()
		return
	for g in ghosts.get_children():
		g.position = (g.get_meta("lcoords") as Vector3) * ctx.world_scale


func clear_ghosts() -> void:
	if ghosts != null:
		ghosts.queue_free()
		ghosts = null

