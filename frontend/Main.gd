extends Node3D

## MVP frontend for the tokencollider sidecar. Type a word and press Enter to add a
## landmark. Fly with WASD + right-mouse look. C drops the cursor ahead of the
## camera, I interrogates it, X exports the conditioning tensor at it, R
## refreshes the layout.
##
## Viewport position shows layout dims 1-3 (uniformly scaled — shape-preserving);
## dims 4-6 arrive pre-collapsed into each landmark's color by the sidecar.
## The cursor's color dims come from the color picker: pick a color, and the
## inverse of the sidecar's Oklab mapping turns it into z-scores on axes 4-6 —
## matching a landmark's ball color steers toward it in those dimensions.

## The on-screen key help. The panel is a fixed height and clips silently, so
## the line count matters: see where it is applied in _ready. M must stay
## listed, since the menu is the only way to Credits and Godot's licence notice.
const KEY_HELP := "[b]TokenCollider[/b]\nTab: word box, Enter: back to flight\nF: search + fly to a landmark\nWASD + right-mouse: fly (Q/E, Shift)\nmousewheel: fly speed\nleft/right: step one layer\nbottom sliders: layer range\nC: drop cursor    I: interrogate\nShift+I: realize ghosts as landmarks\nG: colonize   L: world/local axes\nO: centre cursor   @path / drop: images\nX: export stack   Shift+X: single\nCtrl+X: full sweep\nleft-drag: box select   Del: delete\nT: trails  N: norm/abs  R: refresh\nU: snapshot   V: release GPU\ncolor swatch: cursor dims 4-6\nEsc: clear  M: menu  Ctrl+Q: quit"

const Matcher := preload("res://lib/Matcher.gd")
const Oklab := preload("res://lib/Oklab.gd")
const ShapeAtlas := preload("res://lib/ShapeAtlas.gd")
const SidecarClient := preload("res://lib/Sidecar.gd")
const RequestGate := preload("res://lib/RequestGate.gd")
const Config := preload("res://lib/Config.gd")



var landmarks := {}  # text -> Node3D
var cursor: Node3D = null
var cursor_layout_pos := Vector3.ZERO  # layout-space coords captured at drop
var cursor_view := ""  # view signature at drop; a cursor is only valid in it
var view_restored := false  # session view state applied at most once
var selected := {}  # text -> true; box-selected landmarks (drawn white)
var landmark_colors := {}  # text -> semantic color, for un-tinting
var select_rect: ColorRect = null  # rubber band while box-dragging
var drag_start := Vector2.ZERO
var box_dragging := false
var color_timer: Timer = null  # debounce: re-solve masses after recoloring
var sampler_layer := -1  # cook stop: the deepest sampler tap; a view topping out below it cooks on export
var scrubbing := false
var has_world := false  # the sidecar knows an atlas this neighborhood was carved from
var last_center = null  # this neighborhood's centre in the current axes: {coords, color}
var cursor_at_center := false  # the cursor IS the landmark mean (O), not a lifted 6D point
# Normalized by default: dense universes open up instead of funneling out of
# the origin. N toggles to absolute; restored sessions bring their own mode.
var view_state := preload("res://lib/ViewState.gd").new()  # layer band, axes, scaling

@onready var word_entry: LineEdit = $UI/WordEntry
@onready var info: RichTextLabel = $UI/Info
@onready var status_label: Label = $UI/Status
@onready var camera: Camera3D = $Camera
@onready var layer_slider: HSlider = $UI/LayerSlider
@onready var layer_slider_lo: HSlider = $UI/LayerSliderLo
@onready var layer_label: Label = $UI/LayerLabel
@onready var color_picker: ColorPickerButton = $UI/ColorPicker
@onready var fly_entry: LineEdit = $UI/FlyEntry
@onready var fly_matches: ItemList = $UI/FlyMatches

func _ready() -> void:
	settings = Config.new()
	settings.load_settings()
	sidecar = SidecarClient.new()
	sidecar.name = "Sidecar"
	factory = preload("res://viewport/LandmarkFactory.gd").new()
	factory.name = "LandmarkFactory"
	factory.shapes = shapes
	factory.base_url_provider = func() -> String: return sidecar.base_url()
	factory.headers_provider = func() -> PackedStringArray: return sidecar.auth_headers()
	add_child(factory)

	# The shared seam. Components read state from it and emit intent back;
	# none of them reaches into this scene, which is what lets any of them be
	# moved, tested, or replaced on its own.
	ctx = preload("res://viewport/Context.gd").new()
	ctx.view = view_state
	ctx.landmarks = landmarks
	ctx.api = func(path: String, body = null): return await sidecar.request(path, body)
	ctx.report = func(what: String) -> void: status_label.text = what

	trails = preload("res://viewport/Constellations.gd").new()
	trails.name = "Constellations"
	trails.ctx = ctx
	trails.needs_refresh.connect(_refresh)
	add_child(trails)

	ghost_layer = preload("res://viewport/Ghosts.gd").new()
	ghost_layer.name = "Ghosts"
	ghost_layer.ctx = ctx
	ghost_layer.applied_layout.connect(_apply_layout)
	ghost_layer.trails_stale.connect(func(): trails.mark_stale())
	add_child(ghost_layer)

	menu = preload("res://ui/MenuOverlay.gd").new()
	menu.name = "MenuOverlay"
	menu.settings = settings
	menu.quit_requested.connect(func(): get_tree().quit())
	# The viewport takes its mouse back when the menu closes; the menu never
	# reaches in to do that itself.
	menu.resumed.connect(func(): status_label.text = "resumed")
	add_child(menu)

	mass = preload("res://viewport/MassView.gd").new()
	mass.name = "MassView"
	add_child(mass)
	# The client announces failures; this screen decides they belong in the
	# status line. A different screen can decide differently.
	sidecar.request_failed.connect(func(what: String): status_label.text = what)
	add_child(sidecar)

	# Saved settings first, then whatever `tokencollider view` says it actually
	# started, which wins (see Sidecar.apply_launch).
	sidecar.host = settings.get_value("sidecar/host")
	sidecar.port = int(settings.get_value("sidecar/port"))
	if OS.has_feature("web"):
		sidecar.apply_launch({}, str(JavaScriptBridge.eval("window.location.href", true)))
	else:
		sidecar.apply_launch({
			"TOKENCOLLIDER_PORT": OS.get_environment("TOKENCOLLIDER_PORT"),
			"TOKENCOLLIDER_TOKEN": OS.get_environment("TOKENCOLLIDER_TOKEN"),
		})
	word_entry.text_submitted.connect(_on_word_submitted)
	layer_slider.value_changed.connect(_on_layer_hi_changed)
	layer_slider_lo.value_changed.connect(_on_layer_lo_changed)
	color_picker.color_changed.connect(_on_cursor_color_changed)
	fly_entry.text_changed.connect(_on_fly_text_changed)
	fly_entry.text_submitted.connect(_on_fly_submitted)
	fly_matches.item_selected.connect(_on_fly_match_picked)
	get_window().files_dropped.connect(_on_files_dropped)
	select_rect = ColorRect.new()
	select_rect.color = Color(1, 1, 1, 0.10)
	select_rect.visible = false
	select_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	$UI.add_child(select_rect)
	color_timer = Timer.new()
	color_timer.one_shot = true
	color_timer.wait_time = 0.35
	color_timer.timeout.connect(func():
		if cursor != null and cursor_view == _view_signature():
			_update_mass_view())
	add_child(color_timer)
	# The panel is a fixed 336px tall with scroll_active off, so anything past
	# its height is clipped silently, not scrolled to. The old list ran 22
	# lines and lost everything from V down. Keep it short and keep the font
	# small enough that this fits with room to spare.
	info.add_theme_font_size_override("normal_font_size", 13)
	info.add_theme_font_size_override("bold_font_size", 13)
	info.text = KEY_HELP
	_refresh()

# --- sidecar API ---------------------------------------------------------

var sidecar: Node = null      # lib/Sidecar.gd, created in _ready
var settings: RefCounted = null  # lib/Config.gd, durable user settings

func _api(path: String, body = null) -> Variant:
	## Thin pass-through so call sites read the same as before. All transport,
	## URL building and error classification is in lib/Sidecar.gd; this only
	## decides that a failure shows up in the status label.
	return await sidecar.request(path, body)

func _layout_path() -> String:
	var path := "/layout?axes=" + _axes_value()
	if view_state.layer_hi >= 0:
		path += "&layer=" + _layer_param()
	return path

func _refresh() -> void:
	var layout = await _api(_layout_path())
	if layout != null:
		_apply_layout(layout)

func _on_files_dropped(files: PackedStringArray) -> void:
	## Pictures (or folders of them) dropped on the window become image
	## landmarks, the same as typing "@path" in the word box.
	if OS.has_feature("web"):
		# The browser hands the page the file's bytes, never its path, and
		# the sidecar can only open paths. Say what works instead.
		status_label.text = "Dropping files only works in the desktop viewport. " \
			+ "Press Tab and type @ and the image or folder path, then Enter."
		return
	var paths: Array = []
	for f in files:
		paths.append(f)
	status_label.text = "embedding %d dropped file(s) through the vision tower ..." % paths.size()
	var body := {"images": paths}
	if view_state.layer_hi >= 0:
		body["layer"] = _layer_body_value()
	body["axes"] = _axes_value()
	var layout = await _api("/landmarks", body)
	if layout != null:
		_apply_layout(layout)
		status_label.text = "%d landmarks, %d axes" % [layout["n_landmarks"], layout["n_axes"]]

func _on_word_submitted(text: String) -> void:
	# Enter always hands the keyboard back to flight (Tab re-focuses the box),
	# so entry stays a quick detour: Tab, type, Enter, fly. A leading "@" is
	# an image path or a folder of images.
	word_entry.release_focus()
	text = text.strip_edges()
	if text.is_empty():
		return
	word_entry.clear()
	status_label.text = ("embedding '%s' ..." % text) if not text.begins_with("@") \
		else "embedding %s through the vision tower ..." % text
	var body := {"text": text}
	if view_state.layer_hi >= 0:
		body["layer"] = _layer_body_value()
	body["axes"] = _axes_value()
	var layout = await _api("/landmarks", body)
	if layout != null:
		_apply_layout(layout)
		status_label.text = "%d landmarks, %d axes" % [layout["n_landmarks"], layout["n_axes"]]
		_constellation_stale()

# --- fly-to: autocomplete over loaded landmarks ---------------------------

func _on_fly_text_changed(text: String) -> void:
	## Ranking lives in lib/Matcher.gd; this only puts the result on screen.
	fly_matches.clear()
	for t in Matcher.rank(text, landmarks):
		fly_matches.add_item(t)
	fly_matches.visible = fly_matches.item_count > 0

func _on_fly_submitted(text: String) -> void:
	var target := text.strip_edges()
	if fly_matches.item_count > 0:
		target = fly_matches.get_item_text(0)
	_finish_fly_search()
	_fly_to(target)

func _on_fly_match_picked(idx: int) -> void:
	var target := fly_matches.get_item_text(idx)
	_finish_fly_search()
	_fly_to(target)

func _finish_fly_search() -> void:
	fly_entry.clear()
	fly_matches.clear()
	fly_matches.hide()
	fly_entry.release_focus()

func _fly_to(text: String) -> void:
	if not landmarks.has(text):
		status_label.text = "no landmark '%s' in this universe" % text
		return
	var node: Node3D = landmarks[text]
	# Approach from the current direction, stopping a few units short.
	var away := camera.global_position - node.position
	if away.length() < 0.01:
		away = Vector3(0, 0, 1)
	var to_pos: Vector3 = node.position + away.normalized() * 4.0
	var look := Transform3D(Basis.looking_at(node.position - to_pos, Vector3.UP), to_pos)
	var target_rot := look.basis.get_euler()
	# Take the short way around on yaw; keep roll locked like FlyCamera does.
	target_rot.y = camera.rotation.y + wrapf(target_rot.y - camera.rotation.y, -PI, PI)
	var tw := create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	tw.tween_property(camera, "global_position", to_pos, 0.7)
	tw.parallel().tween_property(camera, "rotation",
		Vector3(target_rot.x, target_rot.y, 0.0), 0.7)
	status_label.text = "flying to '%s'" % text

# --- layer scrubbing ------------------------------------------------------

func _layer_param() -> String:
	return view_state.layer_param()

func _axes_value() -> String:
	return "world" if view_state.world_axes else "local"

func _view_signature() -> String:
	# The identity of the space the user is looking at. A cursor dropped in
	# one space is meaningless in another: layer bases are different
	# subspaces, so reprojection would be silently wrong, not approximate.
	# World and local axes are different charts of the same depth, so they
	# are part of the identity too.
	return view_state.signature()

func _toggle_axes() -> void:
	## World axes = the parent atlas's chart; local = this neighborhood's own
	## PCA. Same depth, same landmarks, two projections. At the atlas itself
	## they coincide, so there is nothing to toggle.
	if not has_world:
		status_label.text = "this is the atlas: local and world axes coincide (G to carve a neighborhood first)"
		return
	view_state.world_axes = not view_state.world_axes
	_update_layer_label()
	status_label.text = ("world axes: the atlas's chart, so positions compare across neighborhoods and models"
		if view_state.world_axes else "local axes: this neighborhood's own principal directions")
	_refresh()

func _layer_body_value() -> Variant:
	return view_state.layer_hi if view_state.layer_lo == view_state.layer_hi else [view_state.layer_lo, view_state.layer_hi]

func _update_layer_label() -> void:
	var top := int(layer_slider.max_value)
	if view_state.layer_lo == view_state.layer_hi:
		layer_label.text = "layer %d / %d" % [view_state.layer_hi, top]
	else:
		layer_label.text = "layers %d-%d averaged / %d" % [view_state.layer_lo, view_state.layer_hi, top]
	layer_label.text += "  ·  norm" if view_state.normalized else "  ·  abs"
	if has_world:
		layer_label.text += "  ·  world axes" if view_state.world_axes else "  ·  local axes"
	if sampler_layer >= 0 and view_state.layer_hi < sampler_layer:
		layer_label.text += "  ·  cooks on export"

func _on_layer_hi_changed(value: float) -> void:
	var v := int(value)
	if v < view_state.layer_lo:
		layer_slider_lo.set_value_no_signal(v)
		view_state.layer_lo = v
	if v == view_state.layer_hi and int(layer_slider_lo.value) == view_state.layer_lo:
		return
	view_state.layer_hi = v
	_update_layer_label()
	_scrub()

func _on_layer_lo_changed(value: float) -> void:
	var v := int(value)
	if v > view_state.layer_hi:
		layer_slider.set_value_no_signal(v)
		view_state.layer_hi = v
	if v == view_state.layer_lo:
		return
	view_state.layer_lo = v
	_update_layer_label()
	_scrub()

func _step_layer(delta: int) -> void:
	## Walk the whole selection one depth at a time (a range keeps its width).
	if view_state.layer_hi < 0:
		return
	if view_state.layer_lo + delta < int(layer_slider.min_value) or view_state.layer_hi + delta > int(layer_slider.max_value):
		return
	view_state.layer_lo += delta
	view_state.layer_hi += delta
	layer_slider_lo.set_value_no_signal(view_state.layer_lo)
	layer_slider.set_value_no_signal(view_state.layer_hi)
	_update_layer_label()
	_scrub()

func _toggle_normalized() -> void:
	view_state.normalized = not view_state.normalized
	_update_layer_label()
	status_label.text = "depth view: %s" % (
		"normalized — each layer scaled by its own spread" if view_state.normalized
		else "absolute — raw magnitude across depth")
	if trails.is_showing():
		await trails.draw()
	else:
		_refresh()

func _scrub() -> void:
	if scrubbing:
		return  # the loop below will catch up to the latest range
	scrubbing = true
	var fetched := ""
	while fetched != _layer_param():
		fetched = _layer_param()
		status_label.text = "fetching layers %s ... (first visit may warm the cache — progress in the terminal)" % fetched
		var layout = await _api(_layout_path())
		if layout == null:
			break
		_apply_layout(layout)
		status_label.text = "layers %s ready" % fetched
	scrubbing = false

func _configure_slider(layout: Dictionary, view = null) -> void:
	if layer_slider.visible or layout.get("n_layers") == null:
		return
	var n := int(layout["n_layers"])
	# The profile's layer_min/layer_max confine scrubbing to the band where meaning lives —
	# layers outside it aren't hidden data, just hidden controls.
	var lo_bound := 0
	var hi_bound := n - 1
	if layout.get("layer_min") != null:
		lo_bound = clampi(int(layout["layer_min"]), 0, n - 1)
	if layout.get("layer_max") != null:
		hi_bound = clampi(int(layout["layer_max"]), lo_bound, n - 1)
	var start := hi_bound
	var start_lo := -1
	if layout.get("layer") != null:
		if layout["layer"] is Array:  # band-configured embedder (layer: lo-hi)
			start_lo = clampi(int(layout["layer"][0]), lo_bound, hi_bound)
			start = int(layout["layer"][1])
		else:
			start = int(layout["layer"])
	sampler_layer = start  # charted depth; the sidecar's cook_stop overrides it
	if layout.get("cook_stop") != null:
		# Where a cooked export stops. Z-Image charts at its sampler depth,
		# so the two agree; Krea 2 charts at 20 and cooks to 35, so they do
		# not, and keying the label on the chart hid that every export cooked.
		sampler_layer = int(layout["cook_stop"])
	# Open on the profile's charting layer, never on the whole bounded band.
	# The interrogator and G read the cache slice of the viewed selection,
	# and `warm` writes the profile's layer, so any other opening view has
	# no vocabulary in it. layer_min/layer_max only bound the sliders.
	start = clampi(start, lo_bound, hi_bound)
	view_state.layer_hi = start
	view_state.layer_lo = start_lo if start_lo >= 0 else start
	if view != null and view.has("layer_lo") and view.has("layer_hi"):
		# A restored session's selection beats every default.
		view_state.layer_lo = clampi(int(view["layer_lo"]), lo_bound, hi_bound)
		view_state.layer_hi = clampi(int(view["layer_hi"]), view_state.layer_lo, hi_bound)
	# Seed values BEFORE tightening bounds: raising min_value above the
	# current value clamps it and emits value_changed, which would fire a
	# scrub while layer_lo/hi are still the -1 sentinels.
	layer_slider.set_value_no_signal(view_state.layer_hi)
	layer_slider_lo.set_value_no_signal(view_state.layer_lo)
	layer_slider.min_value = lo_bound
	layer_slider.max_value = hi_bound
	layer_slider_lo.min_value = lo_bound
	layer_slider_lo.max_value = hi_bound
	_update_layer_label()
	layer_slider.visible = true
	layer_slider_lo.visible = true
	layer_label.visible = true
	# If the seeded selection isn't the view this layout was rendered in,
	# fetch the matching one so sliders and world never disagree.
	var shown = layout.get("layer")
	var matches: bool = (shown is Array and int(shown[0]) == view_state.layer_lo and int(shown[1]) == view_state.layer_hi) \
		or (shown != null and not shown is Array and view_state.layer_lo == view_state.layer_hi and int(shown) == view_state.layer_hi)
	if not matches:
		_scrub()

# --- layout -> world -----------------------------------------------------

func _apply_layout(layout: Dictionary) -> void:
	# A reply for a view the user has since left (a slow embed landing after a
	# fast scrub, say) is dropped, and the view on screen is fetched instead.
	# The one exception is the reply that restores a saved session's view.
	var restoring: bool = not view_restored and layout.get("view") != null
	if not restoring and not view_state.matches_layout(layout):
		stale_replies += 1
		if stale_replies <= 3:
			_refresh()
			return
		# The sidecar keeps answering a different view than asked. Take what
		# it says rather than refetch forever.
		push_warning("layout reply never matched the view; applying it anyway")
	stale_replies = 0
	var restored_view = null
	if not view_restored and layout.get("view") != null:
		# A saved session rides in the universe header: camera, band,
		# normalized mode, swatch, cursor — applied exactly once, on load.
		view_restored = true
		restored_view = layout["view"]
		_apply_view_state(restored_view)
	_configure_slider(layout, restored_view)
	has_world = bool(layout.get("has_world", false))
	last_center = layout.get("center")
	if view_state.world_axes and str(layout.get("axes", "local")) != "world":
		# Asked for world axes, got local: there is no atlas here. Fall back
		# before the view signature is consulted, so the cursor survives.
		view_state.world_axes = false
	_update_layer_label()
	if view_state.normalized:
		# Each depth fills the same volume: shape change without magnitude.
		var rms: float = ctx.rms_spread(layout["landmarks"])
		ctx.world_scale = ctx.WORLD_TARGET / rms if rms > 0.0 else 1.0
	elif ctx.locked_scale > 0.0:
		# Depth-stable scale from the trajectory extent: spheres land exactly
		# on their constellation lines and scrubbing never rescales the world.
		ctx.world_scale = ctx.locked_scale
	else:
		var max_extent := 0.0
		for entry in layout["landmarks"]:
			for i in 3:
				max_extent = maxf(max_extent, absf(entry["coords"][i]))
		ctx.world_scale = ctx.WORLD_TARGET / max_extent if max_extent > 0.0 else 1.0

	if cursor != null:
		if cursor_view != _view_signature():
			# The world under the cursor changed spaces; the point it named
			# no longer exists here.
			_clear_cursor()
			status_label.text = "cursor cleared — view changed (C to re-drop)"
		else:
			# Same view, possibly rescaled (e.g. constellation locked the
			# scale): keep the cursor on its captured layout point.
			cursor.position = cursor_layout_pos * ctx.world_scale
	ghost_layer.follow_layout(_view_signature())
	mass.clear_tethers()  # landmarks are about to move; masses stay

	var seen := {}
	for entry in layout["landmarks"]:
		seen[entry["text"]] = true
		var src: String = entry.get("source", "preload")
		var dens: int = int(entry["density"]) if entry.get("density") != null else -1
		var fresh: bool = not landmarks.has(entry["text"])
		if not fresh and str(landmarks[entry["text"]].get_meta("source", "")) != src:
			landmarks[entry["text"]].queue_free()
			landmarks.erase(entry["text"])
			fresh = true
		if fresh:
			landmarks[entry["text"]] = factory.make_landmark(entry["text"], src, dens, entry)
		var node: Node3D = landmarks[entry["text"]]
		if int(node.get_meta("density", -1)) != dens:
			factory.set_density(node, dens)
		# Annotated because ctx is typed RefCounted, so ctx.world_scale is a
		# Variant and := cannot infer through it.
		var target: Vector3 = Vector3(entry["coords"][0], entry["coords"][1], entry["coords"][2]) * ctx.world_scale
		if fresh:
			node.position = target
		else:
			_glide(node, target)
		var color := Color.html(entry["color"])
		landmark_colors[entry["text"]] = color
		if selected.has(entry["text"]):
			color = Color.WHITE
		if node.get_meta("image", false):
			# A picture keeps its own colours; its semantic colour goes to
			# the tag, and selection brightens the picture instead of
			# painting it.
			(node.get_node("Shape") as MeshInstance3D).material_override.albedo_color = \
				Color(1.4, 1.4, 1.4) if selected.has(entry["text"]) else Color.WHITE
		else:
			(node.get_node("Shape") as MeshInstance3D).material_override.albedo_color = color
		(node.get_node("Tag") as Label3D).modulate = color.lightened(0.4)
	for text in landmarks.keys():
		if not seen.has(text):
			landmarks[text].queue_free()
			landmarks.erase(text)
			landmark_colors.erase(text)
			selected.erase(text)
	if restored_view != null:
		_restore_cursor(restored_view)
	print("[frontend] layout applied: %d landmarks, scale %.3f" % [landmarks.size(), ctx.world_scale])

# --- constellations: delegated to viewport/Constellations.gd --------------

func _toggle_constellation() -> void:
	await trails.toggle()

func _constellation_stale() -> void:
	trails.mark_stale()

func _glide(node: Node3D, target: Vector3) -> void:
	if node.has_meta("tween"):
		var old: Tween = node.get_meta("tween")
		if old.is_valid():
			old.kill()
	var tw := create_tween()
	tw.tween_property(node, "position", target, 0.35).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_OUT)
	node.set_meta("tween", tw)


var shapes := ShapeAtlas.new()   # lib/ShapeAtlas.gd, sprite textures
var factory: Node = null         # viewport/LandmarkFactory.gd, world objects
var ctx: RefCounted = null       # viewport/Context.gd, the shared seam
var trails: Node3D = null        # viewport/Constellations.gd
var ghost_layer: Node3D = null   # viewport/Ghosts.gd
var mass: Node3D = null          # viewport/MassView.gd
var menu: CanvasLayer = null     # ui/MenuOverlay.gd, the front door
var stale_replies := 0           # layout replies dropped in a row (see _apply_layout)
var interrogate_gate = RequestGate.new()  # drops /interrogate replies for a view since left
var blend_gate = RequestGate.new()        # drops /blend replies for a cursor since moved

# --- color axes ----------------------------------------------------------
# The maths lives in lib/Oklab.gd, verified against tokencollider/oklab.py. The picker's
# default swatch in Main.tscn is the z=0 neutral.

func _on_cursor_color_changed(c: Color) -> void:
	if cursor != null:
		(cursor.get_node("Shape") as MeshInstance3D).material_override.albedo_color = c
		color_timer.start()  # debounced re-solve: masses/tethers follow color
	var z := Oklab.color_to_zscores(c)
	status_label.text = "cursor color -> dims 4-6 z: (%.2f, %.2f, %.2f)" % [z.x, z.y, z.z]

# --- cursor: select, interrogate, export ---------------------------------

# --- box select: left-drag a region, Delete removes it, Esc clears --------

func _refresh_tint(text: String) -> void:
	if not landmarks.has(text):
		return
	var node: Node3D = landmarks[text]
	var color: Color = Color.WHITE if selected.has(text) else landmark_colors.get(text, Color.WHITE)
	(node.get_node("Shape") as MeshInstance3D).material_override.albedo_color = color
	(node.get_node("Tag") as Label3D).modulate = color.lightened(0.4)

func _clear_selection() -> void:
	var had := selected.keys()
	selected.clear()
	for text in had:
		_refresh_tint(text)

func _finish_box_select(r: Rect2) -> void:
	if r.size.length() < 8.0:
		# A plain click on empty space: deselect, like every editor.
		if selected.size() > 0:
			_clear_selection()
			status_label.text = "selection cleared"
		return
	_clear_selection()
	for text in landmarks:
		var node: Node3D = landmarks[text]
		if camera.is_position_behind(node.global_position):
			continue
		if r.has_point(camera.unproject_position(node.global_position)):
			selected[text] = true
			_refresh_tint(text)
	if selected.size() > 0:
		status_label.text = "%d selected — Delete removes them, Esc clears" % selected.size()
	else:
		status_label.text = "nothing in the box"

func _delete_selected() -> void:
	var texts := selected.keys()
	status_label.text = "removing %d landmarks ..." % texts.size()
	var body := {"texts": texts}
	if view_state.layer_hi >= 0:
		body["layer"] = _layer_body_value()
	body["axes"] = _axes_value()
	var layout = await _api("/landmarks/remove", body)
	if layout == null:
		return
	_clear_selection()
	_apply_layout(layout)
	status_label.text = "removed %d — %d landmarks left" % [layout["removed"], layout["n_landmarks"]]
	_constellation_stale()
	if cursor != null:
		_update_mass_view()  # the blend just lost a whole region

func _shortcut_input(event: InputEvent) -> void:
	# Quit lives here, not in _unhandled_input: shortcuts are dispatched
	# before the focused control eats the key, so Ctrl+Q works mid-word too.
	# `tokencollider view` takes the sidecar down when the viewport closes.
	if event is InputEventKey and event.pressed and not event.echo \
			and event.keycode == KEY_Q and event.ctrl_pressed:
		get_viewport().set_input_as_handled()
		get_tree().quit()


func _unhandled_input(event: InputEvent) -> void:
	# While the menu is up it owns the keyboard. Without this, R would refresh
	# and X would export behind an open dialog.
	if menu != null and menu.is_open():
		if event is InputEventKey and event.pressed and event.keycode == KEY_M:
			menu.close()
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT \
			and Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
		if event.pressed:
			drag_start = event.position
			box_dragging = true
			select_rect.visible = false
		elif box_dragging:
			box_dragging = false
			select_rect.visible = false
			_finish_box_select(Rect2(drag_start, event.position - drag_start).abs())
		return
	if event is InputEventMouseMotion and box_dragging:
		var r := Rect2(drag_start, event.position - drag_start).abs()
		select_rect.position = r.position
		select_rect.size = r.size
		select_rect.visible = r.size.length() > 6.0
		return
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_TAB: word_entry.grab_focus()
			KEY_F: fly_entry.grab_focus()
			KEY_C: _drop_cursor()
			KEY_I:
				# I asks what is nearby, Shift+I keeps the answer.
				if event.shift_pressed:
					_realize_ghosts()
				else:
					_interrogate()
			KEY_X:
				# Stack is the default: the single-depth export is the
				# shallowest paraphrase and historically underwhelmed.
				if event.ctrl_pressed:
					_export_sweep()
				elif event.shift_pressed:
					_export()
				else:
					_export_stack()
			KEY_G: _colonize()
			KEY_L: _toggle_axes()
			KEY_O: _drop_cursor_at_center()
			KEY_V: _release_gpu()
			KEY_DELETE:
				if selected.size() > 0:
					_delete_selected()
				elif Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
					# Hover needs a visible pointer; while flying there's
					# nothing under the (hidden) mouse to delete.
					_delete_hovered()
			KEY_M: menu.open()
			KEY_R: _refresh()
			KEY_T: _toggle_constellation()
			KEY_N: _toggle_normalized()
			KEY_U: _export_universe()
			KEY_LEFT: _step_layer(-1)
			KEY_RIGHT: _step_layer(1)
			KEY_ESCAPE:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
				get_viewport().gui_release_focus()
				_clear_ghosts()
				_clear_selection()
				fly_matches.hide()

func _make_cursor() -> Node3D:
	# Triangle = cursor, in the flat-sprite taxonomy. Its digit shows the
	# void census at the dropped point, on the same 0-9 scale the landmarks
	# wear (filled in when the blend solves).
	var root := Node3D.new()
	root.set_meta("show_digit", true)
	root.add_child(factory.make_node_sprite("triangle", 0.9))
	root.add_child(factory.make_digit_label())
	var tag := Label3D.new()
	tag.name = "Tag"
	tag.text = "<cursor>"
	tag.position.y = 0.7
	tag.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	tag.alpha_cut = Label3D.ALPHA_CUT_DISCARD  # opaque pass = real occlusion
	tag.font_size = 48
	tag.pixel_size = 0.008
	root.add_child(tag)
	add_child(root)
	return root

func _drop_cursor() -> void:
	if cursor == null:
		cursor = _make_cursor()
	var mat: StandardMaterial3D = cursor.get_node("Shape").material_override
	mat.albedo_color = color_picker.color
	# Exactly the camera's position: where you are IS the selection — no
	# eyeballing a projected point. (The sphere is invisible from inside;
	# back away to see it.)
	cursor.position = camera.global_position
	# Freeze the selection in layout space + view identity: scale changes in
	# the same view reposition it, view changes invalidate it.
	cursor_layout_pos = cursor.position / ctx.world_scale
	cursor_view = _view_signature()
	cursor_at_center = false
	status_label.text = "cursor at %s — I to interrogate, X to export" % cursor.position
	_update_mass_view()

func _drop_cursor_at_center() -> void:
	## The neighborhood's centre, the mean of its landmarks: the origin in
	## local axes, a specific point in world axes. The same phrases under
	## two models give two centres that name the same concept, which is the
	## one cursor a finetune and its base can be compared from.
	if last_center == null:
		status_label.text = "no centre yet (need at least two landmarks)"
		return
	var c: Array = last_center["coords"]
	if cursor == null:
		cursor = _make_cursor()
	# Dims 4-6 ride in the swatch, so set it to the centre's colour; the
	# picker's signal recolours the cursor and reports the z-scores.
	color_picker.color = Color.html(str(last_center["color"]))
	(cursor.get_node("Shape") as MeshInstance3D).material_override.albedo_color = color_picker.color
	cursor_layout_pos = Vector3(c[0], c[1], c[2])
	cursor.position = cursor_layout_pos * ctx.world_scale
	cursor_view = _view_signature()
	# The marker sits at the mean's projection into this chart; the sidecar
	# is told to use the mean itself, since in world axes the projection
	# drops the part of the mean outside the atlas's six directions.
	cursor_at_center = true
	status_label.text = "cursor at the neighborhood centre (%s axes) — I to interrogate, X to export" % _axes_value()
	_update_mass_view()

# --- stellar mass + gravity tethers: the blend made visible ----------------

func _update_mass_view() -> void:
	## Weight matters and position can't show it: the least-squares blend can
	## hand a distant hub word the whole mixture. On every drop, landmarks
	## swell by their |weight| (stellar mass) and tethers draw the pull —
	## green positive, red negative, brightness = magnitude. A fat tether to
	## something far away means a hub word has taken over the blend.
	var ticket: Dictionary = blend_gate.open(_view_signature())
	var res = await _api("/blend", _cursor_body())
	# A newer drop, or a view change, makes this reply's weights and tethers
	# describe a cursor that is no longer there.
	if res == null or cursor == null or not blend_gate.is_current(ticket, _view_signature()):
		return
	mass.show_blend(res["weights"], landmarks, cursor.position)
	if not mass.is_showing():
		return
	# The compass scalars: neg% = how much of the recipe is subtraction;
	# align ~1.00 = the red cone is only re-centering (no antonym here),
	# low align + real neg% = a specific opposition axis exists.
	var negline := ""
	if res.has("negativity"):
		var negativity: Dictionary = res["negativity"]
		negline = " | neg %d%%" % int(100.0 * float(negativity["fraction"]))
		if negativity["centroid_alignment"] != null:
			negline += ", centroid-align %.2f" % float(negativity["centroid_alignment"])
	if res.get("void") != null and res["void"].has("ratio"):
		# >1 = sparser than a typical landmark's surroundings: void country.
		negline += " | void x%.1f" % float(res["void"]["ratio"])
	if cursor != null and res.get("void") != null and res["void"].has("digit"):
		factory.set_density(cursor, int(res["void"]["digit"]))
	status_label.text = "cursor dropped — err %.3f%s — X exports" % [float(res["relative_error"]), negline]

func _clear_mass_view() -> void:
	mass.clear(landmarks)

func _clear_cursor() -> void:
	if cursor != null:
		cursor.queue_free()
		cursor = null
	_clear_mass_view()

func _cursor_valid() -> bool:
	if cursor == null:
		status_label.text = "drop the cursor first (C)"
		return false
	if cursor_view != _view_signature():
		_clear_cursor()
		status_label.text = "cursor was dropped in another view — re-drop (C)"
		return false
	return true

func _cursor_coords() -> Array:
	# Spatial dims: the drop-time capture, not a live reprojection — scale
	# drift between drop and use can never skew the selection. Color dims:
	# read live from the picker, so recoloring after a drop needs no re-drop.
	var z: Vector3 = Oklab.color_to_zscores(color_picker.color)
	return [cursor_layout_pos.x, cursor_layout_pos.y, cursor_layout_pos.z, z.x, z.y, z.z]

func _cursor_body(extra: Dictionary = {}) -> Dictionary:
	var body := {"coords": _cursor_coords()}
	if cursor_at_center:
		body["center"] = true
	if view_state.layer_hi >= 0:
		body["layer"] = _layer_body_value()
	body["axes"] = _axes_value()
	body.merge(extra)
	return body

func _interrogate() -> void:
	## Nearest landmarks land in the status bar (and the sidecar's terminal);
	## nearest cached-vocab phrases materialize as ghost cubes IN the world —
	## what's actually nearby, shown where it actually is.
	if not _cursor_valid():
		return
	status_label.text = "interrogating ..."
	var ticket: Dictionary = interrogate_gate.open(_view_signature())
	var res = await _api("/interrogate", _cursor_body({"k": 12}))
	if res == null:
		return
	if not interrogate_gate.is_current(ticket, _view_signature()):
		# Scrubbed, switched axes or asked again while this was out: its
		# coordinates belong to a chart that is no longer on screen.
		if str(ticket["view"]) != _view_signature():
			status_label.text = "view changed during interrogation (I to ask again)"
		return
	ghost_layer.begin(str(ticket["view"]))
	var top := "(no landmarks)"
	if res["nearest_landmarks"].size() > 0:
		var n0 = res["nearest_landmarks"][0]
		top = "%s %.3f" % [n0["text"], n0["cosine"]]
	if res.has("nearest_vocab") and res["nearest_vocab"].size() > 0:
		for n in res["nearest_vocab"]:
			_make_ghost(n["text"], n["coords"], float(n["cosine"]))
		status_label.text = "nearest: %s — %d vocab ghosts placed (Esc clears)" % [
			top, res["nearest_vocab"].size()]
	else:
		status_label.text = "nearest: %s — no cached vocab under this view (warm a wordlist)" % top

# --- ghosts: delegated to viewport/Ghosts.gd ------------------------------

func _make_ghost(text: String, coords: Array, cosine: float) -> void:
	ghost_layer.make_ghost(text, coords, cosine)

func _realize_ghosts() -> void:
	await ghost_layer.realize()

func _clear_ghosts() -> void:
	ghost_layer.clear_ghosts()

func _colonize() -> void:
	## Grab the neighborhood around the cursor: the sidecar gathers the nearest
	## cached-vocabulary phrases plus every manual landmark into a child
	## universe (parent fingerprint + chart recorded in its header) and swaps
	## the live world to it.
	if not _cursor_valid():
		return
	status_label.text = "colonizing neighborhood ..."
	var res = await _api("/colonize", _cursor_body())
	if res == null:
		return
	# Arrive in world axes. The cursor was dropped either in the atlas's own
	# chart (which IS the world now) or already in world axes, and either way
	# its coordinates still name the same point, so it stays as the
	# beachhead. Only a cursor dropped in a nested neighborhood's local axes
	# has no meaning in the world and is cleared.
	var carried: bool = (not has_world) or view_state.world_axes
	view_state.world_axes = true
	if carried:
		cursor_view = _view_signature()
	else:
		_clear_cursor()
	_clear_ghosts()
	# Drop trails outright rather than redrawing: a redraw would warm every
	# native at every depth, and the old universe's trails are moot anyway.
	trails.clear()
	if res.get("layout") != null:
		_apply_layout(res["layout"])
	if cursor != null:
		_update_mass_view()
	status_label.text = "colonized: %d natives + %d settlements -> %s  (world axes; L toggles, O centres the cursor)" % [
		res["natives"].size(), res["manual"].size(), res["path"]]

const DELETE_HIT_RADIUS_PX := 40.0

func _delete_hovered() -> void:
	## Remove the landmark under the mouse pointer (screen-space nearest
	## within DELETE_HIT_RADIUS_PX — no colliders needed).
	var pick := get_viewport().get_mouse_position()
	var best_text := ""
	var best_d := DELETE_HIT_RADIUS_PX
	for text in landmarks:
		var node: Node3D = landmarks[text]
		if camera.is_position_behind(node.global_position):
			continue
		var d := camera.unproject_position(node.global_position).distance_to(pick)
		if d < best_d:
			best_d = d
			best_text = text
	if best_text == "":
		status_label.text = "no landmark under the pointer — hover a node, then Delete"
		return
	status_label.text = "removing '%s' ..." % best_text
	var body := {"text": best_text}
	if view_state.layer_hi >= 0:
		body["layer"] = _layer_body_value()
	body["axes"] = _axes_value()
	var layout = await _api("/landmarks/remove", body)
	if layout != null:
		_apply_layout(layout)
		status_label.text = "removed '%s' — %d landmarks left" % [best_text, layout["n_landmarks"]]
		_constellation_stale()
		if cursor != null:
			_update_mass_view()  # the blend just lost a competitor

func _view_state() -> Dictionary:
	## Presentation state for session snapshots: everything ephemeral that a
	## restart would otherwise lose. Geometry state is already durable
	## (universe file + embedding cache).
	var v := {
		"camera_pos": [camera.global_position.x, camera.global_position.y, camera.global_position.z],
		"camera_rot": [camera.rotation.x, camera.rotation.y],
		"layer_lo": view_state.layer_lo,
		"layer_hi": view_state.layer_hi,
		"normalized": view_state.normalized,
		"world_axes": view_state.world_axes,
		"color": color_picker.color.to_html(false),
	}
	if cursor != null and cursor_view == _view_signature():
		v["cursor"] = [cursor_layout_pos.x, cursor_layout_pos.y, cursor_layout_pos.z]
		v["cursor_at_center"] = cursor_at_center
	return v

func _apply_view_state(v: Dictionary) -> void:
	view_state.normalized = bool(v.get("normalized", view_state.normalized))
	view_state.world_axes = bool(v.get("world_axes", view_state.world_axes))
	if v.has("color"):
		color_picker.color = Color.html(str(v["color"]))
	if v.has("camera_pos"):
		var p: Array = v["camera_pos"]
		camera.global_position = Vector3(p[0], p[1], p[2])
	if v.has("camera_rot"):
		var r: Array = v["camera_rot"]
		camera.rotation = Vector3(r[0], r[1], 0.0)

func _restore_cursor(v: Dictionary) -> void:
	if not v.has("cursor"):
		return
	var c: Array = v["cursor"]
	if cursor == null:
		cursor = _make_cursor()
	cursor_at_center = bool(v.get("cursor_at_center", false))
	cursor_layout_pos = Vector3(c[0], c[1], c[2])
	cursor.position = cursor_layout_pos * ctx.world_scale
	cursor_view = _view_signature()
	(cursor.get_node("Shape") as MeshInstance3D).material_override.albedo_color = color_picker.color
	_update_mass_view()

func _release_gpu() -> void:
	status_label.text = "releasing GPU ..."
	var res = await _api("/gpu/release", {})
	if res == null:
		return
	if res.get("released", false):
		status_label.text = "GPU released — hand it to ComfyUI; cached ops keep working, new embeds/cooks reload the model"
	else:
		status_label.text = "GPU was already free (model not loaded)"

func _export_universe() -> void:
	var res = await _api("/export_universe", {"view": _view_state()})
	if res != null:
		status_label.text = "session saved: %d phrases + view -> %s" % [res["n_phrases"], res["path"]]

func _export_stack() -> void:
	## Same cursor, same solved blend recipe, several cook depths across the
	## band: deepest injection = most reinterpreted (schedule it on early
	## sigmas), shallowest = most literal (late sigmas).
	if not _cursor_valid():
		return
	status_label.text = "exporting cook-depth stack ... (runs the model once per depth)"
	var res = await _api("/export", _cursor_body({"stack": true}))
	if res == null:
		return
	var depths := ""
	for d in res["depths"]:
		depths += ("" if depths.is_empty() else ", ") + str(d)
	status_label.text = "exported %d-tensor stack (cook depths %s) -> exports/" % [
		res["stack"].size(), depths]

func _export_sweep() -> void:
	## The synthetic walk: one cook from EVERY depth up to the sampler layer,
	## same cursor, same recipe. Render the files with a fixed seed to map
	## where the lift actually lives instead of trusting the band by feel.
	if not _cursor_valid():
		return
	status_label.text = "exporting FULL depth sweep ... (one cook per layer — takes minutes, progress in terminal)"
	var res = await _api("/export", _cursor_body({"stack": true, "sweep": true}))
	if res == null:
		return
	status_label.text = "exported %d-depth sweep -> exports/ — render with a fixed seed to map the lift" % res["stack"].size()

func _export() -> void:
	if not _cursor_valid():
		return
	status_label.text = "exporting ... (cooked bands run the model — watch the sidecar terminal)"
	var res = await _api("/export", _cursor_body())
	if res != null:
		var how: String = "cooked from band %s" % str(res["band"]) if res.get("cooked", false) else "sampler blend"
		# Weight matters: a lone hub word can eat a whole blend, so the top
		# weights are always in your face (full table in the terminal).
		var tops := ""
		for i in mini(3, res["top_weights"].size()):
			var tw: Array = res["top_weights"][i]
			tops += "%s %+.2f  " % [str(tw[0]), float(tw[1])]
		status_label.text = "exported (%s) top: %s-> %s" % [how, tops, res["path"]]
