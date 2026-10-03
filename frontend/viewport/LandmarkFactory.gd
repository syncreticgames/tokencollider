extends Node
## Builds the things that appear in the world: node sprites, image sprites,
## density digits, and the landmark that carries them.
##
## Everything it needs from outside is INJECTED (the shape atlas, a way to
## fetch a thumbnail's bytes), so it never reaches back into the viewport for
## state, which is what made this code impossible to move before. It knows no
## URL and does no HTTP: lib/Sidecar.gd does.

var shapes: RefCounted = null   ## lib/ShapeAtlas.gd
## path -> PackedByteArray, or null when the fetch failed (reported by whoever
## does the fetching). Main wires it to Sidecar.request_bytes.
var fetch_bytes: Callable = func(_path: String): return null


func make_node_sprite(shape_kind: String, size: float) -> MeshInstance3D:
	var shape := MeshInstance3D.new()
	shape.name = "Shape"
	var quad := QuadMesh.new()
	quad.size = Vector2(size, size)
	shape.mesh = quad
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	# Scissor, not alpha blend: blended sprites never write depth, so nodes
	# drew straight through each other regardless of distance. Binary alpha
	# puts them in the opaque pass, where they occlude properly.
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	mat.alpha_scissor_threshold = 0.5
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.albedo_texture = shapes.texture(shape_kind)
	shape.material_override = mat
	shape.set_meta("kind", shape_kind)
	return shape

func make_digit_label() -> Label3D:
	var digit := Label3D.new()
	digit.name = "Digit"
	digit.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	digit.font_size = 44
	digit.pixel_size = 0.009
	digit.modulate = Color(0.08, 0.08, 0.08)
	digit.outline_size = 10
	digit.outline_modulate = Color(1, 1, 1, 0.85)
	digit.render_priority = 2  # draw over the flat shape at the same depth
	digit.visible = false
	return digit

func set_density(node: Node3D, density: int) -> void:
	node.set_meta("density", density)
	var shape := node.get_node("Shape") as MeshInstance3D
	if not node.get_meta("image", false):
		shape.material_override.albedo_texture = shapes.texture(
			str(shape.get_meta("kind")), density)
	# There is one cursor, so it can afford a number. Two thousand landmarks
	# cannot: their density rides the fill/stroke split instead.
	var digit := node.get_node("Digit") as Label3D
	digit.visible = density >= 0 and bool(node.get_meta("show_digit", false))
	if digit.visible:
		digit.text = str(density)

func make_image_sprite(url: String, size: float) -> MeshInstance3D:
	## A picture in the world: an unshaded billboard quad that takes the
	## sidecar's thumbnail as its texture once it arrives.
	var shape := MeshInstance3D.new()
	shape.name = "Shape"
	var quad := QuadMesh.new()
	quad.size = Vector2(size, size)
	shape.mesh = quad
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.albedo_color = Color(0.5, 0.5, 0.5)
	shape.material_override = mat
	shape.set_meta("kind", "image")
	load_image_texture(shape, url)
	return shape

func load_image_texture(shape: MeshInstance3D, url: String) -> void:
	## The picture stays grey if the thumbnail can't be had; the fetcher has
	## already said why.
	var data = await fetch_bytes.call(url)
	if data == null or not is_instance_valid(shape):
		return
	var img := Image.new()
	if img.load_jpg_from_buffer(data) != OK:
		return
	var mat: StandardMaterial3D = shape.material_override
	mat.albedo_texture = ImageTexture.create_from_image(img)
	# Keep the picture's aspect: the quad is square by construction.
	var quad: QuadMesh = shape.mesh
	var w := float(img.get_width())
	var h := float(img.get_height())
	var s := quad.size.x
	quad.size = Vector2(s, s * h / w) if w >= h else Vector2(s * w / h, s)

func make_landmark(text: String, source: String = "preload", density: int = -1, entry: Dictionary = {}) -> Node3D:
	var root := Node3D.new()
	root.set_meta("source", source)
	var is_image: bool = str(entry.get("kind", "phrase")) == "image"
	root.set_meta("image", is_image)
	if is_image:
		root.add_child(make_image_sprite(str(entry.get("image", "")), 1.6))
	else:
		var kind := "square" if source == "manual" else "circle"
		root.add_child(make_node_sprite(kind, 0.8 if source == "manual" else 0.7))
	root.add_child(make_digit_label())
	set_density(root, density)
	var tag := Label3D.new()
	tag.name = "Tag"
	tag.text = str(entry.get("label", text))
	# Above the picture (1.6 tall, centred, so its top is at 0.8), or just
	# above a node sprite.
	tag.position.y = 1.0 if is_image else 0.7
	tag.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	tag.alpha_cut = Label3D.ALPHA_CUT_DISCARD  # opaque pass = real occlusion
	tag.font_size = 48
	tag.pixel_size = 0.008
	root.add_child(tag)
	add_child(root)
	return root

