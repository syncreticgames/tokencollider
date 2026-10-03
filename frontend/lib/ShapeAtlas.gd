extends RefCounted
## Node sprite textures: silhouettes, distance fields and density fills.
##
## Pure generation. No scene, no node, no sidecar. Give it a shape kind and a
## density digit and it hands back an ImageTexture, caching what it has already
## built. That is the whole contract, which is why it can be tested headlessly
## and why adding a shape is a matter of writing one predicate.
##
## Every node is a flat, unshaded, billboarded shape with its semantic color
## painted flat (screen-eyedropping reads the true color, no shading
## gradient) and its density digit superimposed. Shape is taxonomy: circle =
## survey node, square = manual addition, triangle = cursor.

const SHAPE_TEX_SIZE := 128
const STROKE_PX := 5.0  # ring width, in texture pixels
const KINDS := ["circle", "square", "triangle"]

var _textures := {}  # "kind:digit" -> ImageTexture
var _fields := {}    # kind -> distance field, shared by all ten digits


## A shape is one predicate. Everything else (ring, fill, all ten density
## variants) derives from it, so adding a heart or a star stays a matter of
## writing where its interior is.
static func shape_inside(kind: String, p: Vector2, half: float, size: int, y: int) -> bool:
	match kind:
		"circle":
			return p.length() <= half - 3.0
		"square":
			return absf(p.x) <= half - 8.0 and absf(p.y) <= half - 8.0
		"triangle":  # apex up
			var t := (y + 0.5) / size  # 0 top .. 1 bottom
			return absf(p.x) <= t * (half - 3.0) and t > 0.06
	return false

func shape_field(kind: String) -> PackedFloat32Array:
	## Pixels from each interior point to the nearest point outside the shape,
	## by two-pass chamfer. Eroding this field is what retreats the fill along
	## the silhouette, so a star's fill stays star-shaped as it shrinks.
	if _fields.has(kind):
		return _fields[kind]
	var size := SHAPE_TEX_SIZE
	var half := size / 2.0
	var d := PackedFloat32Array()
	d.resize(size * size)
	for y in size:
		for x in size:
			var p := Vector2(x - half + 0.5, y - half + 0.5)
			d[y * size + x] = float(size * 4) if shape_inside(kind, p, half, size, y) else 0.0
	var d1 := 1.0
	var d2 := 1.4142135
	for y in size:
		for x in size:
			var i := y * size + x
			if d[i] == 0.0:
				continue
			var v := d[i]
			if x > 0:
				v = minf(v, d[i - 1] + d1)
			if y > 0:
				v = minf(v, d[i - size] + d1)
				if x > 0:
					v = minf(v, d[i - size - 1] + d2)
				if x < size - 1:
					v = minf(v, d[i - size + 1] + d2)
			d[i] = v
	for y in range(size - 1, -1, -1):
		for x in range(size - 1, -1, -1):
			var i := y * size + x
			if d[i] == 0.0:
				continue
			var v := d[i]
			if x < size - 1:
				v = minf(v, d[i + 1] + d1)
			if y < size - 1:
				v = minf(v, d[i + size] + d1)
				if x < size - 1:
					v = minf(v, d[i + size + 1] + d2)
				if x > 0:
					v = minf(v, d[i + size - 1] + d2)
			d[i] = v
	_fields[kind] = d
	return d

func texture(kind: String, digit: int = -1) -> ImageTexture:
	## Density as fill vs stroke: the ring always marks the node's full extent,
	## the fill retreats from it as density drops. 9 (densest tenth) reads as a
	## solid shape, 0 (the void frontier) as an outline around a dot. Alpha
	## stays binary so the sprite can depth-write, and the fill is never
	## blended, so eyedropping it still reads the true semantic color.
	var key := "%s:%d" % [kind, digit]
	if _textures.has(key):
		return _textures[key]
	var size := SHAPE_TEX_SIZE
	var field := shape_field(kind)
	var max_dist := 0.0
	for v in field:
		max_dist = maxf(max_dist, v)
	# No density reported (dev universes, ghosts, a cursor before its blend
	# solves) stays solid: the shape reads exactly as it did before density
	# had a channel of its own.
	var inset := STROKE_PX
	if digit >= 0:
		var t := 1.0 - clampf(digit / 9.0, 0.0, 1.0)
		inset = STROKE_PX + t * maxf(max_dist - STROKE_PX, 0.0)
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	img.fill(Color(0, 0, 0, 0))
	for y in size:
		for x in size:
			var d := field[y * size + x]
			if d > 0.0 and (d <= STROKE_PX or d >= inset):
				img.set_pixel(x, y, Color.WHITE)
	var tex := ImageTexture.create_from_image(img)
	_textures[key] = tex
	return tex
