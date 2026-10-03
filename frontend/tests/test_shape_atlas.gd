extends RefCounted
## Sprite texture generation. Pure enough to test properly: same inputs, same
## image, no scene involved.
const ShapeAtlas := preload("res://lib/ShapeAtlas.gd")

var harness

func test_every_kind_produces_a_texture() -> void:
	var a = ShapeAtlas.new()
	for kind in ShapeAtlas.KINDS:
		var tex: ImageTexture = a.texture(kind)
		harness.ok(tex != null, "%s texture built" % kind)
		harness.eq(tex.get_width(), ShapeAtlas.SHAPE_TEX_SIZE, "%s width" % kind)
		harness.eq(tex.get_height(), ShapeAtlas.SHAPE_TEX_SIZE, "%s height" % kind)

func test_texture_is_cached_by_kind_and_digit() -> void:
	var a = ShapeAtlas.new()
	harness.ok(a.texture("circle", 5) == a.texture("circle", 5), "same key, same object")
	harness.ok(a.texture("circle", 5) != a.texture("circle", 6), "digit changes the texture")

func test_shape_predicate_bounds() -> void:
	## The predicate is the whole definition of a shape, so pin its edges.
	var half := 64.0
	harness.ok(ShapeAtlas.shape_inside("circle", Vector2.ZERO, half, 128, 64), "circle centre inside")
	harness.ok(not ShapeAtlas.shape_inside("circle", Vector2(70, 0), half, 128, 64), "outside radius")
	harness.ok(ShapeAtlas.shape_inside("square", Vector2(10, 10), half, 128, 64), "square centre")
	harness.ok(not ShapeAtlas.shape_inside("square", Vector2(60, 60), half, 128, 64), "square corner out")
	harness.ok(not ShapeAtlas.shape_inside("nonsense", Vector2.ZERO, half, 128, 64), "unknown kind is empty")

func test_density_changes_the_fill_not_the_extent() -> void:
	## The ring always marks the node's full extent; only the fill retreats.
	## Densest (9) must therefore have at least as many lit pixels as sparsest.
	var a = ShapeAtlas.new()
	var lit := func(digit: int) -> int:
		var img: Image = a.texture("circle", digit).get_image()
		var n := 0
		for y in img.get_height():
			for x in img.get_width():
				if img.get_pixel(x, y).a > 0.5:
					n += 1
		return n
	var dense: int = lit.call(9)
	var sparse: int = lit.call(0)
	harness.ok(dense > sparse, "dense fills more than sparse (%d > %d)" % [dense, sparse])
	harness.ok(sparse > 0, "sparsest still draws its ring")
