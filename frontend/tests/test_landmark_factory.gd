extends RefCounted
## Landmarks as built for the world: where the label sits, and that a picture
## comes from the injected fetcher (the factory knows no URL and does no HTTP).
const LandmarkFactory := preload("res://viewport/LandmarkFactory.gd")
const ShapeAtlas := preload("res://lib/ShapeAtlas.gd")

var harness

func _factory():
	var f = LandmarkFactory.new()
	f.shapes = ShapeAtlas.new()
	return f

func test_image_label_sits_above_the_picture() -> void:
	## The picture is 1.6 tall and centered, so its top is at 0.8. The label's
	## image height used to be overwritten by the node one, putting it on top
	## of the picture.
	var f = _factory()
	var asked: Array[String] = []
	f.fetch_bytes = func(path: String):
		asked.append(path)
		return null
	var pic = f.make_landmark("image:abc", "manual", -1,
		{"kind": "image", "label": "sunset", "image": "/image/abc"})
	var word = f.make_landmark("red", "manual")
	harness.ok(pic.get_node("Tag").position.y > 0.8, "image label clears the picture")
	harness.close(word.get_node("Tag").position.y, 0.7, 1e-6, "phrase label unchanged")
	harness.eq(asked, ["/image/abc"] as Array[String], "the thumbnail came from the fetcher")
	pic.free()
	word.free()
	f.free()
