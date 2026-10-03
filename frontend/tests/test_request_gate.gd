extends RefCounted
## Out-of-order replies: only the newest request, in the view it was asked
## in, gets to change the world.
const RequestGate := preload("res://lib/RequestGate.gd")

var harness

func test_newest_reply_in_same_view_counts() -> void:
	var g = RequestGate.new()
	var t = g.open("20/norm/local")
	harness.ok(g.is_current(t, "20/norm/local"), "only request, same view")

func test_superseded_request_is_dropped() -> void:
	var g = RequestGate.new()
	var first = g.open("20/norm/local")
	var second = g.open("20/norm/local")
	harness.ok(not g.is_current(first, "20/norm/local"), "older reply landing late is dropped")
	harness.ok(g.is_current(second, "20/norm/local"), "newer reply counts")

func test_view_change_drops_reply() -> void:
	var g = RequestGate.new()
	var t = g.open("20/norm/local")
	harness.ok(not g.is_current(t, "30/norm/local"), "scrubbed since: dropped")
	harness.ok(not g.is_current(t, "20/norm/world"), "axes switched since: dropped")
