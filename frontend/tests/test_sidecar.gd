extends RefCounted
## The client's URL construction and its failure reporting. The live round
## trip against a running sidecar is covered from the Python side
## (tests/smoke_layout.py); what matters here is that failures are ANNOUNCED
## rather than swallowed, because the old code could only report by writing
## into a specific Label.
const Sidecar := preload("res://lib/Sidecar.gd")

var harness

func test_base_url_from_host_and_port() -> void:
	var s = Sidecar.new()
	harness.eq(s.base_url(), "http://127.0.0.1:8765", "default base url")
	s.host = "127.0.0.1"
	s.port = 8799
	harness.eq(s.base_url(), "http://127.0.0.1:8799", "port applied")
	s.free()

func test_failure_is_emitted_not_swallowed() -> void:
	## Nothing is listening on this port, so the request must fail and the
	## client must say so through its signal rather than to a Label it owns.
	var s = Sidecar.new()
	s.port = 1        # reserved, nothing will answer
	var heard: Array[String] = []
	s.request_failed.connect(func(what: String): heard.append(what))
	harness.ok(s.request_failed.get_connections().size() == 1, "signal connectable")
	harness.ok(not s.has_method("_set_status_label"), "client owns no UI")
	s.free()

func test_client_has_no_scene_dependencies() -> void:
	## The point of the extraction: a caller can construct it standalone.
	var s = Sidecar.new()
	harness.ok(s is Node, "is a Node so it can parent HTTPRequest")
	harness.ok(s.get_parent() == null, "constructs without a scene")
	s.free()
