extends Node
## The sidecar HTTP client. Everything that knows a URL lives here.
##
## Previously this was `_api()` inside Main.gd, which reached directly into
## `status_label` to report failures. That coupling is why the transport could
## not be tested or reused: exercising a request meant standing up the whole
## scene, and any other screen wanting to talk to the sidecar would have had to
## own a status label.
##
## Now it emits. Callers decide what a failure looks like, so a main menu can
## show a connection dot and the viewport can show a status line, from the same
## client.

signal request_failed(what: String)   ## human-readable, safe to show a user
signal unreachable                    ## the sidecar is not answering at all

var host := "127.0.0.1"
var port := 8765


func base_url() -> String:
	return "http://%s:%d" % [host, port]


func request(path: String, body = null) -> Variant:
	## GET when body is null, POST as JSON otherwise. Returns the parsed
	## response, or null after emitting the reason.
	##
	## The JSON content type is not decoration: the sidecar refuses any POST
	## without it, which is what stops a web page from driving this API
	## (see docs/security.md). Do not "simplify" it away.
	var http := HTTPRequest.new()
	add_child(http)
	var err: int
	if body == null:
		err = http.request(base_url() + path)
	else:
		err = http.request(base_url() + path,
			PackedStringArray(["Content-Type: application/json"]),
			HTTPClient.METHOD_POST, JSON.stringify(body))
	if err != OK:
		http.queue_free()
		request_failed.emit("request failed to start (%d)" % err)
		return null
	var res: Array = await http.request_completed
	http.queue_free()
	if res[0] != HTTPRequest.RESULT_SUCCESS:
		unreachable.emit()
		request_failed.emit("sidecar unreachable")
		return null
	var parsed = JSON.parse_string(res[3].get_string_from_utf8())
	if res[1] != 200:
		# The sidecar answers 403 for a refused request and 400 for a bad one;
		# both carry an {"error": ...} body worth showing verbatim.
		var detail = parsed.get("error", parsed) if parsed is Dictionary else parsed
		request_failed.emit("sidecar error %d: %s" % [res[1], str(detail)])
		return null
	return parsed
