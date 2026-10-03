extends RefCounted
## Keeps a reply from landing in a world it wasn't asked about.
##
## A request is stamped with the view signature and a sequence number when it
## is sent. Its reply counts only if nothing newer of the same kind was sent
## since, and the view is still the one it was asked in. Interrogation ghosts
## and blend weights both carry positions in one view's chart; drawn into
## another, they sit somewhere that means nothing.

var _sent := 0


func open(view: String) -> Dictionary:
	_sent += 1
	return {"seq": _sent, "view": view}


func is_current(ticket: Dictionary, view: String) -> bool:
	return int(ticket["seq"]) == _sent and str(ticket["view"]) == view
