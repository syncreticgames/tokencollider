extends RefCounted
## Ranking for the fly-to autocomplete.
##
## Split from the widget that displays it because the RANKING is the part with
## a right answer, and the part someone will want to change. Prefix beats
## substring, both alphabetical, capped. Testing that against an ItemList means
## standing up a scene; testing it here is a list in and a list out.

const MAX_MATCHES := 12


static func rank(query: String, candidates, limit := MAX_MATCHES) -> Array[String]:
	## Prefix matches outrank substring matches, each group alphabetical.
	## Case-insensitive. An empty query matches nothing rather than
	## everything, because a dropdown of two thousand landmarks is not a
	## suggestion.
	var out: Array[String] = []
	var q := query.strip_edges().to_lower()
	if q.is_empty():
		return out
	var prefix: Array[String] = []
	var inside: Array[String] = []
	for t in candidates:
		var low := str(t).to_lower()
		if low.begins_with(q):
			prefix.append(str(t))
		elif low.contains(q):
			inside.append(str(t))
	prefix.sort()
	inside.sort()
	for t in prefix + inside:
		if out.size() >= limit:
			break
		out.append(t)
	return out
