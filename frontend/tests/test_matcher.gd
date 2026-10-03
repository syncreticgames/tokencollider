extends RefCounted
## Autocomplete ranking. Cheap to test now that it is not welded to a widget.
const Matcher := preload("res://lib/Matcher.gd")

var harness

func test_prefix_outranks_substring() -> void:
	var got := Matcher.rank("mar", ["Denmark", "Marble", "marzipan", "smart"])
	harness.eq(got[0], "Marble", "prefix first")
	harness.eq(got[1], "marzipan", "prefix second, alphabetical")
	harness.ok(got.find("Denmark") > 1, "substring ranks after prefixes")

func test_case_insensitive_but_preserves_original() -> void:
	var got := Matcher.rank("MARB", ["Marble"])
	harness.eq(got.size(), 1, "case-insensitive match")
	harness.eq(got[0], "Marble", "original casing preserved for display")

func test_empty_query_matches_nothing() -> void:
	## A dropdown of every landmark is not a suggestion.
	harness.eq(Matcher.rank("", ["a", "b"]).size(), 0, "empty query")
	harness.eq(Matcher.rank("   ", ["a", "b"]).size(), 0, "whitespace query")

func test_limit_is_enforced() -> void:
	var many: Array[String] = []
	for i in 50:
		many.append("item%02d" % i)
	harness.eq(Matcher.rank("item", many).size(), Matcher.MAX_MATCHES, "capped")
	harness.eq(Matcher.rank("item", many, 3).size(), 3, "explicit limit")

func test_no_match_is_empty_not_everything() -> void:
	harness.eq(Matcher.rank("zzz", ["alpha", "beta"]).size(), 0, "no match")
