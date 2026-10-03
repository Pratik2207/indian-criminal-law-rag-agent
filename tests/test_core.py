"""Fast unit tests (no models, no vector DB, no LLM)."""
import lawrag.citations as citations
import lawrag.retrieval as retrieval
from lawrag.citations import Source, section_mentions, verify_and_link
from lawrag.ingest import MAX_CHARS, split_section
from lawrag.parser import SECTION_RE, Section
from lawrag.retrieval import analyze


def test_section_regex_handles_missing_space_and_amendment_marks():
    assert SECTION_RE.match("104.Whoever, being under sentence").group(1) == "104"
    m = SECTION_RE.match("1[304A. Causing death by negligence.—")
    assert (m.group(1), m.group(2)) == ("304", "A")
    assert SECTION_RE.match("(2) In every case") is None


def test_split_section_respects_limit_and_keeps_text():
    body = "103. (1) Whoever commits murder.\n" + "\n".join(f"({i}) " + "x " * 300 for i in range(2, 8))
    chunks = split_section(body)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS + 50 for c in chunks)
    assert chunks[0].startswith("103.")


def test_section_text_strips_amendment_markers():
    s = Section("IPC", "Indian Penal Code, 1860", "302", "Punishment for murder", "", 1, "IPC_1860.pdf", "",
                ["302. Punishment for murder.—Whoever commits murder shall be punished with death or",
                 "1[imprisonment for life], and shall also be liable to fine."])
    assert "1[" not in s.text


def test_analyze_detects_refs_and_maps_equivalents(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "302"): [("BNS", "103")]})
    qa = analyze("What is IPC 302 called in BNS?", translate=False)
    assert ("IPC", "302") in qa.refs
    assert ("BNS", "103") in qa.mapped
    assert set(qa.acts) == {"IPC", "BNS"}

    qa = analyze("Show me Section 103 of the Bharatiya Nyaya Sanhita", translate=False)
    assert qa.refs == [("BNS", "103")]
    qa = analyze("punishment for theft", translate=False)
    assert qa.refs == [] and qa.acts == []


def test_section_mentions():
    text = "Under Section 103 of the BNS murder is punishable; formerly IPC 302. Not BNS for 10 years."
    m = section_mentions(text)
    assert ("BNS", "103") in m and ("IPC", "302") in m
    assert ("BNS", "10") not in m


def test_verify_and_link(monkeypatch):
    monkeypatch.setattr(citations, "known_sections",
                        lambda: frozenset({("BNS", "103"), ("IPC", "302"), ("BNS", "101")}))
    srcs = [Source("S1", "BNS s.103 — Punishment for murder", "app/static/BNS_2023.pdf#page=34", "", "statute",
                   "BNS", "103"),
            Source("W1", "Some judgment", "https://indiankanoon.org/doc/1/", "", "caselaw")]
    raw = ("Murder is punished under Section 103 of the BNS [S1, W1]. See also BNS 101 [S3]. "
           "And Section 999 of IPC.")
    v = verify_and_link(raw, srcs)
    assert "[[BNS s.103]](app/static/BNS_2023.pdf#page=34)" in v.markdown
    assert "(https://indiankanoon.org/doc/1/)" in v.markdown
    assert v.invalid_tags == ["S3"]
    assert ("BNS", "101") in v.unsupported
    assert ("IPC", "999") in v.nonexistent
    assert not v.ok


def test_claim_citation_mismatch(monkeypatch):
    monkeypatch.setattr(citations, "known_sections",
                        lambda: frozenset({("BNS", "103"), ("BNS", "302"), ("IPC", "302")}))
    srcs = [Source("S1", "BNS s.103 — Punishment for murder", "u", "", "statute", "BNS", "103")]
    v = verify_and_link("Murder is punished under Section 302 of the BNS [S1]. Formerly IPC 302 [S1].", srcs)
    assert v.mismatched == [(("BNS", "302"), "BNS s.103")]  # IPC 302 is a different Act: not flagged


def test_semantic_query_drops_section_numbers(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "302"): [("BNS", "103")]})
    qa = analyze("What is IPC 302 called in BNS?", translate=False)
    assert "302" not in qa.semantic_query
    assert qa.correspondences == ["IPC s.302 corresponds to BNS s.103"]


def test_subsection_tags_are_linked(monkeypatch):
    monkeypatch.setattr(citations, "known_sections", lambda: frozenset({("BNS", "310")}))
    srcs = [Source("S3", "BNS s.310 — Dacoity", "u310", "", "statute", "BNS", "310")]
    v = verify_and_link("Punishment is life imprisonment [S3(2)]. See also [S3(3), S3].", srcs)
    assert "[[BNS s.310]](u310)" in v.markdown and "S3(2)" not in v.markdown
    assert v.invalid_tags == []


def test_correspondence_notes(monkeypatch):
    import lawrag.answer as answer

    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "499"): [("BNS", "356")]})
    srcs = [Source("S1", "IPC s.499 — Defamation", "u", "", "statute", "IPC", "499")]
    assert answer.correspondence_notes(srcs) == ["IPC s.499 corresponds to BNS s.356"]


def test_bad_correspondence_flagged(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map",
                        lambda: {("IPC", "499"): [("BNS", "356")], ("IPC", "302"): [("BNS", "103")]})
    monkeypatch.setattr(citations, "known_sections",
                        lambda: frozenset({("IPC", "499"), ("BNS", "353"), ("BNS", "356"), ("IPC", "302"), ("BNS", "103")}))
    v = verify_and_link("Defamation is Section 499 of the IPC (now replaced by Section 353 of the BNS). "
                        "IPC 302 now corresponds to BNS 103.", [])
    assert v.bad_mappings == [(("IPC", "499"), ("BNS", "353"))]


def test_analyze_ignores_quantities(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map", lambda: {})
    assert analyze("Punishment under BNS for theft is up to 3 years", translate=False).refs == []
    assert analyze("Under the BNS what happens if 5 persons commit robbery", translate=False).refs == []
    for q, ref in [("IPC 302", ("IPC", "302")), ("302 IPC", ("IPC", "302")), ("u/s 438 CrPC", ("CrPC", "438")),
                   ("CrPC 438 corresponding section in BNSS", ("CrPC", "438"))]:
        assert analyze(q, translate=False).refs == [ref], q


def test_loose_citation_formats_are_normalized():
    from lawrag.citations import normalize_tags

    assert normalize_tags("defined in the IPC (S1) and BNS (S2, S6).") == "defined in the IPC [S1] and BNS [S2][S6]."
    assert normalize_tags("fine. [S2, Section 307]") == "fine. [S2]"
    assert normalize_tags("[S1, S2: Section 420, Section 318]") == "[S1][S2]"
    assert normalize_tags("life [S3(2)] and (Section 65) stay") == "life [S3(2)] and (Section 65) stay"


def test_bad_correspondence_with_called_phrasing(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "420"): [("BNS", "318")]})
    monkeypatch.setattr(citations, "known_sections", lambda: frozenset({("IPC", "420"), ("BNS", "103"), ("BNS", "318")}))
    v = verify_and_link("IPC 420 is called Section 103 of the BNS.", [])
    assert v.bad_mappings == [(("IPC", "420"), ("BNS", "103"))]


def test_mapping_question_detection_and_task(monkeypatch):
    from lawrag.answer import is_mapping_question, mapping_task
    from lawrag.retrieval import Hit

    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "302"): [("BNS", "103")]})
    qa = analyze("What is IPC 302 called in BNS?", translate=False)
    assert is_mapping_question(qa)
    assert not is_mapping_question(analyze("Explain section 302 IPC", translate=False))
    hit = Hit("BNS", "103", "Punishment for murder", "", 34, "BNS_2023.pdf", "", "", "text", 100.0, "mapped")
    task = mapping_task(qa, [hit])
    assert "Section 302 of the IPC corresponds to Section 103 of the BNS (Punishment for murder)" in task


def test_mention_with_in_connector(monkeypatch):
    monkeypatch.setattr(retrieval, "section_map", lambda: {("IPC", "302"): [("BNS", "103")]})
    monkeypatch.setattr(citations, "known_sections", lambda: frozenset({("IPC", "302"), ("BNS", "302"), ("BNS", "103")}))
    v = verify_and_link("IPC 302 is called Section 302 in the BNS.", [])
    assert ("BNS", "302") in v.mentions
    assert v.bad_mappings == [(("IPC", "302"), ("BNS", "302"))]


def test_section_map_file_loads():
    # guards against the map silently going missing (e.g. after moving files)
    retrieval.section_map.cache_clear()
    assert ("BNS", "103") in retrieval.section_map()[("IPC", "302")]
    assert ("BNSS", "482") in retrieval.section_map()[("CrPC", "438")]
