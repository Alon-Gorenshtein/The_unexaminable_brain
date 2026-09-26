import io
import json
import subprocess

import lib_fmt as F
from check_claims import check_rows, has_token, main as claims_main, parse_claims
from lib_digest import build_digest, digest_text, stale_stems
from numbers_diff import main as diff_main, tokens


# ---- lib_fmt -------------------------------------------------------------------------------------

def test_fmt_kinds():
    assert F.fmt(97.44, "pct1") == "97.4%" and F.fmt(97.44, "pct1n") == "97.4"
    assert F.fmt(0.78321, "auc3") == "0.783" and F.fmt(5321, "int") == "5321" and F.fmt(12404, "int") == "12,404"


def test_fmt_int_comma_threshold_and_unknown_kind():
    assert F.fmt(9999, "int") == "9999"
    assert F.fmt(10000, "int") == "10,000"
    assert F.fmt(3.14159, "dec2") == "3.14"
    try:
        F.fmt(1, "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind must raise")


def test_fmt_dec4_keeps_four_decimals_and_rejects_a_three_decimal_literal():
    assert F.fmt(0.0004782744819136209, "dec4") == "0.0005" and F.fmt(0.00028, "dec4") == "0.0003"
    assert _one(0.0004782744819136209, "dec4", "by more than 0.0005") == []
    assert _one(0.0004782744819136209, "dec4", "by more than 0.000")     # three decimals is not the dec4 literal
    assert _one(0.0004782744819136209, "dec4", "by more than 0.0004")    # wrong fourth decimal


def test_fmt_negative_int_uses_the_same_comma_rule():
    assert F.fmt(-12404, "int") == "-12,404"
    assert F.fmt(-9999, "int") == "-9999"
    assert F.fmt(-10000, "int") == "-10,000"


# ---- check_rows ----------------------------------------------------------------------------------

def test_claims_pass_and_trip():
    d = {"audit": {"brennan_lowest_em_auroc": 0.7832}}
    rows = [("m.md", "audit:brennan_lowest_em_auroc", "auc3", "AUROC, 0.783 (95% CI")]
    assert check_rows(rows, d, lambda f: "The AUROC, 0.783 (95% CI, 0.771 to 0.796)") == []
    drifted = check_rows(rows, d, lambda f: "The AUROC, 0.784 (95% CI, 0.771 to 0.796)")   # text drifted
    assert len(drifted) == 1 and "audit:brennan_lowest_em_auroc" in drifted[0] and "not found" in drifted[0]
    assert check_rows([("m.md", "audit:brennan_lowest_em_auroc", "auc3", "AUROC, 0.784")], d,
                      lambda f: "AUROC, 0.784")                                            # literal disagrees with digest


def test_claims_literal_disagreeing_with_digest_names_both_values():
    d = {"audit": {"x": 0.7832}}
    bad = check_rows([("m.md", "audit:x", "auc3", "AUROC, 0.784")], d, lambda f: "AUROC, 0.784")
    assert len(bad) == 1 and "0.783" in bad[0] and "0.784" in bad[0]


def test_claims_unknown_key_is_a_message_not_a_keyerror():
    d = {"audit": {"x": 0.5}}
    bad = check_rows([("m.md", "audit:missing_key", "auc3", "0.500")], d, lambda f: "0.500")
    assert len(bad) == 1 and "audit:missing_key" in bad[0] and "not in digest" in bad[0]
    bad = check_rows([("m.md", "nosuchstem:x", "auc3", "0.500")], d, lambda f: "0.500")
    assert len(bad) == 1 and "nosuchstem:x" in bad[0] and "not in digest" in bad[0]
    bad = check_rows([("m.md", "no_colon_key", "auc3", "0.500")], d, lambda f: "0.500")
    assert len(bad) == 1 and "no_colon_key" in bad[0]


def test_claims_unknown_kind_is_named_and_not_blamed_on_the_digest():
    d = {"audit": {"x": 0.5}}
    bad = check_rows([("m.md", "audit:x", "auc4", "0.500")], d, lambda f: "0.500")
    assert len(bad) == 1 and "unknown kind 'auc4'" in bad[0] and "digest" not in bad[0]


def _one(value, kind, literal, text=None):
    """Run a single claim; the file text defaults to the literal itself."""
    return check_rows([("m.md", "s:v", kind, literal)], {"s": {"v": value}},
                      lambda f: literal if text is None else text)


def test_wrong_number_hidden_inside_a_longer_number_is_rejected():
    assert _one(7.4, "pct1", "97.4%")            # 7.4% inside 97.4%
    assert _one(404, "int", "12,404")            # 404 inside 12,404
    assert _one(0.78, "dec2", "10.78")           # 0.78 inside 10.78
    assert _one(12, "int", "120 patients")       # 12 inside 120
    assert _one(12, "int", "12,404 patients")    # 12 inside 12,404
    assert _one(8, "int", "8.5% of patients")    # 8 inside 8.5
    assert _one(7.4, "pct1", "7.45%")            # different value
    assert _one(7.4, "pct1n", "7.45")


def test_whole_token_numbers_still_pass():
    assert _one(8.5, "pct1", "8.5%", text="Rates were (7.4%, 8.5%) overall.") == []
    assert _one(7.4, "pct1", "7.4%", text="Rates were (7.4%, 8.5%) overall.") == []
    assert _one(12404, "int", "12,404", text="A cohort (n = 12,404) was studied.") == []
    assert _one(0.783, "auc3", "0.783 (95% CI", text="AUROC was 0.783 (95% CI, 0.771 to 0.796)") == []
    assert _one(8.5, "pct1", "8.5%", text="The rate was 8.5%.") == []
    assert _one(8.5, "pct1", "8.5%", text="Was 8.5%, then 9%.") == []
    assert _one(12, "int", "12", text="12 patients") == []
    assert _one(12, "int", "12", text="n=12.") == []


def test_number_only_literal_is_matched_as_a_token_in_the_file_text_too():
    # the literal is right and matches the digest, but the file only holds it inside a longer number
    bad = _one(8.5, "pct1", "8.5%", text="The rate was 98.5%.")
    assert len(bad) == 1 and "not found" in bad[0]
    # a literal with context words is a plain substring, anchored by the words
    assert _one(0.783, "auc3", "AUROC, 0.783", text="The AUROC, 0.783 (95% CI") == []


def test_has_token_boundaries():
    assert has_token("(7.4%, 8.5%)", "8.5%") and has_token("was 8.5%.", "8.5%")
    assert not has_token("97.4%", "7.4%") and not has_token("12,404", "404")
    assert not has_token("10.78", "0.78") and not has_token("120", "12") and not has_token("8.5", "8")
    assert not has_token("12,404", "12") and not has_token("1.25.3", "1.25")


def test_unreadable_registered_file_is_reported_not_raised():
    def boom(f):
        raise FileNotFoundError(2, "No such file or directory")
    bad = check_rows([("nope.md", "s:v", "int", "12")], {"s": {"v": 12}}, boom)
    assert len(bad) == 1 and "s:v" in bad[0] and "nope.md" in bad[0] and "cannot read" in bad[0]


def test_empty_literal_is_reported():
    bad = check_rows([("m.md", "s:v", "int", "  ")], {"s": {"v": 12}}, lambda f: "12")
    assert len(bad) == 1 and "empty literal" in bad[0]


# ---- literals with context words are anchored at their own number edges --------------------------

def test_context_literal_hidden_inside_a_longer_number_in_the_file_is_rejected():
    for value, kind, lit, text in [
        (7.4, "pct1", "7.4% of", "97.4% of patients"),
        (12, "int", "12 patients", "were 112 patients"),
        (0.783, "auc3", "0.783 (95% CI", "AUROC 10.783 (95% CI, 0.771 to 0.796)"),
        (0.783, "auc3", "AUROC, 0.783", "The AUROC, 0.7831 was"),
        (5509, "int", "n = 5509", "(n = 55091)"),
        (0.5, "pct1n", "0.5 of", "-0.5 of patients"),          # sign flip on the file side
    ]:
        bad = _one(value, kind, lit, text=text)
        assert len(bad) == 1 and "not found" in bad[0], (lit, text, bad)


def test_context_literal_positive_controls_still_pass():
    assert _one(0.783, "auc3", "AUROC, 0.783 (95% CI", text="The AUROC, 0.783 (95% CI, 0.771 to 0.796)") == []
    assert _one(12, "int", "12 patients", text="There were 12 patients.") == []
    assert _one(5509, "int", "n = 5509", text="The cohort (n = 5509)") == []
    assert _one(3, "int", "aged 3-15 years", text="children aged 3-15 years") == []


# ---- sign rule -----------------------------------------------------------------------------------

def test_sign_flips_are_rejected_and_ranges_are_not_signs():
    assert _one(0.5, "pct1n", "-0.5")                           # digest 0.5, literal -0.5
    assert _one(-0.5, "dec2", "0.50")                           # digest -0.5, literal 0.50
    assert _one(-0.5, "dec2", "-0.50") == []                    # both negative
    assert has_token("range 3-15", "15") and has_token("2020-2021", "2021")
    assert has_token("delta \u22120.5", "-0.5")                # U+2212 is a minus sign
    assert not has_token("delta \u22120.5", "0.5")
    assert not has_token("delta -0.5", "0.5") and not has_token("delta 0.5", "-0.5")
    assert not has_token("3-0.5", "-0.5")                       # a hyphen glued to a digit is a range, not a sign
    assert has_token("(-0.5, 0.3)", "-0.5") and has_token("(-0.5, 0.3)", "0.3")


def test_word_character_before_a_number_is_rejected():
    assert not has_token("S12 was", "12") and not has_token("IL12 was", "12")
    assert has_token("(12 was", "12") and has_token("n=12", "12")


# ---- claims file parsing -------------------------------------------------------------------------

def test_parse_claims_reports_short_and_long_rows_with_line_numbers_and_skips_blanks():
    src = "m.md\ta:x\tint\t12\n\n   \nbad\trow\tonly\nm.md\ta:y\tint\t13\textra\nm.md\ta:z\tint\t14\n"
    rows, bad = parse_claims(io.StringIO(src))
    assert rows == [("m.md", "a:x", "int", "12"), ("m.md", "a:z", "int", "14")]
    assert len(bad) == 2
    assert "line 4" in bad[0] and "got 3" in bad[0]
    assert "line 5" in bad[1] and "got 5" in bad[1]


def test_parse_claims_keeps_quotes_in_a_literal():
    rows, bad = parse_claims(io.StringIO('m.md\ta:x\tauc3\tAUROC "0.783" (95% CI\n'))
    assert bad == [] and rows == [("m.md", "a:x", "auc3", 'AUROC "0.783" (95% CI')]


# ---- digest --------------------------------------------------------------------------------------

def test_build_digest_skips_sidecars_and_itself_and_is_deterministic(tmp_path):
    a, b = tmp_path / "one", tmp_path / "two"
    for d, order in ((a, ("a", "b")), (b, ("b", "a"))):
        d.mkdir()
        for name in order:
            (d / f"{name}.json").write_text(json.dumps({"k": name, "v": 1.5}))
        (d / "._a.json").write_bytes(b"\xb0\xff garbage")              # AppleDouble sidecar
        (d / "revision_digest.json").write_text('{"stale": {}}')       # must not feed itself
    da, db = build_digest(a), build_digest(b)
    assert sorted(da) == ["a", "b"]
    assert digest_text(da) == digest_text(build_digest(a)) == digest_text(db)


def test_build_digest_carries_frozen_digests_and_small_tables_and_notices_a_change(tmp_path):
    out = tmp_path / "output"
    rev, v1 = out / "revision", out / "v1_submitted"
    rev.mkdir(parents=True)
    v1.mkdir()
    (rev / "audit.json").write_text(json.dumps({"n": 3}))
    (v1 / "external_eicu_digest.json").write_text(json.dumps(
        {"full": {"mortality_pct": [16.6, 15.7, 17.6]}, "rows": [{"phenotype": "AIS", "v": 1}, {"phenotype": "ICH", "v": 2}]}))
    (rev / "excluded_vs_included.csv").write_text("comparison,group,n,mortality_pct\nverbal,excluded,42,61.9047\n")
    d = build_digest(rev)
    assert d["frozen_eicu"] == {"full.mortality_pct.0": 16.6, "full.mortality_pct.1": 15.7, "full.mortality_pct.2": 17.6,
                                "rows.AIS.phenotype": "AIS", "rows.AIS.v": 1, "rows.ICH.phenotype": "ICH", "rows.ICH.v": 2}
    assert d["excluded_vs_included"] == {"verbal_excluded.n": 42, "verbal_excluded.mortality_pct": 61.9047}
    committed = json.loads(digest_text(d))
    assert stale_stems(build_digest(rev), committed) == []
    (v1 / "external_eicu_digest.json").write_text(json.dumps({"full": {"mortality_pct": [16.7, 15.7, 17.6]}}))
    assert stale_stems(build_digest(rev), committed) == ["frozen_eicu"]


def test_stale_stems_reports_changed_added_and_removed_stems():
    base = {"a": {"x": 1}, "b": {"y": 2.5}}
    assert stale_stems(base, {"a": {"x": 1}, "b": {"y": 2.5}}) == []
    assert stale_stems(base, {"a": {"x": 1}, "b": {"y": 2.6}}) == ["b"]
    assert stale_stems(base, {"a": {"x": 1}}) == ["b"]
    assert stale_stems(base, {**base, "c": {}}) == ["c"]


# ---- the tool end to end on a throwaway tree -----------------------------------------------------

def _tree(tmp_path, claims, text="n = 12,404 and 97.4% overall. Rate 7.4%.", stale=False):
    rev = tmp_path / "output" / "revision"
    rev.mkdir(parents=True)
    (rev / "a.json").write_text(json.dumps({"n": 12404, "p": 7.4}))
    (rev / "revision_digest.json").write_text(digest_text(build_digest(rev)))
    if stale:
        (rev / "a.json").write_text(json.dumps({"n": 12405, "p": 7.4}))
    (rev / "claims.tsv").write_text(claims)
    (tmp_path / "m.md").write_text(text)
    return tmp_path


def test_empty_registry_exits_zero_with_a_warning(tmp_path, capsys):
    assert claims_main(root=_tree(tmp_path, "")) == 0
    out, err = capsys.readouterr()
    assert "all 0 claims hold" in out and "WARNING: 0 claims registered" in err


def test_stale_digest_is_refused(tmp_path, capsys):
    assert claims_main(root=_tree(tmp_path, "", stale=True)) == 1
    out, _ = capsys.readouterr()
    assert "digest is stale: re-run code/72_revision_digest.py" in out and "differ: a" in out


def test_missing_digest_is_refused(tmp_path, capsys):
    root = _tree(tmp_path, "")
    (root / "output" / "revision" / "revision_digest.json").unlink()
    assert claims_main(root=root) == 1
    assert "72_revision_digest.py" in capsys.readouterr().out


def test_one_good_and_two_bad_rows_report_exactly_two(tmp_path, capsys):
    claims = ("m.md\ta:n\tint\t12,404\n"                  # good
              "m.md\ta:p\tpct1\t97.4%\n"                  # 7.4% only as part of 97.4%
              "m.md\ta:n\tint\t12,404 cases\n")           # not in the file
    assert claims_main(root=_tree(tmp_path, claims)) == 1
    out, _ = capsys.readouterr()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2 and lines[0].startswith("a:p") and lines[1].startswith("a:n") and "not found" in lines[1]


def test_malformed_claims_line_fails_the_run(tmp_path, capsys):
    assert claims_main(root=_tree(tmp_path, "m.md\ta:n\tint\t12,404\nm.md\ta:n\tint\n")) == 1
    assert "line 2" in capsys.readouterr().out


def test_missing_registered_file_is_a_message_not_a_traceback(tmp_path, capsys):
    assert claims_main(root=_tree(tmp_path, "nope.md\ta:n\tint\t12,404\n")) == 1
    out, _ = capsys.readouterr()
    assert "nope.md" in out and "cannot read" in out and "Traceback" not in out


def test_missing_claims_file_is_a_message_not_a_traceback(tmp_path, capsys):
    root = _tree(tmp_path, "")
    assert claims_main(root=root, claims_path=root / "absent.tsv") == 1
    assert "absent.tsv" in capsys.readouterr().out


# ---- numbers_diff --------------------------------------------------------------------------------

def test_numbers_diff_tokens():
    assert tokens("The 12,404 patients, 5.3% and 0.783") == {"12,404", "5.3%", "0.783"}


def test_numbers_diff_signs_and_leading_dot_decimals():
    assert tokens("-0.001") == {"-0.001"} and tokens("0.001") == {"0.001"}
    assert tokens("-0.001") != tokens("0.001")
    assert tokens("P < .001") == {".001"}
    assert tokens("mean −0.5 (CI, −1.2 to 0.3)") == {"-0.5", "-1.2", "0.3"}   # U+2212 minus is normalised


def test_numbers_diff_comma_lists_and_thousands():
    assert tokens("12,404") == {"12,404"}
    assert tokens("cited 1,2,3.") == {"1", "2", "3"}
    assert tokens("refs (3,4)") == {"3", "4"}
    assert tokens("1,234,567 rows and 1,2345") == {"1,234,567", "1", "2345"}


def test_numbers_diff_ranges_do_not_gain_a_sign():
    assert tokens("GCS 3-15 and 2020-2021, 0.771-0.796") == {"3", "15", "2020", "2021", "0.771", "0.796"}


def _git_repo(path, tag=True):
    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *a],
                       cwd=path, check=True, capture_output=True)
    git("init", "-q")
    (path / "m.md").write_text("n = 12,404 and 5.3%\n")
    git("add", "m.md")
    git("commit", "-q", "-m", "x")
    if tag:
        git("tag", "jicm-v1")


def test_numbers_diff_usage_is_printed(capsys):
    assert diff_main([]) != 0
    assert "usage" in capsys.readouterr().err.lower()


def test_numbers_diff_missing_tag_is_reported_with_the_ref_name(tmp_path, monkeypatch, capsys):
    _git_repo(tmp_path, tag=False)
    monkeypatch.chdir(tmp_path)
    assert diff_main(["m.md"]) == 1
    assert "jicm-v1" in capsys.readouterr().err


def test_numbers_diff_file_absent_from_the_worktree_is_reported(tmp_path, monkeypatch, capsys):
    _git_repo(tmp_path)
    (tmp_path / "m.md").unlink()
    monkeypatch.chdir(tmp_path)
    assert diff_main(["m.md"]) == 1
    assert "m.md" in capsys.readouterr().err


def test_numbers_diff_lists_the_changed_numbers(tmp_path, monkeypatch, capsys):
    _git_repo(tmp_path)
    (tmp_path / "m.md").write_text("n = 12,405 and 5.3%\n")
    monkeypatch.chdir(tmp_path)
    assert diff_main(["m.md"]) == 0
    out = capsys.readouterr().out
    assert "only in submitted: ['12,404']" in out and "only in revised:   ['12,405']" in out
