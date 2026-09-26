import pandas as pd
import pytest

from config import REV, TAB

pytestmark = pytest.mark.requires_study_files(
    "output/revision/audit.json",
    "output/revision/audit_by_rule.csv",
    "output/revision/audit_by_phenotype.csv",
    "output/revision/sofa_crosstab.csv",
    "output/v1_submitted/table3_aim3_by_phenotype.csv")

V1 = TAB.parent / "v1_submitted"


def test_one_row_per_rule_and_strategy():
    r = pd.read_csv(REV / "audit_by_rule.csv")
    assert not r.duplicated(["rule", "slug"]).any()
    assert set(r.rule) == {"lowest_em", "earliest", "latest", "carryforward_lowest_em", "legacy_population_lowest_em"}


def test_corrected_population_is_larger_than_the_submitted_one():
    r = pd.read_csv(REV / "audit_by_rule.csv")
    n = lambda rule: r[(r.rule == rule) & (r.slug == "brennan")].n.iloc[0]
    v1 = pd.read_csv(V1 / "table3_aim3_by_phenotype.csv")
    submitted_n = v1[(v1.phenotype == "ALL") & (v1.strategy == "kramer")].n.iloc[0]
    assert n("legacy_population_lowest_em") == submitted_n < n("lowest_em")


def test_legacy_population_matches_submitted_counts_and_auroc():
    r = pd.read_csv(REV / "audit_by_rule.csv")
    got = r[r.rule == "legacy_population_lowest_em"].set_index("slug")      # the submitted population
    v1 = pd.read_csv(V1 / "table3_aim3_by_phenotype.csv")
    v1 = v1[v1.phenotype == "ALL"].set_index("strategy")
    for slug, old in [("complete_case", "drop"), ("impute1", "impute1"), ("gcs_t", "gcs_t"), ("brennan", "kramer")]:
        assert got.loc[slug, "n"] == v1.loc[old, "n"]
        assert got.loc[slug, "deaths"] == v1.loc[old, "n_deaths"]
        assert abs(got.loc[slug, "auroc"] - v1.loc[old, "auroc"]) < 0.003
        assert abs(got.loc[slug, "mean_gcs"] - v1.loc[old, "mean_tg"]) < 0.05
        assert abs(got.loc[slug, "brier"] - v1.loc[old, "brier"]) < 0.003


def test_interval_brackets_estimate():
    r = pd.read_csv(REV / "audit_by_rule.csv")
    assert ((r.lo < r.auroc) & (r.auroc < r.hi)).all()


def test_selection_block_is_consistent_with_the_table():
    import json
    a = json.loads((REV / "audit.json").read_text())
    assert a["sel_n_total"] == a["brennan_lowest_em_n"]
    assert a["sel_n_total"] - a["sel_n_excluded"] == a["complete_case_lowest_em_n"]
    assert a["sel_rr"] > 1 and a["sel_rr_lo"] < a["sel_rr"] < a["sel_rr_hi"]
    assert 0 < a["reclass_impute1_vs_brennan_pct"] < 100


def test_phenotype_rows_add_up_to_the_overall_cohort():
    p = pd.read_csv(REV / "audit_by_phenotype.csv")
    r = pd.read_csv(REV / "audit_by_rule.csv")
    overall = r[(r.rule == "lowest_em") & (r.slug == "brennan")].n.iloc[0]
    assert p[p.slug == "brennan"].n.sum() == overall


def test_phenotype_deaths_add_up_to_the_overall_deaths():
    p = pd.read_csv(REV / "audit_by_phenotype.csv")
    r = pd.read_csv(REV / "audit_by_rule.csv")
    overall = r[(r.rule == "lowest_em") & (r.slug == "brennan")].deaths.iloc[0]
    assert p[p.slug == "brennan"].deaths.sum() == overall


def test_sofa_crosstab_total_equals_the_recorded_count():
    import json
    a = json.loads((REV / "audit.json").read_text())
    ct = pd.read_csv(REV / "sofa_crosstab.csv", index_col="brennan_estimate")
    assert list(ct.columns) == [f"official_{k}" for k in range(5)]
    assert int(ct.values.sum()) == a["sofa_n"]
    assert 0 < a["sofa_n"] <= a["brennan_lowest_em_n"]


def test_paired_blocks_record_their_stay_count():
    import json
    a = json.loads((REV / "audit.json").read_text())
    for k in ("brennan_minus_official", "brennan_minus_impute1", "brennan_minus_gcs_t", "impute1_minus_gcs_t"):
        assert a[f"paired_{k}_n"] == a["brennan_lowest_em_n"]


def test_sofa_official0_cells_match_the_crosstab():
    import json
    a = json.loads((REV / "audit.json").read_text())
    ct = pd.read_csv(REV / "sofa_crosstab.csv", index_col="brennan_estimate")
    assert a["sofa_official0_n"] == int(ct["official_0"].sum())
    assert a["sofa_brennan3_official0_n"] == int(ct.loc[3, "official_0"])
    assert a["sofa_brennan4_official0_n"] == int(ct.loc[4, "official_0"])


@pytest.mark.requires_study_files("output/intermediate/aim4_exams_fixed.parquet",
                                  "output/revision/official_first_day_gcs.parquet")
def test_sofa_and_apache_use_the_per_stay_first_day_minimum_not_the_selected_examination():
    """Recompute the Brennan first-day minimum over all examinations in the first 24 hours of each stay; the
    APACHE II points and the SOFA CNS cross-tabulation must equal it, and the single selected examination must not."""
    import json
    from lib_audit import add_totals, load_first_day, select_exam
    from lib_feasible import sofa_cns
    a = json.loads((REV / "audit.json").read_text())
    official = pd.read_parquet(REV / "official_first_day_gcs.parquet").set_index("stay_id")["gcs_min"]
    d1 = load_first_day(24)
    day_min = add_totals(d1).groupby("stay_id")["tg_brennan"].min().rename("brennan_min").to_frame()
    day_min = day_min.join(official.rename("official_min"), how="inner").dropna()
    assert len(day_min) == a["sofa_n"]
    assert abs((15 - day_min["brennan_min"]).mean() - a["apache_pts_brennan"]) < 1e-12
    assert abs((15 - day_min["official_min"]).mean() - a["apache_pts_official"]) < 1e-12
    ct = pd.crosstab(day_min["brennan_min"].map(sofa_cns), day_min["official_min"].map(sofa_cns))
    ct = ct.reindex(index=range(5), columns=range(5), fill_value=0)
    saved = pd.read_csv(REV / "sofa_crosstab.csv", index_col="brennan_estimate")
    assert (ct.values == saved.values).all() and int(saved.values.sum()) == a["sofa_n"]
    # the single lowest eye-plus-motor examination is a different quantity and gives a different mean
    sel = add_totals(select_exam(d1, "lowest_em")).set_index("stay_id")["tg_brennan"].reindex(day_min.index)
    assert abs((15 - sel).mean() - a["apache_pts_brennan"]) > 1e-6
