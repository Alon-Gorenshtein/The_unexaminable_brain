import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pytest

from config import INT, REV, TAB

pytestmark = pytest.mark.requires_study_files(
    "output/revision/selection_model.json",
    "output/revision/selection_model_corrected.csv",
    "output/revision/selection_model_legacy.csv",
    "output/revision/smd_legacy.csv",
    "output/v1_submitted/table_selection_model.csv",
    "output/v1_submitted/table_smd.csv",
    "output/intermediate/infusions.parquet")

V1 = TAB.parent / "v1_submitted"


def test_legacy_population_reproduces_the_submitted_selection_model_and_smds():
    new = pd.read_csv(REV / "selection_model_legacy.csv").set_index("predictor")
    old = pd.read_csv(V1 / "table_selection_model.csv").set_index("predictor")
    assert set(old.index) <= set(new.index)
    for k in old.index:
        for c in ("OR", "lo", "hi"):
            assert abs(new.loc[k, c] - old.loc[k, c]) < 0.011, (k, c)
    ns = pd.read_csv(REV / "smd_legacy.csv").set_index("variable")
    os_ = pd.read_csv(V1 / "table_smd.csv").set_index("variable")
    for k in os_.index:
        assert abs(ns.loc[k, "smd"] - os_.loc[k, "smd"]) < 0.011, k
        assert abs(ns.loc[k, "excluded"] - os_.loc[k, "excluded"]) < 0.06, k


def test_corrected_population_has_the_same_predictors():
    a = pd.read_csv(REV / "selection_model_corrected.csv")
    b = pd.read_csv(REV / "selection_model_legacy.csv")
    assert list(a.predictor) == list(b.predictor)


def test_corrected_ventilation_or_is_recomputed_from_the_corrected_frame_not_read_from_stale_files():
    """Refit the ventilation odds ratio here, through a different code path (a model formula rather than a
    hand-built design matrix), from the corrected selected-examination frame, and require the saved CSV and
    JSON to agree with it. A stale or legacy-population file cannot pass."""
    from lib_audit import add_stay_covariates, selected_frames

    inf = pd.read_parquet(INT / "infusions.parquet", columns=["stay_id", "cls"])
    sed_ids = set(inf.loc[inf["cls"] == "sedative", "stay_id"])
    w = add_stay_covariates(selected_frames(24)["lowest_em"])
    w["vent"], w["vaso"] = w["vent_stay"], w["vaso_stay"]
    w["sed"] = w["stay_id"].isin(sed_ids).astype(int)
    w["na"] = w["nonassess"].astype(int)
    w["died"] = w["y"].astype(int)
    assert len(w) == 14192
    fit = smf.logit("na ~ C(phenotype, Treatment('AIS')) + age + vent + sed + vaso + died", data=w).fit(disp=0)
    direct_or = float(np.exp(fit.params["vent"]))

    saved = pd.read_csv(REV / "selection_model_corrected.csv").set_index("predictor")
    js = json.loads((REV / "selection_model.json").read_text())
    assert abs(saved.loc["vent", "OR"] - direct_or) < 1e-6
    assert abs(js["corrected_or_vent"] - direct_or) < 1e-6
    # the corrected and legacy populations must give different answers, or one file is a copy of the other
    assert abs(js["corrected_or_vent"] - js["legacy_or_vent"]) > 0.01
