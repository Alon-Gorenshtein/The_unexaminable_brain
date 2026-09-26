import pandas as pd

import lib_eicu as E

NEAR_MISS = "surgery|pulmonary therapies|mechanical ventilation"
NIV = E.VENT_STEM + "|non-invasive ventilation"


def test_first_ventilation_offset_uses_stem_and_earliest():
    t = pd.DataFrame({"patientunitstayid": [1, 1, 1, 1, 2, 3],
                      "treatmentoffset": [300, 100, 50, 5, 10, 20],
                      "treatmentstring": [E.VENT_STEM, E.VENT_STEM,
                                          "pulmonary|ventilation and oxygenation|oxygen therapy (40% to 60%)",
                                          NEAR_MISS,
                                          "cardiovascular|shock|vasopressors",
                                          NEAR_MISS]})
    s = E.first_ventilation_offset(t)
    assert s.to_dict() == {1: 100}          # the near-miss stem (offset 5, stay 3) must not count


def test_first_ventilation_offset_invasive_only_and_first_row_not_invasive():
    t = pd.DataFrame({"patientunitstayid": [1, 1, 2, 3, 3, 4],
                      "treatmentoffset": [10, 50, 30, 40, 40, 60],
                      "treatmentstring": [NIV, E.VENT_STEM,          # stay 1: NIV row comes first
                                          NIV,                        # stay 2: only NIV
                                          NIV, E.VENT_STEM,           # stay 3: tie at 40, generic row present
                                          E.VENT_STEM]})              # stay 4: generic only
    assert E.first_ventilation_offset(t).to_dict() == {1: 10, 2: 30, 3: 40, 4: 60}
    assert E.first_ventilation_offset(t, invasive_only=True).to_dict() == {1: 50, 3: 40, 4: 60}
    assert sorted(E.first_row_not_invasive(t)) == [1, 2]


def test_window_bounds_and_sign():
    tv = pd.Series({1: 100})
    gcs = pd.DataFrame({"stay": [1] * 5, "offset": [99, 100, 460, 461, 40], "total": [15, 10, 11, 15, 15]})
    r = E.records_in_window(gcs, tv, 0, 360)
    assert sorted(r.offset) == [100, 460]           # 99 is one minute before; 461 is one minute past
    pre = E.records_in_window(gcs, tv, -360, -1)
    assert sorted(pre.offset) == [40, 99]


def test_pre_window_lower_and_upper_bounds_exact():
    tv = pd.Series({1: 1000})
    gcs = pd.DataFrame({"stay": [1] * 4, "offset": [639, 640, 999, 1000], "total": [3, 4, 5, 6]})
    pre = E.records_in_window(gcs, tv, *E.WINDOWS["pre_6h"][:2])
    assert sorted(pre.rel) == [-360, -1]            # rel -361 and 0 are outside, -360 and -1 are inside
    assert sorted(pre.total) == [4, 5]


def test_first_record_per_stay_takes_earliest():
    r = pd.DataFrame({"stay": [1, 1, 2], "offset": [200, 120, 50], "total": [10, 15, 11]})
    f = E.first_record_per_stay(r)
    assert f.set_index("stay").offset.to_dict() == {1: 120, 2: 50}
    assert f.set_index("stay").total.to_dict() == {1: 15, 2: 11}


def test_last_record_per_stay_takes_latest_with_its_own_total():
    r = pd.DataFrame({"stay": [1, 1, 1, 2], "offset": [120, 300, 200, 50], "total": [15, 9, 12, 11]})
    f = E.last_record_per_stay(r)
    assert f.set_index("stay").offset.to_dict() == {1: 300, 2: 50}
    assert f.set_index("stay").total.to_dict() == {1: 9, 2: 11}


def test_window_records_pre_is_nearest_and_earliest_variant_is_furthest():
    tv = pd.Series({1: 1000})
    gcs = pd.DataFrame({"stay": [1] * 5, "offset": [650, 900, 990, 1010, 1300], "total": [15, 14, 4, 5, 6]})
    near = E.window_records(gcs, tv, "pre_6h")
    far = E.window_records(gcs, tv, "pre_6h_earliest")
    post = E.window_records(gcs, tv, "w0_6h")
    assert near.set_index("stay").total.to_dict() == {1: 4}          # offset 990, ten minutes before
    assert far.set_index("stay").total.to_dict() == {1: 15}          # offset 650, the record furthest away
    assert post.set_index("stay").total.to_dict() == {1: 5}          # offset 1010, nearest after the anchor


def test_load_gcs_coerces_drops_and_filters(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text("1\t100\t15\n1\t110\t\n2\t50\tabc\n2\t60\t2\n2\t70\t16\n3\t5\t3\n")
    raw = E.load_gcs_raw(p)
    assert len(raw) == 6 and int(raw.total.isna().sum()) == 2
    g = E.load_gcs(p)
    assert sorted(zip(g.stay, g.offset, g.total)) == [(1, 100, 15), (3, 5, 3)]


def test_wilson():
    p, lo, hi = E.wilson_pct(50, 100)
    assert p == 50 and lo < 50 < hi
