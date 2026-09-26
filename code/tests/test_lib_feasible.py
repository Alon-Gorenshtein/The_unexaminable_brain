import numpy as np

from lib_feasible import outside_sofa_bounds, outside_total_bounds, sofa_bounds, sofa_cns, total_bounds


def test_total_bounds_and_default_15():
    assert total_bounds(3) == (4, 8)
    assert bool(outside_total_bounds(9, 15)) and not bool(outside_total_bounds(10, 15))


def test_sofa_categories():
    assert [sofa_cns(g) for g in (15, 14, 13, 12, 10, 9, 6, 5, 3)] == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_sofa_bounds():
    assert sofa_bounds(3) == (3, 4) and sofa_bounds(10) == (0, 2)
    assert outside_sofa_bounds(3, 15) and not outside_sofa_bounds(10, 15)


def test_default_equal_to_the_upper_bound_is_never_outside():
    em = np.arange(2, 11)
    assert not outside_total_bounds(em, em + 5).any()
