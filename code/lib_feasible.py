"""Reference-free bounds on the total GCS when the verbal component is unobservable.
The verbal score is 1 to 5, so the total lies between eye+motor+1 and eye+motor+5."""


def total_bounds(em):
    return em + 1, em + 5


def sofa_cns(g):
    if g >= 15:
        return 0
    if g >= 13:
        return 1
    if g >= 10:
        return 2
    if g >= 6:
        return 3
    return 4


def sofa_bounds(em):
    lo, hi = total_bounds(em)
    return sofa_cns(hi), sofa_cns(lo)


def outside_total_bounds(em, total):
    lo, hi = total_bounds(em)
    return (total < lo) | (total > hi)


def outside_sofa_bounds(em, total):
    a, b = sofa_bounds(em)
    d = sofa_cns(total)
    return d < a or d > b
