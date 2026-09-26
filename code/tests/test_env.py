import sys

import numpy
import pandas
import scipy
import sklearn
import statsmodels


def test_environment_matches_supplement_s1_6():
    assert sys.version_info[:2] == (3, 9)
    assert sklearn.__version__.startswith("1.6")
    assert pandas.__version__.startswith("2.3")
    assert numpy.__version__.startswith("2.0")
    assert statsmodels.__version__.startswith("0.14")
    assert scipy.__version__.startswith("1.13")
