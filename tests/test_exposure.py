import numpy as np
import pytest

from catbond.exposure import county_exposure, household_counts


def test_household_table_matches_source_total():
    hh = household_counts()
    assert len(hh) == 67 and hh["county"].is_unique
    assert hh["households_2020"].sum() == 8_529_067  # Florida total printed in the BEBR table
    assert hh.set_index("county").loc["Miami-Dade", "households_2020"] == 967_414


def test_county_exposure_real(florida):
    ex = county_exposure(florida, tiv_usd=100e9)
    assert len(ex) == 67
    assert ex["tiv_usd"].sum() == pytest.approx(100e9)
    md = ex.set_index("county").loc["Miami-Dade"]
    assert md["share"] == pytest.approx(967_414 / 8_529_067)
    assert md["fips"] == "12086"
    assert ex["lon"].between(-87.7, -79.9).all() and ex["lat"].between(24.4, 31.1).all()


def test_county_exposure_rejects_unmatched(square_region):
    with pytest.raises(ValueError, match="without a match"):
        county_exposure(square_region)
    assert np.isfinite(household_counts()["households_2020"]).all()
