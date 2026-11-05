"""Parametric wind field: Holland (1980) profile, translation asymmetry, Kaplan-DeMaria decay.

Units: winds in kt (1 kt = 1.852 km/h = 0.514444 m/s), distances in km, time in hours.

**Profile.** Holland (1980) gradient-wind profile written in terms of the peak wind (the
cyclostrophic form, Coriolis term neglected):

    V_s(r) = V_m h(r),   h(r) = [ (R/r)^B exp(1 - (R/r)^B) ]^(1/2),

so h(R) = 1, h -> 0 as r -> 0 and as r -> infinity. Because the HURDAT2 peak wind is already a
1-minute 10-m surface wind (Landsea and Franklin 2013), the profile is used directly as a surface
wind profile; no gradient-to-surface reduction, gust factor or terrain roughness is applied.

**Radius of maximum wind and B.** Willoughby and Rahn (2004), equations 12.1 and 12.2 (fits to
493 quality-controlled aircraft sorties, 1977-2000; V_m in m/s, phi latitude in degrees):

    R = 46.29 exp(-0.0153 V_m + 0.0166 phi) km,      B = 0.886 + 0.0177 V_m - 0.0094 phi.

V_m in those fits is a flight-level wind; it is applied here to the HURDAT2 surface wind, which
is lower, so R is somewhat too large and B somewhat too small (a stated approximation). B is
clipped to [1.0, 2.5]. No random scatter is added to R; ``rmw_scale`` is a sensitivity knob.

**Translation asymmetry** (stated assumption, not taken from a paper): with translation speed
V_t and theta the angle between the storm motion and the cyclonic tangential wind direction at
the point (cos theta = c / r, c the signed cross-track distance, positive to the right of the
motion), the wind is

    V(r, theta) = h(r) max(V_m - V_t (1 - cos theta), 0),

so the peak V_m is reached at r = R on the right of the track and V_m - 2 V_t on the left.

**Inland decay.** Kaplan and DeMaria (1995): after landfall (t > 0 hours)

    V_m(t) = V_b + (R_f V_0 - V_b) exp(-alpha t),

V_b = 26.7 kt, R_f = 0.9, alpha = 0.095 per hour (the values for the US East and Gulf coasts in
the NCEP/NHC implementation ``decayo.f90``); before landfall V_m(t) = V_0. The track is a
straight line through the landfall point at constant heading and speed; decay continues if the
track re-emerges over water (no re-intensification).

**Peak wind at a site.** The maximum over time is taken over candidate track positions: the
landfall point itself; positions s_c + delta R along the track, where s_c is the along-track
coordinate of the site's closest approach and delta runs over ``DEFAULT_OFFSETS`` (-2 to 2 in
steps of 1/8); and, when the cross-track distance c is below R, the two eyewall passages
s_c +/- sqrt(R^2 - c^2). The test suite compares this with a dense grid.
"""

from __future__ import annotations

import numpy as np

from catbond.geo import local_xy

KT_PER_MS = 1.0 / 0.514444
KMH_PER_KT = 1.852

KD_VB_KT = 26.7
KD_R = 0.9
KD_ALPHA_PER_H = 0.095

DEFAULT_OFFSETS = np.linspace(-2.0, 2.0, 33)  # steps of R / 8


def holland_shape(r_km: np.ndarray, rmw_km: np.ndarray, b: np.ndarray) -> np.ndarray:
    """h(r) = [(R/r)^B exp(1 - (R/r)^B)]^(1/2); r is floored at 1e-3 km."""
    x = (np.asarray(rmw_km) / np.maximum(np.asarray(r_km, dtype=float), 1e-3)) ** np.asarray(b)
    return np.sqrt(x * np.exp(1.0 - x))


def rmw_willoughby_rahn(vmax_kt: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Radius of maximum wind (km), Willoughby and Rahn (2004) eq. 12.1, V in m/s."""
    v_ms = np.asarray(vmax_kt, dtype=float) / KT_PER_MS
    return 46.29 * np.exp(-0.0153 * v_ms + 0.0166 * np.asarray(lat_deg, dtype=float))


def holland_b_willoughby_rahn(vmax_kt: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Holland B, Willoughby and Rahn (2004) eq. 12.2 (V in m/s), clipped to [1.0, 2.5]."""
    v_ms = np.asarray(vmax_kt, dtype=float) / KT_PER_MS
    return np.clip(0.886 + 0.0177 * v_ms - 0.0094 * np.asarray(lat_deg, dtype=float), 1.0, 2.5)


def kaplan_demaria(v0_kt: np.ndarray, t_hours: np.ndarray, decay: bool = True) -> np.ndarray:
    """Peak wind (kt) t hours after landfall: V_b + (R_f V0 - V_b) exp(-alpha t) for t > 0."""
    v0 = np.asarray(v0_kt, dtype=float)
    t = np.asarray(t_hours, dtype=float)
    if not decay:
        return np.broadcast_to(v0, np.broadcast_shapes(v0.shape, t.shape)).copy()
    after = KD_VB_KT + (KD_R * v0 - KD_VB_KT) * np.exp(-KD_ALPHA_PER_H * np.maximum(t, 0.0))
    return np.where(t > 0, after, v0)


def peak_site_winds(
    lon0: np.ndarray,
    lat0: np.ndarray,
    heading_deg: np.ndarray,
    speed_kmh: np.ndarray,
    vmax_kt: np.ndarray,
    rmw_km: np.ndarray,
    b: np.ndarray,
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    offsets: np.ndarray = DEFAULT_OFFSETS,
    decay: bool = True,
) -> np.ndarray:
    """Peak 1-minute wind (kt) at each site for each event: array (n_events, n_sites).

    Event arrays have shape (n_events,); heading is the direction of motion in degrees clockwise
    from north; speed is the translation speed in km/h (floored at 1 km/h).
    """
    e = lambda a: np.asarray(a, dtype=float)[:, None, None]  # noqa: E731
    x, y = local_xy(
        np.asarray(site_lon)[None, :],
        np.asarray(site_lat)[None, :],
        np.asarray(lon0)[:, None],
        np.asarray(lat0)[:, None],
    )
    x, y = x[:, :, None], y[:, :, None]
    hd = np.deg2rad(e(heading_deg))
    ux, uy = np.sin(hd), np.cos(hd)
    s_c = x * ux + y * uy  # along-track coordinate of closest approach
    c = x * uy - y * ux  # signed cross-track distance, > 0 to the right of the motion
    rmw = e(rmw_km)
    eyewall = np.sqrt(np.maximum(rmw**2 - c**2, 0.0))
    s = np.concatenate(
        [
            s_c + np.asarray(offsets)[None, None, :] * rmw,
            np.zeros_like(s_c),
            s_c - eyewall,
            s_c + eyewall,
        ],
        axis=2,
    )
    a = s_c - s
    r = np.hypot(a, c)
    v = np.maximum(e(speed_kmh), 1.0)
    vm = kaplan_demaria(e(vmax_kt), s / v, decay=decay)
    vt = v / KMH_PER_KT
    cos_theta = c / np.maximum(r, 1e-3)
    wind = holland_shape(r, rmw, e(b)) * np.maximum(vm - vt * (1.0 - cos_theta), 0.0)
    return wind.max(axis=2)
