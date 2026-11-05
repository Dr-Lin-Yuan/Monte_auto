"""Tests for mie_toolkit.py (single-sphere Mie, size correction, channel split, near field)
and a smoke test for mie_tab.py.

Run from the repository root with either
    python3 -m pytest tests/
or
    python3 tests/test_mie_toolkit.py
Benchmarks: mie_toolkit_research.md section E / spec amendment 8.
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import mie_toolkit as mt  # noqa: E402

DIEL = REPO_ROOT / "dielectric_permittivity"
N_WATER = 1.333  # benchmark medium (research notes E)


def _jc(metal):
    return mt.load_epsilon_table(DIEL / f"JohnsonChristy1972_{metal}_nk.txt")


def _spec(metal, D, lam, n_m=N_WATER, size_correction=True, A=0.75):
    E, eps = _jc(metal)
    d = mt.DRUDE_DEFAULTS[metal]
    return mt.compute_spectrum(E, eps, lam, D, n_m, size_correction, d["omega_p_eV"], d["gamma_b_eV"],
                               d["v_F_m_s"], A, d["E_ib_eV"])


def _absE(m, x, X, Y, Z):
    ex, ey, ez = mt.near_field_points(m, x, X, Y, Z)
    return np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)


# ---------------------------------------------------------------- Mie benchmarks (fixed inputs)
def test_bohren_huffman_dielectric_benchmark():
    q = mt.mie_efficiencies(1.55, 2 * np.pi * 0.525 / 0.6328)
    assert abs(q["Q_ext"][0] - 3.10543) < 5e-6 and abs(q["Q_sca"][0] - 3.10543) < 5e-6


def test_wiscombe_absorbing_benchmark():
    q = mt.mie_efficiencies(1.5 + 1j, 1.0)
    assert abs(q["Q_ext"][0] - 2.336321) < 1e-6
    assert abs(q["Q_sca"][0] - 0.6634538) < 1e-7


def test_metal_regression_benchmark():
    # 100 nm Au in water at 532 nm with fixed n_Au = 0.5415 + 2.2355i (scattnlay = miepython = ours)
    m = (0.5415 + 2.2355j) / N_WATER
    x = 2 * np.pi * N_WATER * 50 / 532
    assert abs(m - (0.406227 + 1.677044j)) < 1e-6 and abs(x - 0.787170) < 1e-6
    q = mt.mie_efficiencies(m, x)
    assert abs(q["Q_ext"][0] - 5.139529) < 2e-6
    assert abs(q["Q_sca"][0] - 2.488916) < 2e-6
    assert abs(q["Q_abs"][0] - 2.650614) < 3e-6
    pts = {(1.0001, 0, 0): 3.99745, (0.9999, 0, 0): 1.59775, (0, 0, 1.0001): 1.78048,
           (0, 1.0001, 0): 1.53468, (2, 0, 0): 0.75667, (0, 0, 0): 1.08856, (0.5, 0, 0.5): 1.41778}
    P = np.array(list(pts))
    got = _absE(m, x, P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(got - np.array(list(pts.values())))) < 2e-5, got


def test_rayleigh_limit():
    eps, em = -4 + 0.5j, N_WATER ** 2
    m, x = np.sqrt(eps / em), 1e-3
    alpha = (m ** 2 - 1) / (m ** 2 + 2)
    q = mt.mie_efficiencies(m, x)
    assert abs(q["Q_abs"][0] / (4 * x * alpha.imag) - 1) < 1e-3
    assert abs(q["Q_sca"][0] / (8 / 3 * x ** 4 * abs(alpha) ** 2) - 1) < 1e-3


def test_extinction_paradox():
    for m in (1.5 + 0.1j, 1.33 + 0.01j):
        q = mt.mie_efficiencies(m, 1000.0)
        assert abs(q["Q_ext"][0] - 2.0) < 0.03, q["Q_ext"]


# ---------------------------------------------------------------- full JC chain, 100 nm Au in water
def test_au100_water_jc_chain():
    """PCHIP-in-eps chain (production). The published numbers used linear n,k interpolation;
    the JC table is sparse (0.12 eV), so Q differs by ~0.7 % -> 1 % tolerance on Q."""
    E, eps = _jc("Au")
    for lam, q_ref, emax_ref, th_ref in ((532.0, 5.140, 4.83, 61.0), (570.0, 6.94, 6.85, 71.0)):
        e = mt.interpolate_epsilon(E, eps, np.array([mt.HC_EV_NM / lam]))
        m, x = np.sqrt(e) / N_WATER, 2 * np.pi * N_WATER * 50 / lam
        q = mt.mie_efficiencies(m, x)
        assert abs(q["Q_ext"][0] / q_ref - 1) < 0.01, (lam, q["Q_ext"])
        emax, th, ph = mt.surface_max_field(m, x)
        assert abs(emax - emax_ref) < 0.05, (lam, emax)
        assert abs(th - th_ref) < 2.0 and ph == 0.0, (th, ph)


def test_au100_linear_nk_reproduces_published_numbers():
    """Same benchmark with the interpolation it was computed with (linear n,k in eV): tight match."""
    E, eps = _jc("Au")
    N = np.sqrt(eps)
    Eq = mt.HC_EV_NM / 532.0
    n_au = np.interp(Eq, E, N.real) + 1j * np.interp(Eq, E, N.imag)
    m, x = n_au / N_WATER, 2 * np.pi * N_WATER * 50 / 532.0
    assert abs(mt.mie_efficiencies(m, x)["Q_ext"][0] - 5.1397) < 5e-4
    assert abs(mt.surface_max_field(m, x)[0] - 4.83) < 0.005


# ---------------------------------------------------------------- size correction + channels
def test_eps_sc_reduces_to_eps_tab_without_surface_term():
    E, eps = _jc("Au")
    d = mt.DRUDE_DEFAULTS["Au"]
    assert np.array_equal(mt.size_corrected_epsilon(E, eps, d["omega_p_eV"], d["gamma_b_eV"], 0.0), eps)
    s = _spec("Au", 20, np.arange(300, 901, 5.0), size_correction=False)
    assert np.all(s["gamma_s_eV"] == 0) and np.allclose(s["eps_sc"], s["eps_tab"])
    assert np.all(s["f_Landau"] == 0)


def test_im_eps_sc_ge_im_eps_tab():
    lam = np.arange(300, 901, 1.0)
    for metal in ("Au", "Ag", "Cu"):
        for D in (5, 10, 100):
            s = _spec(metal, D, lam)
            assert np.all(s["eps_sc"].imag >= s["eps_tab"].imag), (metal, D)
            assert np.all(s["gamma_s_eV"] > 0)


def test_channel_weights_sum_to_one_and_bounded():
    lam = np.arange(300, 901, 1.0)
    for metal in ("Au", "Ag", "Cu"):
        for D in (10, 100):
            s = _spec(metal, D, lam)
            f = np.stack([s[k] for k in ("f_interband", "f_Landau", "f_bulk", "f_residual")])
            assert np.all(f >= 0) and np.all(f <= 1), (metal, D)
            assert np.max(np.abs(f.sum(0) - 1)) < 1e-12, (metal, D)
            sig = sum(s[k] for k in ("sigma_interband", "sigma_Landau", "sigma_bulk", "sigma_residual"))
            assert np.allclose(sig, s["sigma_abs"], rtol=1e-12, atol=0)
            assert np.allclose(s["eta_HE"], s["f_interband"] + s["f_Landau"])


def test_interband_gating_ag_lspr_is_residual():
    s = _spec("Ag", 20, np.arange(300, 901, 1.0))
    i = mt.peak_index(s["Q_abs"])
    assert s["E_eV"][i] < 3.9 and s["f_interband"][i] == 0 and s["f_residual"][i] > 0


def test_govorov_landau_fraction_cross_check():
    lam = np.arange(480, 700.01, 0.5)
    for D, ref in ((5, 0.391), (10, 0.245), (20, 0.140), (40, 0.077), (100, 0.041)):
        s = _spec("Au", D, lam)
        i = int(np.argmax(s["Q_abs"]))
        assert abs(s["f_Landau"][i] - ref) < 0.01, (D, s["f_Landau"][i])


def test_field_weighted_gamma_reduces_to_kreibig_for_small_spheres():
    E, eps = _jc("Au")
    d = mt.DRUDE_DEFAULTS["Au"]
    s = mt.compute_spectrum(E, eps, np.arange(480, 700.01, 1.0), 10, N_WATER, True, d["omega_p_eV"],
                            d["gamma_b_eV"], d["v_F_m_s"], 0.75, d["E_ib_eV"], field_weighted=True)
    ratio = s["gamma_s_eV"] / mt.surface_gamma(0.75, d["v_F_m_s"], 5.0)
    assert np.all(np.abs(ratio - 1) < 0.01)


# ---------------------------------------------------------------- near field
def test_far_field_tends_to_incident():
    s = _spec("Au", 20, np.arange(400, 800.01, 1.0))
    i = mt.peak_index(s["Q_abs"])
    ph = np.linspace(0, 2 * np.pi, 73)
    for X, Y, Z in ((10 * np.cos(ph), 0 * ph, 10 * np.sin(ph)), (10 * np.cos(ph), 10 * np.sin(ph), 0 * ph)):
        assert np.max(np.abs(_absE(s["m"][i], s["x"][i], X, Y, Z) - 1)) < 0.02


def test_tangential_continuity_and_normal_D_on_equator():
    s = _spec("Au", 100, np.array([532.0]), size_correction=False)
    m, x = s["m"][0], s["x"][0]
    ph = np.linspace(0, 2 * np.pi, 37)
    out = {}
    for r in (1 - 1e-4, 1 + 1e-4):
        ex, ey, ez = mt.near_field_points(m, x, r * np.cos(ph), r * np.sin(ph), 0 * ph)
        e_r = ex * np.cos(ph) + ey * np.sin(ph)
        e_phi = -ex * np.sin(ph) + ey * np.cos(ph)
        e_theta = -ez  # theta_hat = -z_hat on the equator
        out[r > 1] = (e_r, e_phi, e_theta)
    (ri, pi_, ti), (ro, po, to) = out[False], out[True]
    tan_jump = np.sqrt(np.abs(pi_ - po) ** 2 + np.abs(ti - to) ** 2)
    tan_ref = np.max(np.sqrt(np.abs(po) ** 2 + np.abs(to) ** 2))
    assert np.max(tan_jump) / tan_ref < 0.02
    assert np.max(np.abs(m ** 2 * ri - ro)) / np.max(np.abs(ro)) < 0.02


def test_dipolar_pattern_20nm_au():
    s = _spec("Au", 20, np.arange(400, 800.01, 1.0))
    i = mt.peak_index(s["Q_abs"])
    emax, th, ph = mt.surface_max_field(s["m"][i], s["x"][i])
    assert 500 < s["lam_nm"][i] < 540 and abs(th - 90) < 5 and ph == 0.0
    nf = mt.near_field_map(s["m"][i], s["x"][i], s["R_nm"], "xz", 2.5, 101)
    j, k = np.unravel_index(np.argmax(np.where(nf["inside"], 0, nf["E_norm"])), nf["E_norm"].shape)
    assert abs(nf["Y"][j, k]) < 0.3 * s["R_nm"] < abs(nf["X"][j, k])  # hot spot on the x (E0) axis


def test_quasistatic_near_field_analytic():
    eps, em = -4 + 0.5j, N_WATER ** 2
    m, x = np.sqrt(eps / em), 1e-3
    a = (eps - em) / (eps + 2 * em)
    got = _absE(m, x, np.array([1.000001, 0.0, 0.3]), np.zeros(3), np.array([0.0, 1.000001, 0.0]))
    ref = np.array([abs(1 + 2 * a), abs(1 - a), abs(3 * em / (eps + 2 * em))])
    assert np.max(np.abs(got / ref - 1)) < 1e-4


def test_near_field_planes():
    s = _spec("Au", 50, np.array([530.0]))
    for plane in ("xz", "yz", "xy"):
        nf = mt.near_field_map(s["m"][0], s["x"][0], s["R_nm"], plane, 2.5, 41)
        assert nf["E_norm"].shape == (41, 41) and np.all(np.isfinite(nf["E_norm"]))
        assert abs(nf["X"][0, 0] + 2.5 * s["R_nm"]) < 1e-9


# ---------------------------------------------------------------- loader
def test_loader_formats_agree():
    E, eps = _jc("Au")
    lam_nm = mt.HC_EV_NM / E
    n = np.sqrt(eps)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p_nm = tmp / "a.txt"
        np.savetxt(p_nm, np.c_[lam_nm, n.real, n.imag], delimiter=",")  # comma, no header, nm
        p_ev = tmp / "b.txt"
        rows = np.r_[[[0.0, 8.9e5, 0.48]], np.c_[E, eps.real, eps.imag]]  # Rahm-style E = 0 row
        np.savetxt(p_ev, rows[::-1], header="Col 1: Energy (eV)\nCol 2: Real dielectric function")
        E1, eps1, info1 = mt.load_epsilon_table(p_nm, return_info=True)
        E2, eps2, info2 = mt.load_epsilon_table(p_ev, return_info=True)
    assert info1["format"] == "lambda_nm_nk" and info2["format"] == "E_eV_eps" and info2["n_dropped"] == 1
    assert np.allclose(E1, E) and np.allclose(eps1, eps) and np.allclose(E2, E) and np.allclose(eps2, eps)
    _, _, info3 = mt.load_epsilon_table(DIEL / "JohnsonChristy1972_Au_nk.txt", return_info=True)
    assert info3["format"] == "lambda_um_nk"
    with np.testing.assert_raises(ValueError):
        mt.interpolate_epsilon(E, eps, np.array([E[-1] + 0.1]))


def test_repo_tables_load_and_restrict_range():
    for p in sorted(DIEL.glob("*.txt")):
        E, eps = mt.load_epsilon_table(p)
        assert np.all(np.diff(E) > 0) and np.all(np.isfinite(eps)), p.name
    p = DIEL / "dielectric_function_Pd_0.47_Au_99.53.txt"
    if p.exists():
        E, eps = mt.load_epsilon_table(p)
        s = mt.compute_spectrum(E, eps, np.arange(300, 901, 1.0), 10, 1.33)
        assert s["n_dropped"] == 10 and s["lam_nm"][0] == 310.0


# ---------------------------------------------------------------- UI module + timing
def test_mie_tab_import_and_figure_builders():
    import mie_tab  # noqa: F401  (needs streamlit; render() is not executed)

    s = _spec("Au", 50, np.arange(300, 901, 1.0), n_m=1.33)
    i = mt.peak_index(s["Q_abs"])
    nf = mt.near_field_map(s["m"][i], s["x"][i], s["R_nm"], "xz", 2.5, 41)
    for fig in (mie_tab.fig_spectra(s["lam_nm"], s["Q_ext"], s["Q_sca"], s["Q_abs"], s["R_nm"], True),
                mie_tab.fig_hot_electron(s["lam_nm"], s, 1.9),
                mie_tab.fig_near_field(nf["X"], nf["Y"], nf["E_norm"], nf["R"], nf["xlabel"], nf["ylabel"], True)):
        assert fig.axes
    csv = mie_tab.spectrum_csv(s, ["test"]).splitlines()
    assert csv[1].count(",") == 15 and len(csv) == 2 + s["lam_nm"].size
    assert callable(mie_tab.render)


def test_timing():
    E, eps = _jc("Au")
    lam = np.linspace(300, 900, 601)
    t0 = time.perf_counter()
    s = mt.compute_spectrum(E, eps, lam, 50, 1.33)
    t1 = time.perf_counter()
    i = mt.peak_index(s["Q_abs"])
    mt.near_field_map(s["m"][i], s["x"][i], s["R_nm"], "xz", 2.5, 201)
    t2 = time.perf_counter()
    print(f"TIMING spectrum 601 pts: {1e3 * (t1 - t0):.1f} ms; near field 201x201: {1e3 * (t2 - t1):.1f} ms")
    assert t1 - t0 < 1.0 and t2 - t1 < 1.0


def _run_all():
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    n_fail = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - n_fail} passed, {n_fail} failed")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
