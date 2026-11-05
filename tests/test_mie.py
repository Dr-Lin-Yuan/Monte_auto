"""Tests for the vectorized Mie solver, full-Mie support particles and the legacy loss.

Run from the repository root with either
    python3 -m pytest tests/test_mie.py
or
    python3 tests/test_mie.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import sim_core  # noqa: E402
from sim_core import (  # noqa: E402
    compute_cu_mie_cross_sections,
    diameter_distribution_weights,
    get_cu_mie_lut,
    load_dielectric_function,
    mie_ab,
    mie_ceo2_approx,
    mie_sphere_efficiencies,
    mie_tio2_approx,
    truncated_normal,
)

CU_EPS_FILE = REPO_ROOT / "dielectric_permittivity" / "dielectric_function_Cu_smoothed_1p55to3p54eV.txt"
EXP_FILE = REPO_ROOT / "Cu_exp.txt"

# Default CeO2 k-model parameters (same values as fit_pipeline.default_base_params()).
CEO2_K_DEFAULTS = dict(
    uv_cut_nm=525.0,
    uv_amp=0.10,
    uv_decay_nm=40.0,
    vis_peak_nm=525.0,
    vis_width_nm=90.0,
    vis_k0=4e-3,
)


def _reference_cross_sections(
    diameters_nm,
    file_eps,
    n_medium=1.75,
    lambda_min=350.0,
    lambda_max=800.0,
    lambda_step=1.0,
):
    """Copy of the OLD double-loop compute_cu_mie_cross_sections (scalar mie_ab per pair)."""
    lambdas_nm = np.arange(lambda_min, lambda_max + lambda_step, lambda_step, dtype=np.float64)
    r_nm = np.asarray(diameters_nm, dtype=np.float64) / 2.0

    df_eps = load_dielectric_function(file_eps)
    eps1 = np.interp(lambdas_nm, df_eps["lambda_nm"], df_eps["eps1"])
    eps2 = np.interp(lambdas_nm, df_eps["lambda_nm"], df_eps["eps2"])
    eps_complex = eps1 + 1j * eps2
    n_particle = np.sqrt(eps_complex)
    m_rel = n_particle / n_medium

    k_medium = 2.0 * np.pi * n_medium / lambdas_nm
    n_part = r_nm.size
    n_lambda = lambdas_nm.size

    cext = np.zeros((n_part, n_lambda), dtype=np.float64)
    csca = np.zeros_like(cext)

    for j, _ in enumerate(lambdas_nm):
        m = m_rel[j]
        k_j = k_medium[j]
        x_all = k_j * r_nm
        n_max_all = np.ceil(2 + x_all + 4.05 * x_all ** (1 / 3)).astype(int)

        for i, (x, n_max) in enumerate(zip(x_all, n_max_all)):
            an, bn = mie_ab(m, x, n_max)
            ns = np.arange(1, n_max + 1)

            sca_sum = np.sum((2 * ns + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2))
            ext_sum = np.sum((2 * ns + 1) * np.real(an + bn))

            csca[i, j] = (2 * np.pi / k_j ** 2) * sca_sum
            cext[i, j] = (2 * np.pi / k_j ** 2) * ext_sum

    cext, csca, cabs = sim_core.sanitize_cross_sections(cext, csca, "Cu")
    return lambdas_nm, cext, csca, cabs


def _output_root():
    base = os.environ.get("MONTE_AUTO_TEST_OUTPUT_ROOT")
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="pipeline_smoke_", dir=base))
    return Path(tempfile.mkdtemp(prefix="monte_auto_pipeline_smoke_"))


def test_vectorized_cu_mie_matches_scalar_reference():
    diameters = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    lambda_min, lambda_max = 400.0, 800.0
    lambda_step = (lambda_max - lambda_min) / 49.0  # 50 wavelengths
    kwargs = dict(
        file_eps=str(CU_EPS_FILE),
        n_medium=1.75,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_step=lambda_step,
    )
    wl_ref, cext_ref, csca_ref, cabs_ref = _reference_cross_sections(diameters, **kwargs)
    wl_new, cext_new, csca_new, cabs_new = compute_cu_mie_cross_sections(diameters, **kwargs)

    assert cext_new.shape == (5, 50), cext_new.shape
    np.testing.assert_array_equal(wl_new, wl_ref)
    for ref, new, label in ((cext_ref, cext_new, "cext"), (csca_ref, csca_new, "csca"), (cabs_ref, cabs_new, "cabs")):
        rel = np.max(np.abs(new - ref) / np.abs(ref))
        assert rel < 1e-8, f"{label} max relative error {rel:.3e}"


def test_mie_sphere_efficiencies_rayleigh_limit():
    x = 0.01
    m = 1.5
    wl = 500.0
    n_medium = 1.0
    d = x * wl / (np.pi * n_medium)
    qsca, qabs, qext = mie_sphere_efficiencies(d, np.array([wl]), m + 0j, n_medium)
    q_rayleigh = (8.0 / 3.0) * x**4 * abs((m**2 - 1.0) / (m**2 + 2.0)) ** 2
    assert abs(qsca[0] / q_rayleigh - 1.0) < 0.01, (qsca[0], q_rayleigh)
    assert qabs[0] < 1e-12 * qsca[0] + 1e-30


def test_mie_sphere_efficiencies_extinction_paradox():
    x = 200.0
    wl = 500.0
    n_medium = 1.0
    d = x * wl / (np.pi * n_medium)
    qsca, qabs, qext = mie_sphere_efficiencies(d, np.array([wl]), 1.33 + 0j, n_medium)
    assert abs(qext[0] - 2.0) / 2.0 < 0.10, qext[0]
    assert np.isfinite(qsca[0]) and np.isfinite(qabs[0])


def test_ceo2_support_cross_sections_are_smooth():
    diameters = np.arange(60.0, 151.0, 1.0)
    wl = np.array([550.0])
    csca = np.empty(diameters.size)
    cabs = np.empty(diameters.size)
    for i, d in enumerate(diameters):
        s, a = mie_ceo2_approx(d, wl, n_medium=1.75, **CEO2_K_DEFAULTS)
        csca[i] = s[0]
        cabs[i] = a[0]
    cext = csca + cabs
    for arr, label in ((cext, "cext"), (csca, "csca"), (cabs, "cabs")):
        assert np.all(np.isfinite(arr)) and np.all(arr > 0.0), label
        jumps = np.abs(np.diff(arr)) / arr[:-1]
        assert np.max(jumps) <= 0.20, f"{label} max consecutive relative jump {np.max(jumps):.3f}"


def test_tio2_support_returns_three_arrays():
    wl = np.linspace(400.0, 800.0, 41)
    csca, cabs, cext = mie_tio2_approx(80.0, wl, n_medium=1.75)
    assert csca.shape == cabs.shape == cext.shape == wl.shape
    np.testing.assert_allclose(cext, csca + cabs, rtol=1e-12)
    assert np.all(csca > 0.0) and np.all(cabs >= 0.0)


def test_lut_ensemble_matches_direct_sampling():
    params = {
        "file_eps": str(CU_EPS_FILE),
        "n_medium": 1.75,
        "mie_lambda_min_nm": 400.0,
        "mie_lambda_max_nm": 800.0,
        "mie_lambda_step_nm": 5.0,
        "cu_lower_nm": 20.0,
        "cu_upper_nm": 40.0,
        "cu_mean_nm": 30.0,
        "cu_sigma_nm": 5.0,
        "mie_diameter_grid_size": 160,
    }
    lambdas_lut, d_grid, cext_lut, _, _ = get_cu_mie_lut(params)
    weights = diameter_distribution_weights(d_grid, 20.0, 40.0, 30.0, 5.0)
    cext_lut_mean = np.sum(weights[:, None] * cext_lut, axis=0)

    rng = np.random.default_rng(12345)
    d_samples = truncated_normal(20000, 20.0, 40.0, 30.0, 5.0, rng=rng)
    lambdas_dir, cext_dir, _, _ = compute_cu_mie_cross_sections(
        d_samples,
        str(CU_EPS_FILE),
        n_medium=1.75,
        lambda_min=400.0,
        lambda_max=800.0,
        lambda_step=5.0,
    )
    cext_dir_mean = np.mean(cext_dir, axis=0)

    np.testing.assert_array_equal(lambdas_lut, lambdas_dir)
    rel = np.max(np.abs(cext_lut_mean - cext_dir_mean) / cext_dir_mean)
    assert rel < 0.02, f"LUT vs direct ensemble <Cext> max relative difference {rel:.4f}"


def test_legacy_loss_has_baseline_term():
    from fit_agent import legacy_loss

    exp_wl, exp_fr = sim_core.load_experiment_txt(str(EXP_FILE))
    sim_wl = np.linspace(400.0, 800.0, 100)
    sim_y = np.exp(-((sim_wl - 580.0) ** 2) / (2.0 * 40.0**2)) + 0.1
    total, exp_interp_norm, details = legacy_loss(sim_wl, sim_y, exp_wl, exp_fr)
    assert np.isfinite(total)
    assert exp_interp_norm.shape == sim_wl.shape
    for key in ("baseline_loss", "sim_tail_mean", "exp_tail_mean"):
        assert key in details, key
    assert details["sim_tail_mean"] is not None and details["exp_tail_mean"] is not None
    delta = details["sim_tail_mean"] - details["exp_tail_mean"]
    expected = (delta / 0.20) ** 2 / (1.0 + abs(delta) / 0.20)
    assert abs(details["baseline_loss"] - expected) < 1e-12
    assert details["baseline_loss"] >= 0.0


def test_pipeline_smoke():
    from fit_pipeline import run_fit_pipeline

    output_root = _output_root()
    result = run_fit_pipeline(
        experiment_file=str(EXP_FILE),
        dielectric_file=str(CU_EPS_FILE),
        output_root=str(output_root),
        n_trials=2,
        particle_lower_nm=20.0,
        particle_upper_nm=40.0,
        mean_nm=30.0,
        sigma_nm=5.0,
        n_particles=20,
        n_wavelengths=10,
        n_photons=200,
        n_search_photons=100,
        simulation_mode="fast_hybrid_gpu",
        allow_cpu_fallback=True,
    )
    assert (output_root / "optuna_study.db").exists()
    meta_file = output_root / "logs" / "best_full_precision" / "meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert np.isfinite(meta["loss"])
    assert "baseline_loss" in meta["loss_details"]
    assert np.isfinite(result["best_value"])


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
