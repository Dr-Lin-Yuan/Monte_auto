import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import spherical_jn, spherical_yn
from scipy.stats import truncnorm

from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)

try:
    if cuda.is_available():
        cuda.select_device(0)
except Exception:
    pass


def truncated_normal(n_samples, lower, upper, mean, sigma, batch_size=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_samples = int(n_samples)
    lower = float(lower)
    upper = float(upper)
    mean = float(mean)
    sigma = float(sigma)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if upper < lower:
        raise ValueError("upper diameter bound must be >= lower bound")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if upper == lower:
        return np.full(n_samples, lower, dtype=np.float64)

    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    return truncnorm.rvs(a, b, loc=mean, scale=sigma, size=n_samples, random_state=rng).astype(np.float64)


def load_experiment_txt(file_exp):
    try:
        data = np.loadtxt(file_exp)
    except Exception:
        data = np.loadtxt(file_exp, delimiter=",")
    wl = data[:, 0].astype(float)
    fr = data[:, 1].astype(float)
    mask = np.isfinite(wl) & np.isfinite(fr)
    wl = wl[mask]
    fr = fr[mask]
    order = np.argsort(wl)
    return wl[order], fr[order]


def load_dielectric_function(file_eps):
    df_eps = pd.read_csv(
        file_eps,
        sep=r"[\s,]+",
        header=None,
        names=["E_eV", "eps1", "eps2"],
        comment="#",
        engine="python",
    )
    df_eps = df_eps.apply(pd.to_numeric, errors="coerce")
    df_eps = df_eps.dropna().reset_index(drop=True)
    if df_eps.empty:
        raise ValueError(f"No valid dielectric rows found in {file_eps}")
    df_eps["lambda_nm"] = 1240.0 / df_eps["E_eV"]
    df_eps = df_eps.sort_values("lambda_nm").reset_index(drop=True)
    return df_eps


def sanitize_cross_sections(cext, csca, label):
    cext = np.asarray(cext, dtype=np.float64)
    csca = np.asarray(csca, dtype=np.float64)
    if not np.all(np.isfinite(cext)) or not np.all(np.isfinite(csca)):
        raise ValueError(f"{label} Mie calculation produced non-finite cross sections")

    csca = np.clip(csca, 0.0, None)
    cext = np.clip(cext, 0.0, None)
    cabs = np.clip(cext - csca, 0.0, None)
    cext = np.maximum(cext, csca + cabs)
    return cext, csca, cabs


def checked_unit_interval(values, label, tol=1e-6):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(values < -tol) or np.any(values > 1.0 + tol):
        raise ValueError(
            f"{label} must stay in [0, 1]; got range "
            f"{np.min(values):.6g} to {np.max(values):.6g}"
        )
    return np.clip(values, 0.0, 1.0)


def checked_nonnegative(values, label):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(values < 0.0):
        raise ValueError(f"{label} contains negative values")
    return values


def dielectric_cache_key(file_eps):
    path = Path(file_eps).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return str(path), 0, 0


def diameter_distribution_weights(diameters_nm, lower, upper, mean, sigma):
    diameters_nm = np.asarray(diameters_nm, dtype=np.float64)
    lower = float(lower)
    upper = float(upper)
    mean = float(mean)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if upper < lower:
        raise ValueError("upper diameter bound must be >= lower bound")
    if diameters_nm.size == 1:
        return np.ones(1, dtype=np.float64)

    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    weights = truncnorm.pdf(diameters_nm, a, b, loc=mean, scale=sigma)
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
        idx = int(np.argmin(np.abs(diameters_nm - np.clip(mean, lower, upper))))
        weights = np.zeros_like(diameters_nm, dtype=np.float64)
        weights[idx] = 1.0
        return weights
    return weights / np.sum(weights)


def representative_diameters(n_samples, lower, upper, mean, sigma):
    n_samples = int(n_samples)
    lower = float(lower)
    upper = float(upper)
    mean = float(mean)
    sigma = float(sigma)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if upper == lower:
        return np.full(n_samples, lower, dtype=np.float64)
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    quantiles = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    return truncnorm.ppf(quantiles, a, b, loc=mean, scale=sigma).astype(np.float64)


@lru_cache(maxsize=16)
def compute_cu_mie_lut(
    file_eps,
    file_mtime_ns,
    file_size,
    n_medium,
    lambda_min,
    lambda_max,
    lambda_step,
    diameter_lower,
    diameter_upper,
    diameter_grid_size,
):
    del file_mtime_ns, file_size
    grid_size = max(1, int(diameter_grid_size))
    diameter_lower = float(diameter_lower)
    diameter_upper = float(diameter_upper)
    if diameter_upper < diameter_lower:
        raise ValueError("diameter_upper must be >= diameter_lower")
    if diameter_upper == diameter_lower:
        diameters_nm = np.array([diameter_lower], dtype=np.float64)
    else:
        diameters_nm = np.linspace(diameter_lower, diameter_upper, grid_size, dtype=np.float64)

    lambdas_nm, cext, csca, cabs = compute_cu_mie_cross_sections(
        diameters_nm=diameters_nm,
        file_eps=file_eps,
        n_medium=float(n_medium),
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        lambda_step=float(lambda_step),
    )
    return lambdas_nm, diameters_nm, cext, csca, cabs


def get_cu_mie_lut(params):
    file_eps, mtime_ns, file_size = dielectric_cache_key(params["file_eps"])
    diameter_grid_size = int(params.get("mie_diameter_grid_size", 160))
    return compute_cu_mie_lut(
        file_eps,
        mtime_ns,
        file_size,
        round(float(params["n_medium"]), 8),
        round(float(params["mie_lambda_min_nm"]), 8),
        round(float(params["mie_lambda_max_nm"]), 8),
        round(float(params["mie_lambda_step_nm"]), 8),
        round(float(params["cu_lower_nm"]), 8),
        round(float(params["cu_upper_nm"]), 8),
        diameter_grid_size,
    )


def select_wavelength_grid(lambdas_nm_full, arrays, params):
    mask_wl = (lambdas_nm_full >= params["fit_lambda_min_nm"]) & (lambdas_nm_full <= params["fit_lambda_max_nm"])
    if not np.any(mask_wl):
        raise ValueError("No Mie wavelengths fall inside the requested fit window")
    lambdas_crop = lambdas_nm_full[mask_wl]

    n_pick = min(int(params["n_wavelengths"]), len(lambdas_crop))
    if n_pick <= 0:
        raise ValueError("n_wavelengths must be positive")
    idx_pick = np.linspace(0, len(lambdas_crop) - 1, n_pick, dtype=int)
    selected_arrays = [arr[:, mask_wl][:, idx_pick] for arr in arrays]
    return lambdas_crop[idx_pick], selected_arrays


def mie_ab(m, x, n_max):
    ns = np.arange(1, n_max + 1)

    jn_x = spherical_jn(ns, x)
    jn_1_x = spherical_jn(ns - 1, x)
    yn_x = spherical_yn(ns, x)
    yn_1_x = spherical_yn(ns - 1, x)

    hn_x = jn_x + 1j * yn_x
    hn_1_x = jn_1_x + 1j * yn_1_x

    mx = m * x
    jn_mx = spherical_jn(ns, mx)
    jn_1_mx = spherical_jn(ns - 1, mx)

    xjn_diff = x * jn_1_x - ns * jn_x
    mxjn_diff = mx * jn_1_mx - ns * jn_mx
    xhn_diff = x * hn_1_x - ns * hn_x

    an = (m**2 * jn_mx * xjn_diff - jn_x * mxjn_diff) / (
        m**2 * jn_mx * xhn_diff - hn_x * mxjn_diff
    )
    bn = (jn_mx * xjn_diff - jn_x * mxjn_diff) / (
        jn_mx * xhn_diff - hn_x * mxjn_diff
    )
    return an, bn


def compute_cu_mie_cross_sections(
    diameters_nm,
    file_eps,
    n_medium=1.75,
    lambda_min=350.0,
    lambda_max=800.0,
    lambda_step=1.0,
):
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

    cext, csca, cabs = sanitize_cross_sections(cext, csca, "Cu")
    return lambdas_nm, cext, csca, cabs


def n_ceo2_model(wl_nm):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    return 2.2 + 0.15 * np.exp(-((wl_nm - 350.0) ** 2) / (2.0 * 120.0 ** 2))


def k_ceo2_model(wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    k = np.zeros_like(wl_nm, dtype=np.float64)

    uv_mask = wl_nm < uv_cut_nm
    k[uv_mask] = uv_amp * np.exp(-(wl_nm[uv_mask] - 350.0) / uv_decay_nm)
    k += vis_k0 * np.exp(-((wl_nm - vis_peak_nm) ** 2) / (2.0 * vis_width_nm ** 2))
    return k


def support_efficiencies(
    d_nm,
    wl_nm,
    refractive_index,
    n_medium,
    qsca_geo_scale,
    qabs_geo_offset,
    qabs_k_scale,
    transition_center=0.5,
    transition_width=0.08,
):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    n_medium = float(n_medium)
    if n_medium <= 0:
        raise ValueError("n_medium must be positive")

    m_rel = np.asarray(refractive_index, dtype=np.complex128) / n_medium
    x = np.pi * float(d_nm) * n_medium / wl_nm
    m_eff = (m_rel**2 - 1.0) / (m_rel**2 + 2.0)

    qsca_rayleigh = (8.0 / 3.0) * x**4 * np.abs(m_eff) ** 2
    qabs_rayleigh = 4.0 * x * np.imag(m_eff)

    k = np.maximum(np.imag(refractive_index), 0.0)
    qsca_geo = qsca_geo_scale * (1.0 - 0.05 / np.maximum(x, 1e-12))
    qabs_geo = qabs_geo_offset + qabs_k_scale * k

    blend = 1.0 / (1.0 + np.exp(-(x - transition_center) / transition_width))
    qsca = (1.0 - blend) * qsca_rayleigh + blend * qsca_geo
    qabs = (1.0 - blend) * qabs_rayleigh + blend * qabs_geo

    return np.clip(qsca, 0.0, None), np.clip(qabs, 0.0, None)


def mie_ceo2_approx(d_nm, wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0, n_medium=1.75):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    n = n_ceo2_model(wl_nm)
    k = k_ceo2_model(wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0)
    m = n + 1j * k
    qsca, qabs = support_efficiencies(
        d_nm,
        wl_nm,
        m,
        n_medium,
        qsca_geo_scale=2.0,
        qabs_geo_offset=0.05,
        qabs_k_scale=2.5,
    )

    sigma_geo = np.pi * (d_nm * 1e-9 / 2.0) ** 2
    csca_m2 = sigma_geo * qsca
    cabs_m2 = sigma_geo * qabs
    return csca_m2, cabs_m2


def resolve_support_size_nm(params):
    support_type = str(params.get("support_type", "CeO2"))
    if support_type == "N-doped TiO2":
        return float(params.get("support_size_nm", 30.0))
    return float(params.get("ceo2_size_nm", params.get("support_size_nm", 30.0)))


def n_rutile_devore(wl_nm):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    wl_um = wl_nm / 1000.0
    return np.sqrt(5.913 + 0.2441 / (wl_um ** 2 - 0.0803))


def k_tio2_doped_model(
    wl_nm,
    tio2_uv_amp=0.08,
    tio2_uv_decay_nm=20.0,
    tio2_vis_peak_nm=520.0,
    tio2_vis_width_nm=150.0,
    tio2_vis_k0=5e-3,
):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    k = np.zeros_like(wl_nm, dtype=np.float64)

    uv_mask = wl_nm < 400.0
    k[uv_mask] += tio2_uv_amp * np.exp(-(wl_nm[uv_mask] - 350.0) / tio2_uv_decay_nm)
    k += tio2_vis_k0 * np.exp(-((wl_nm - tio2_vis_peak_nm) ** 2) / (2.0 * tio2_vis_width_nm ** 2))
    return np.clip(k, 0.0, None)


def mie_tio2_approx(
    d_nm,
    wl_nm,
    n_medium=2.55,
    tio2_uv_amp=0.08,
    tio2_uv_decay_nm=20.0,
    tio2_vis_peak_nm=520.0,
    tio2_vis_width_nm=150.0,
    tio2_vis_k0=5e-3,
):
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    n = n_rutile_devore(wl_nm)
    k = k_tio2_doped_model(
        wl_nm,
        tio2_uv_amp=tio2_uv_amp,
        tio2_uv_decay_nm=tio2_uv_decay_nm,
        tio2_vis_peak_nm=tio2_vis_peak_nm,
        tio2_vis_width_nm=tio2_vis_width_nm,
        tio2_vis_k0=tio2_vis_k0,
    )
    m = n + 1j * k

    qsca, qabs = support_efficiencies(
        d_nm,
        wl_nm,
        m,
        n_medium,
        qsca_geo_scale=2.2,
        qabs_geo_offset=0.1,
        qabs_k_scale=3.0,
    )

    sigma_geo = np.pi * (d_nm * 1e-9 / 2.0) ** 2
    csca_m2 = sigma_geo * qsca
    cabs_m2 = sigma_geo * qabs
    cext_m2 = csca_m2 + cabs_m2
    return csca_m2, cabs_m2, cext_m2


def build_support_cross_sections(params, wl_nm):
    support_type = str(params.get("support_type", "CeO2"))
    support_size_nm = resolve_support_size_nm(params)
    n_medium = float(params.get("n_medium", 1.75))

    if support_type == "N-doped TiO2":
        csca_support, cabs_support, cext_support = mie_tio2_approx(
            support_size_nm,
            wl_nm,
            n_medium=n_medium,
            tio2_uv_amp=float(params.get("tio2_uv_amp", 0.08)),
            tio2_uv_decay_nm=float(params.get("tio2_uv_decay_nm", 20.0)),
            tio2_vis_peak_nm=float(params.get("tio2_vis_peak_nm", 520.0)),
            tio2_vis_width_nm=float(params.get("tio2_vis_width_nm", 150.0)),
            tio2_vis_k0=float(params.get("tio2_vis_k0", 5e-3)),
        )
    else:
        csca_support, cabs_support = mie_ceo2_approx(
            support_size_nm,
            wl_nm,
            params["uv_cut_nm"],
            params["uv_amp"],
            params["uv_decay_nm"],
            params["vis_peak_nm"],
            params["vis_width_nm"],
            params["vis_k0"],
            n_medium=n_medium,
        )
        cext_support = csca_support + cabs_support
    cext_support, csca_support, cabs_support = sanitize_cross_sections(
        cext_support,
        csca_support,
        f"{support_type} support",
    )
    return csca_support, cabs_support, cext_support


def _prepare_coefficients(params, rng):
    use_mie_lut = bool(params.get("use_mie_lut", True))
    if use_mie_lut:
        lambdas_nm_full, diameter_grid_nm, cext_full, _, cabs_full = get_cu_mie_lut(params)
        weights = diameter_distribution_weights(
            diameter_grid_nm,
            params["cu_lower_nm"],
            params["cu_upper_nm"],
            params["cu_mean_nm"],
            params["cu_sigma_nm"],
        )
        diameters_nm = representative_diameters(
            int(params["n_particles"]),
            params["cu_lower_nm"],
            params["cu_upper_nm"],
            params["cu_mean_nm"],
            params["cu_sigma_nm"],
        )
    else:
        diameters_nm = truncated_normal(
            int(params["n_particles"]),
            params["cu_lower_nm"],
            params["cu_upper_nm"],
            params["cu_mean_nm"],
            params["cu_sigma_nm"],
            rng=rng,
        )
        lambdas_nm_full, cext_full, _, cabs_full = compute_cu_mie_cross_sections(
            diameters_nm=diameters_nm,
            file_eps=params["file_eps"],
            n_medium=params["n_medium"],
            lambda_min=params["mie_lambda_min_nm"],
            lambda_max=params["mie_lambda_max_nm"],
            lambda_step=params["mie_lambda_step_nm"],
        )
        weights = None

    lambda_sel_nm, (cext_cu_nm2, cabs_cu_nm2) = select_wavelength_grid(
        lambdas_nm_full,
        [cext_full, cabs_full],
        params,
    )
    cext_cu = cext_cu_nm2 * 1e-18
    cabs_cu = cabs_cu_nm2 * 1e-18

    csca_support, cabs_support, cext_support = build_support_cross_sections(params, lambda_sel_nm)

    n_support = params["total_dens"] / (1.0 + params["number_ratio"])
    n_cu = params["total_dens"] * params["number_ratio"] / (1.0 + params["number_ratio"])

    # Hybrid GPU approximation:
    # use ensemble-averaged Cu extinction and ensemble-averaged Cu absorption probability
    if weights is None:
        ext_np_sum = np.sum(cext_cu, axis=0)
        abs_np_sum = np.sum(cabs_cu, axis=0)
        ext_np_avg = np.mean(cext_cu, axis=0)
        p_abs_np_eff = np.divide(abs_np_sum, ext_np_sum, out=np.zeros_like(abs_np_sum), where=ext_np_sum > 0)
    else:
        ext_np_avg = np.sum(weights[:, None] * cext_cu, axis=0)
        abs_np_avg = np.sum(weights[:, None] * cabs_cu, axis=0)
        p_abs_np_eff = np.divide(abs_np_avg, ext_np_avg, out=np.zeros_like(abs_np_avg), where=ext_np_avg > 0)
    p_abs_np_eff = checked_unit_interval(p_abs_np_eff, "Cu absorption probability")

    p_abs_sup = np.divide(cabs_support, cext_support, out=np.zeros_like(cabs_support), where=cext_support > 0)
    p_abs_sup = checked_unit_interval(p_abs_sup, "support absorption probability")

    alpha_np = n_cu * ext_np_avg
    alpha_sup = n_support * cext_support
    alpha_tot = alpha_np + alpha_sup
    alpha_tot = checked_nonnegative(alpha_tot, "total attenuation coefficient")

    choose_np = np.divide(alpha_np, alpha_tot, out=np.zeros_like(alpha_tot), where=alpha_tot > 0)
    choose_np = checked_unit_interval(choose_np, "Cu collision probability")

    return {
        "lambda_nm": lambda_sel_nm.astype(np.float32),
        "diameters_nm": diameters_nm.astype(np.float64),
        "alpha_tot": alpha_tot.astype(np.float32),
        "choose_np": choose_np.astype(np.float32),
        "p_abs_np_eff": p_abs_np_eff.astype(np.float32),
        "p_abs_sup": p_abs_sup.astype(np.float32),
    }


def _prepare_coefficients_legacy_exact(params, rng):
    diameters_nm = truncated_normal(
        int(params["n_particles"]),
        params["cu_lower_nm"],
        params["cu_upper_nm"],
        params["cu_mean_nm"],
        params["cu_sigma_nm"],
        rng=rng,
    )

    lambdas_nm_full, cext_full, csca_full, cabs_full = compute_cu_mie_cross_sections(
        diameters_nm=diameters_nm,
        file_eps=params["file_eps"],
        n_medium=params["n_medium"],
        lambda_min=params["mie_lambda_min_nm"],
        lambda_max=params["mie_lambda_max_nm"],
        lambda_step=params["mie_lambda_step_nm"],
    )

    lambda_sel_nm, (cext_cu_nm2, csca_cu_nm2, cabs_cu_nm2) = select_wavelength_grid(
        lambdas_nm_full,
        [cext_full, csca_full, cabs_full],
        params,
    )
    cext_cu = cext_cu_nm2 * 1e-18
    csca_cu = csca_cu_nm2 * 1e-18
    cabs_cu = cabs_cu_nm2 * 1e-18

    csca_support, cabs_support, cext_support = build_support_cross_sections(params, lambda_sel_nm)

    return {
        "lambda_nm": lambda_sel_nm.astype(np.float64),
        "diameters_nm": diameters_nm.astype(np.float64),
        "cext_cu": cext_cu.astype(np.float64),
        "csca_cu": csca_cu.astype(np.float64),
        "cabs_cu": cabs_cu.astype(np.float64),
        "cext_support": cext_support.astype(np.float64),
        "cabs_support": cabs_support.astype(np.float64),
    }


@cuda.jit
def mc_kernel(alpha_tot, choose_np, p_abs_np_eff, p_abs_sup, h, max_steps, n_photons, rng_states, absorb_counts, reflect_counts):
    tid = cuda.grid(1)
    n_wl = alpha_tot.size
    total_threads = n_wl * n_photons

    if tid >= total_threads:
        return

    idx = tid // n_photons

    alpha = alpha_tot[idx]
    choose = choose_np[idx]
    p_abs_np = p_abs_np_eff[idx]
    p_abs_s = p_abs_sup[idx]

    if alpha <= 0.0:
        cuda.atomic.add(reflect_counts, idx, 1)
        return

    z = 1e-12
    direction = 1.0

    for _ in range(max_steps):
        u = xoroshiro128p_uniform_float32(rng_states, tid)
        if u < 1e-7:
            u = 1e-7

        step = -math.log(u) / alpha
        z += direction * step

        if z <= 0.0:
            cuda.atomic.add(reflect_counts, idx, 1)
            return

        if z >= h:
            z = h - 1e-12
            direction = -1.0
            continue

        u_choose = xoroshiro128p_uniform_float32(rng_states, tid)
        if u_choose < choose:
            u_abs = xoroshiro128p_uniform_float32(rng_states, tid)
            if u_abs < p_abs_np:
                cuda.atomic.add(absorb_counts, idx, 1)
                return
            direction = 1.0 if xoroshiro128p_uniform_float32(rng_states, tid) < 0.5 else -1.0
        else:
            u_abs = xoroshiro128p_uniform_float32(rng_states, tid)
            if u_abs < p_abs_s:
                cuda.atomic.add(absorb_counts, idx, 1)
                return
            direction = 1.0 if xoroshiro128p_uniform_float32(rng_states, tid) < 0.5 else -1.0

    cuda.atomic.add(reflect_counts, idx, 1)


def run_simulation_gpu(params):
    """
    GPU-accelerated hybrid version.

    Important:
    - Mie cross sections are still computed on CPU.
    - Monte Carlo photon transport is moved to GPU.
    - Cu per-collision particle sampling is replaced by an ensemble-averaged Cu absorption probability.
      This usually preserves the main spectrum trend while giving a large speedup.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your NVIDIA driver / CUDA / numba installation.")

    rng = np.random.default_rng(int(params["seed"]))
    prep = _prepare_coefficients(params, rng)

    lambda_sel_nm = prep["lambda_nm"]
    diameters_nm = prep["diameters_nm"]
    alpha_tot = prep["alpha_tot"]
    choose_np = prep["choose_np"]
    p_abs_np_eff = prep["p_abs_np_eff"]
    p_abs_sup = prep["p_abs_sup"]

    n_photons = int(params["n_photons"])
    h = np.float32(params["h_sample_m"])
    max_steps = int(params.get("max_steps_per_photon", 3000))

    n_wl = len(lambda_sel_nm)
    total_threads = n_wl * n_photons

    threads_per_block = 128
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block

    d_alpha_tot = cuda.to_device(alpha_tot)
    d_choose_np = cuda.to_device(choose_np)
    d_p_abs_np_eff = cuda.to_device(p_abs_np_eff)
    d_p_abs_sup = cuda.to_device(p_abs_sup)

    d_absorb_counts = cuda.to_device(np.zeros(n_wl, dtype=np.int32))
    d_reflect_counts = cuda.to_device(np.zeros(n_wl, dtype=np.int32))

    rng_states = create_xoroshiro128p_states(total_threads, seed=int(params["seed"]))

    mc_kernel[blocks_per_grid, threads_per_block](
        d_alpha_tot,
        d_choose_np,
        d_p_abs_np_eff,
        d_p_abs_sup,
        h,
        max_steps,
        n_photons,
        rng_states,
        d_absorb_counts,
        d_reflect_counts,
    )

    absorb_counts = d_absorb_counts.copy_to_host().astype(np.float64)
    reflect_counts = d_reflect_counts.copy_to_host().astype(np.float64)

    a_arr = absorb_counts / n_photons
    r_arr = reflect_counts / n_photons

    r_safe = np.clip(r_arr, 1e-6, 0.999999)
    f_r = (1.0 - r_safe) ** 2 / (2.0 * r_safe)
    f_r_norm = f_r / np.max(f_r)

    return {
        "lambda_nm": lambda_sel_nm.astype(np.float64),
        "A_arr": a_arr,
        "R_arr": r_arr,
        "F_R": f_r,
        "F_R_norm": f_r_norm,
        "diameters_nm": diameters_nm,
    }


def run_simulation_legacy_exact(params):
    rng = np.random.default_rng(int(params["seed"]))
    prep = _prepare_coefficients_legacy_exact(params, rng)

    lambda_sel_nm = prep["lambda_nm"]
    diameters_nm = prep["diameters_nm"]
    cext_cu = prep["cext_cu"]
    cabs_cu = prep["cabs_cu"]
    cext_support = prep["cext_support"]
    cabs_support = prep["cabs_support"]

    n_support = params["total_dens"] / (1.0 + params["number_ratio"])
    n_cu = params["total_dens"] * params["number_ratio"] / (1.0 + params["number_ratio"])

    h = float(params["h_sample_m"])
    n_photons = int(params["n_photons"])
    max_steps = int(params.get("max_steps_per_photon", 3000))

    a_arr = np.zeros(len(lambda_sel_nm), dtype=np.float64)
    r_arr = np.zeros(len(lambda_sel_nm), dtype=np.float64)

    for idx in range(len(lambda_sel_nm)):
        ext_np = cext_cu[:, idx]
        abs_np = cabs_cu[:, idx]

        ext_np_avg = np.mean(ext_np)
        p_abs_np = np.divide(abs_np, ext_np, out=np.zeros_like(abs_np), where=ext_np > 0)
        p_abs_np = checked_unit_interval(p_abs_np, "Cu absorption probability")

        if np.sum(ext_np) > 0:
            weights_np = ext_np / np.sum(ext_np)
        else:
            weights_np = np.ones_like(ext_np) / len(ext_np)

        ext_sup = cext_support[idx]
        abs_sup = cabs_support[idx]
        p_abs_sup = abs_sup / ext_sup if ext_sup > 0 else 0.0
        p_abs_sup = float(checked_unit_interval([p_abs_sup], "support absorption probability")[0])

        alpha_np = n_cu * ext_np_avg
        alpha_sup = n_support * ext_sup
        alpha_tot = alpha_np + alpha_sup
        alpha_tot = float(checked_nonnegative([alpha_tot], "total attenuation coefficient")[0])

        if alpha_tot <= 0:
            a_arr[idx] = 0.0
            r_arr[idx] = 1.0
            continue

        choose_np = alpha_np / alpha_tot
        choose_np = float(checked_unit_interval([choose_np], "Cu collision probability")[0])

        absorb_count = 0
        reflect_count = 0

        for _ in range(n_photons):
            z = 1e-12
            direction = 1.0

            for _ in range(max_steps):
                step = -np.log(max(rng.random(), 1e-300)) / alpha_tot
                z += direction * step

                if z <= 0.0:
                    reflect_count += 1
                    break

                if z >= h:
                    z = h - 1e-12
                    direction = -1.0
                    continue

                if rng.random() < choose_np:
                    j = rng.choice(len(ext_np), p=weights_np)
                    if rng.random() < p_abs_np[j]:
                        absorb_count += 1
                        break
                    direction = 1.0 if rng.random() < 0.5 else -1.0
                else:
                    if rng.random() < p_abs_sup:
                        absorb_count += 1
                        break
                    direction = 1.0 if rng.random() < 0.5 else -1.0
            else:
                reflect_count += 1

        a_arr[idx] = absorb_count / n_photons
        r_arr[idx] = reflect_count / n_photons

    r_safe = np.clip(r_arr, 1e-6, 0.999999)
    f_r = (1.0 - r_safe) ** 2 / (2.0 * r_safe)
    f_r_norm = f_r / np.max(f_r)

    return {
        "lambda_nm": lambda_sel_nm.astype(np.float64),
        "A_arr": a_arr,
        "R_arr": r_arr,
        "F_R": f_r,
        "F_R_norm": f_r_norm,
        "diameters_nm": diameters_nm,
    }


def run_simulation(params):
    simulation_mode = str(params.get("simulation_mode", "fast_hybrid_gpu"))
    if simulation_mode == "legacy_exact_powder":
        return run_simulation_legacy_exact(params)
    if simulation_mode == "fast_hybrid_gpu" and not cuda.is_available():
        if params.get("allow_cpu_fallback", True):
            return run_simulation_legacy_exact(params)
        raise RuntimeError("CUDA is not available and allow_cpu_fallback is disabled.")
    return run_simulation_gpu(params)
