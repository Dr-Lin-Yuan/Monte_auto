import numpy as np
import pandas as pd
from scipy.special import spherical_jn, spherical_yn


def truncated_normal(n_samples, lower, upper, mean, sigma, batch_size=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    accepted = []
    while len(accepted) < n_samples:
        raw = rng.normal(loc=mean, scale=sigma, size=batch_size)
        ok = raw[(raw >= lower) & (raw <= upper)]
        accepted.extend(ok.tolist())
    return np.asarray(accepted[:n_samples])


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
        sep=r"\s+",
        header=None,
        names=["E_eV", "eps1", "eps2"],
        comment="#",
    )
    df_eps = df_eps.apply(pd.to_numeric, errors="coerce")
    df_eps = df_eps.dropna().reset_index(drop=True)
    df_eps["lambda_nm"] = 1240.0 / df_eps["E_eV"]
    df_eps = df_eps.sort_values("lambda_nm").reset_index(drop=True)
    return df_eps


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
    lambdas_nm = np.arange(lambda_min, lambda_max + lambda_step, lambda_step)
    r_nm = np.asarray(diameters_nm) / 2.0

    df_eps = load_dielectric_function(file_eps)
    eps1 = np.interp(lambdas_nm, df_eps["lambda_nm"], df_eps["eps1"])
    eps2 = np.interp(lambdas_nm, df_eps["lambda_nm"], df_eps["eps2"])
    eps_complex = eps1 + 1j * eps2
    n_particle = np.sqrt(eps_complex)
    m_rel = n_particle / n_medium

    k_medium = 2.0 * np.pi * n_medium / lambdas_nm
    n_part = r_nm.size
    n_lambda = lambdas_nm.size

    cext = np.zeros((n_part, n_lambda), dtype=float)
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

    cabs = cext - csca
    return lambdas_nm, cext, csca, cabs


def n_ceo2_model(wl_nm):
    wl_nm = np.asarray(wl_nm, dtype=float)
    return 2.2 + 0.15 * np.exp(-((wl_nm - 350.0) ** 2) / (2.0 * 120.0 ** 2))


def k_ceo2_model(wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0):
    wl_nm = np.asarray(wl_nm, dtype=float)
    k = np.zeros_like(wl_nm, dtype=float)

    uv_mask = wl_nm < uv_cut_nm
    k[uv_mask] = uv_amp * np.exp(-(wl_nm[uv_mask] - 350.0) / uv_decay_nm)
    k += vis_k0 * np.exp(-((wl_nm - vis_peak_nm) ** 2) / (2.0 * vis_width_nm ** 2))
    return k


def mie_ceo2_approx(d_nm, wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0):
    wl_nm = np.asarray(wl_nm, dtype=float)
    n = n_ceo2_model(wl_nm)
    k = k_ceo2_model(wl_nm, uv_cut_nm, uv_amp, uv_decay_nm, vis_peak_nm, vis_width_nm, vis_k0)
    m = n + 1j * k

    x = np.pi * d_nm / wl_nm
    m_eff = (m**2 - 1.0) / (m**2 + 2.0)

    qsca = np.where(
        x < 0.5,
        (8.0 / 3.0) * x**4 * np.abs(m_eff) ** 2,
        2.0 * (1.0 - 0.05 / np.maximum(x, 1e-12)),
    )
    qsca = np.clip(qsca, 0.0, None)

    qabs = np.where(
        x < 0.5,
        4.0 * x * np.imag(m_eff),
        0.05 + 2.5 * k,
    )
    qabs = np.clip(qabs, 0.0, None)

    sigma_geo = np.pi * (d_nm * 1e-9 / 2.0) ** 2
    csca_m2 = sigma_geo * qsca
    cabs_m2 = sigma_geo * qabs
    return csca_m2, cabs_m2


def run_simulation(params):
    rng = np.random.default_rng(int(params["seed"]))

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

    mask_wl = (lambdas_nm_full >= params["fit_lambda_min_nm"]) & (lambdas_nm_full <= params["fit_lambda_max_nm"])
    lambdas_crop = lambdas_nm_full[mask_wl]
    cext_crop = cext_full[:, mask_wl]
    csca_crop = csca_full[:, mask_wl]
    cabs_crop = cabs_full[:, mask_wl]

    n_pick = min(int(params["n_wavelengths"]), len(lambdas_crop))
    idx_pick = np.linspace(0, len(lambdas_crop) - 1, n_pick, dtype=int)

    lambda_sel_nm = lambdas_crop[idx_pick]
    cext_cu = cext_crop[:, idx_pick] * 1e-18
    csca_cu = csca_crop[:, idx_pick] * 1e-18
    cabs_cu = cabs_crop[:, idx_pick] * 1e-18

    csca_ceo2, cabs_ceo2 = mie_ceo2_approx(
        params["ceo2_size_nm"],
        lambda_sel_nm,
        params["uv_cut_nm"],
        params["uv_amp"],
        params["uv_decay_nm"],
        params["vis_peak_nm"],
        params["vis_width_nm"],
        params["vis_k0"],
    )
    cext_ceo2 = csca_ceo2 + cabs_ceo2

    n_ceo2 = params["total_dens"] / (1.0 + params["number_ratio"])
    n_cu = params["total_dens"] * params["number_ratio"] / (1.0 + params["number_ratio"])

    h = params["h_sample_m"]
    n_photons = int(params["n_photons"])

    a_arr = np.zeros(len(lambda_sel_nm), dtype=float)
    r_arr = np.zeros(len(lambda_sel_nm), dtype=float)

    max_steps_per_photon = 3000

    for idx in range(len(lambda_sel_nm)):
        if idx % 10 == 0:
            print(f"  wavelength {idx+1}/{len(lambda_sel_nm)}", flush=True)

        ext_np = cext_cu[:, idx]
        abs_np = cabs_cu[:, idx]

        ext_np_avg = np.mean(ext_np)
        p_abs_np = np.divide(abs_np, ext_np, out=np.zeros_like(abs_np), where=ext_np > 0)

        if np.sum(ext_np) > 0:
            weights_np = ext_np / np.sum(ext_np)
        else:
            weights_np = np.ones_like(ext_np) / len(ext_np)

        ext_sup = cext_ceo2[idx]
        abs_sup = cabs_ceo2[idx]
        p_abs_sup = abs_sup / ext_sup if ext_sup > 0 else 0.0

        alpha_np = n_cu * ext_np_avg
        alpha_sup = n_ceo2 * ext_sup
        alpha_tot = alpha_np + alpha_sup

        if alpha_tot <= 0:
            a_arr[idx] = 0.0
            r_arr[idx] = 1.0
            continue

        choose_np = alpha_np / alpha_tot

        absorb_count = 0
        reflect_count = 0

        for _ in range(n_photons):
            z = 1e-12
            direction = 1.0
            step_counter = 0

            while True:
                step_counter += 1

                if step_counter > max_steps_per_photon:
                    reflect_count += 1
                    break

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

        scale = 1.0 / n_photons
        a_arr[idx] = absorb_count * scale
        r_arr[idx] = reflect_count * scale

    r_safe = np.clip(r_arr, 1e-6, 0.999999)
    f_r = (1.0 - r_safe) ** 2 / (2.0 * r_safe)
    f_r_norm = f_r / np.max(f_r)

    return {
        "lambda_nm": lambda_sel_nm,
        "A_arr": a_arr,
        "R_arr": r_arr,
        "F_R": f_r,
        "F_R_norm": f_r_norm,
        "diameters_nm": diameters_nm,
    }
