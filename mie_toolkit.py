"""Mie toolkit: single-sphere Mie spectra, near fields and semi-classical hot-carrier channels.

Pure physics (numpy/scipy only, no Streamlit). Refactored from the validated reference
implementation `mie_ref.py` (checked against miepython 3.3.1 and scattnlay 2.4 to <= 6e-7).

Conventions (Bohren & Huffman, "Absorption and Scattering of Light by Small Particles",
Wiley 1983, ch. 4; referred to as BH below):
  * time dependence exp(-i omega t); incident field E_inc = E0 x_hat exp(i k z)
  * particle permittivity eps = (n + i kappa)^2 with Im eps >= 0 (absorbing)
  * host medium real, refractive index n_m; m = sqrt(eps)/n_m, x = k R = 2 pi n_m R / lambda0
  * lambda0 is the vacuum wavelength (tables are in vacuum wavelength / photon energy)
  * lengths in nm, energies in eV, cross sections in nm^2

Hot-carrier channel split: semi-classical damping-channel decomposition (Govorov-type,
Besteiro et al., ACS Photonics 4, 2759 (2017)); NOT an ab initio rate.
"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import spherical_jn, spherical_yn

# Photon energy <-> vacuum wavelength: E[eV] = HC_EV_NM / lambda[nm] (value fixed by the tab spec).
HC_EV_NM = 1239.84193
# Reduced Planck constant in eV s (CODATA 2018).
HBAR_EV_S = 6.582119569e-16

# Drude defaults per material.
#   omega_p_eV, gamma_b_eV: least-squares fit eps_inf - omega_p^2/(omega(omega + i gamma)) to the
#     Johnson & Christy (Phys. Rev. B 6, 4370 (1972)) points with 0.64 <= E < 1.3 eV (6 points),
#     i.e. the SAME table the Drude term is subtracted from (research notes C6; fitted eps_inf
#     Au 11.76, Ag 3.88, Cu 8.06 is not needed downstream). Pd: Rakic et al., Appl. Opt. 37,
#     5271 (1998) Lorentz-Drude intraband term, omega_p,eff = sqrt(f0) omega_p, Gamma0.
#   v_F_m_s: free-electron Fermi velocity, Ashcroft & Mermin Table 2.1. Pd is not a free-electron
#     metal (d band at E_F) -> no v_F, size correction / Landau channel disabled by default.
#   E_ib_eV: interband threshold used to gate the interband label (research notes C5):
#     Au 1.9 eV (X-point onset, Guerrisi et al. PRB 12, 557 (1975)); Ag 3.9 eV and Cu 2.1 eV
#     (onset of Im eps_ib in the JC data); Pd 0 (no gap).
DRUDE_DEFAULTS = {
    "Au": {"omega_p_eV": 9.066, "gamma_b_eV": 0.0735, "v_F_m_s": 1.40e6, "E_ib_eV": 1.9,
           "source": "Drude fit to JC 1972 (0.64-1.3 eV); v_F Ashcroft-Mermin; E_ib Guerrisi 1975"},
    "Ag": {"omega_p_eV": 9.175, "gamma_b_eV": 0.0190, "v_F_m_s": 1.39e6, "E_ib_eV": 3.9,
           "source": "Drude fit to JC 1972 (0.64-1.3 eV); v_F Ashcroft-Mermin; E_ib JC onset"},
    "Cu": {"omega_p_eV": 8.871, "gamma_b_eV": 0.1018, "v_F_m_s": 1.57e6, "E_ib_eV": 2.1,
           "source": "Drude fit to JC 1972 (0.64-1.3 eV); v_F Ashcroft-Mermin; E_ib JC onset"},
    "Pd": {"omega_p_eV": 9.72 * np.sqrt(0.330), "gamma_b_eV": 0.008, "v_F_m_s": None, "E_ib_eV": 0.0,
           "source": "Rakic 1998 LD Drude term (9.72 eV x sqrt(0.330), Gamma0 0.008 eV); "
                     "no free-electron v_F (d band at E_F)"},
}

# Rakic et al. 1998 intraband (Drude) terms, omega_p,eff = sqrt(f0) * omega_p. Use these only with
# Rakic-type tables (mixing them with JC data creates fake sub-threshold interband absorption).
RAKIC_DRUDE = {
    "Au": {"omega_p_eV": 9.03 * np.sqrt(0.760), "gamma_b_eV": 0.053},
    "Ag": {"omega_p_eV": 9.01 * np.sqrt(0.845), "gamma_b_eV": 0.048},
    "Cu": {"omega_p_eV": 10.83 * np.sqrt(0.575), "gamma_b_eV": 0.030},
    "Pd": {"omega_p_eV": 9.72 * np.sqrt(0.330), "gamma_b_eV": 0.008},
}

DEFAULT_A = 0.75  # Kawabata & Kubo, J. Phys. Soc. Jpn. 21, 1765 (1966); = Govorov (Besteiro 2017)


# --------------------------------------------------------------------------------------------
# Dielectric tables
# --------------------------------------------------------------------------------------------
TABLE_FORMATS = ("E_eV_eps", "lambda_nm_nk", "lambda_um_nk")


def _read_text(src):
    if isinstance(src, (str, Path)):
        return Path(src).read_text(encoding="utf-8", errors="replace")
    if isinstance(src, (bytes, bytearray)):
        return bytes(src).decode("utf-8", errors="replace")
    data = src.read()
    return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data


def _detect_format(header, col1, col2):
    """Return (fmt, reason). Header keywords first, value ranges second."""
    h = header.lower()
    if "tabulated nk" in h:  # refractiveindex.info YAML: wavelength in um, n, k
        return "lambda_um_nk", "header: refractiveindex.info 'tabulated nk' (um)"
    if re.search(r"wavelength|lambda|λ", h):
        if re.search(r"\bum\b|µm|micron|lambda_um", h):
            return "lambda_um_nk", "header: wavelength in um"
        if re.search(r"\bnm\b|lambda_nm", h):
            return "lambda_nm_nk", "header: wavelength in nm"
        fmt = "lambda_nm_nk" if col1.max() > 50 else "lambda_um_nk"
        return fmt, "header: wavelength (unit from value range)"
    if re.search(r"(?<![a-z])ev\b|energy|eps|dielectric|permittivity", h):
        return "E_eV_eps", "header: energy / dielectric function"
    if col1.max() > 50:
        return "lambda_nm_nk", "values: column 1 > 50 -> wavelength in nm"
    if np.any(col2 < 0):
        return "E_eV_eps", "values: negative column 2 -> eps1 (n is never negative)"
    return "E_eV_eps", "AMBIGUOUS values (no header, all positive); assumed E_eV eps1 eps2"


def load_epsilon_table(src, fmt="auto", return_info=False):
    """Load a tabulated dielectric function and return (E_eV ascending, eps complex).

    Accepted layouts (3 numeric columns, comma/semicolon/whitespace separated, '#' comments,
    non-numeric lines such as headers or YAML keys are skipped):
      'E_eV_eps'     : photon energy (eV), eps1, eps2   (repo / Rahm-library format)
      'lambda_nm_nk' : vacuum wavelength (nm), n, k
      'lambda_um_nk' : vacuum wavelength (um), n, k     (refractiveindex.info)
    fmt='auto' detects the layout from header keywords, then from value ranges.
    Rows with E <= 0 (e.g. the singular E = 0 row of Rahm-library files) or non-finite values are
    dropped; duplicate energies keep the first occurrence. eps = (n + i k)^2 for n,k tables.
    """
    text = _read_text(src)
    header, rows = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(s)
            continue
        s_code = s.split("#", 1)[0].strip()
        parts = [p for p in re.split(r"[\s,;]+", s_code) if p]
        try:
            vals = [float(p) for p in parts[:3]]
        except ValueError:
            header.append(s)
            continue
        if len(vals) == 3:
            rows.append(vals)
    if not rows:
        raise ValueError("No 3-column numeric rows found in dielectric table")
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    c1, c2, c3 = arr.T
    if fmt == "auto":
        fmt, reason = _detect_format("\n".join(header), c1, c2)
    elif fmt in TABLE_FORMATS:
        reason = "user-selected"
    else:
        raise ValueError(f"Unknown table format {fmt!r}; use one of {TABLE_FORMATS} or 'auto'")
    if fmt == "E_eV_eps":
        E, eps = c1, c2 + 1j * c3
    else:
        lam_nm = c1 * (1000.0 if fmt == "lambda_um_nk" else 1.0)
        with np.errstate(divide="ignore"):
            E = HC_EV_NM / lam_nm
        eps = (c2 + 1j * c3) ** 2
    keep = np.isfinite(E) & (E > 0)
    n_dropped = int(np.sum(~keep))
    E, eps = E[keep], eps[keep]
    order = np.argsort(E, kind="stable")
    E, eps = E[order], eps[order]
    E, idx = np.unique(E, return_index=True)
    eps = eps[idx]
    if E.size < 4:
        raise ValueError("Dielectric table needs at least 4 valid rows")
    if return_info:
        info = {"format": fmt, "reason": reason, "n_rows": int(E.size), "n_dropped": n_dropped,
                "E_min_eV": float(E[0]), "E_max_eV": float(E[-1]),
                "lambda_min_nm": HC_EV_NM / float(E[-1]), "lambda_max_nm": HC_EV_NM / float(E[0])}
        return E, eps, info
    return E, eps


def interpolate_epsilon(E_tab, eps_tab, E_query):
    """PCHIP interpolation of eps1 and eps2 in photon-energy space. Never extrapolates."""
    E_query = np.asarray(E_query, dtype=float)
    if np.any(E_query < E_tab[0] - 1e-12) or np.any(E_query > E_tab[-1] + 1e-12):
        raise ValueError("Requested energies are outside the dielectric table (no extrapolation)")
    Eq = np.clip(E_query, E_tab[0], E_tab[-1])
    return (PchipInterpolator(E_tab, eps_tab.real, extrapolate=False)(Eq)
            + 1j * PchipInterpolator(E_tab, eps_tab.imag, extrapolate=False)(Eq))


def table_wavelength_mask(lam_nm, E_tab):
    """True where lambda lies inside the table range (photon energy within [E_min, E_max])."""
    E = HC_EV_NM / np.asarray(lam_nm, dtype=float)
    return (E >= E_tab[0] - 1e-12) & (E <= E_tab[-1] + 1e-12)


def fit_drude(E_tab, eps_tab, E_max_eV=1.3, E_min_eV=0.0):
    """Fit eps_inf - omega_p^2/(omega(omega + i gamma)) to table points E_min <= E < E_max.

    Relative residuals on Re and Im (same procedure that produced DRUDE_DEFAULTS).
    Returns dict(omega_p_eV, gamma_b_eV, eps_inf, n_points) or None if < 4 points.
    """
    sel = (E_tab >= E_min_eV) & (E_tab < E_max_eV)
    if np.sum(sel) < 4:
        return None
    E, eps = E_tab[sel], eps_tab[sel]

    def res(p):
        model = p[0] + drude_epsilon(E, p[1], p[2])
        return np.r_[(model.real - eps.real) / np.abs(eps), (model.imag - eps.imag) / np.abs(eps)]

    s = least_squares(res, [5.0, 9.0, 0.05], bounds=([0.0, 1.0, 1e-3], [20.0, 15.0, 0.5]))
    return {"omega_p_eV": float(s.x[1]), "gamma_b_eV": float(s.x[2]), "eps_inf": float(s.x[0]),
            "n_points": int(np.sum(sel))}


# --------------------------------------------------------------------------------------------
# Drude / size correction
# --------------------------------------------------------------------------------------------
def drude_epsilon(E_eV, omega_p_eV, gamma_eV):
    """Drude term eps_D = -omega_p^2 / (omega^2 + i omega gamma) (energies in eV).

    Im eps_D = omega_p^2 gamma / (omega (omega^2 + gamma^2)) >= 0.
    """
    E = np.asarray(E_eV, dtype=float)
    return -omega_p_eV ** 2 / (E * (E + 1j * gamma_eV))


def surface_gamma(A, v_F_m_s, R_nm):
    """Kreibig surface damping hbar*gamma_s = A hbar v_F / R in eV (R = sphere radius).

    A = 3/4: Kawabata & Kubo (1966) = Govorov (Besteiro 2017, eq. 6/20) = Shahbazyan PRB 94,
    235431 (2016) eq. 22. Returns 0 when v_F is None or 0.
    """
    if not v_F_m_s or R_nm <= 0:
        return 0.0
    return float(A * HBAR_EV_S * v_F_m_s / (R_nm * 1e-9))


def size_corrected_epsilon(E_eV, eps_tab, omega_p_eV, gamma_b_eV, gamma_s_eV):
    """eps_sc = eps_tab - eps_D(gamma_b) + eps_D(gamma_b + gamma_s).

    Replaces the bulk Drude term by the surface-broadened one (research notes C1.2; the sign is
    such that Im eps_sc >= Im eps_tab for 0 < gamma < omega). gamma_s = 0 returns eps_tab.
    gamma_s may be a scalar or an array matching E_eV.
    """
    eps_tab = np.asarray(eps_tab, dtype=complex)
    gamma_s = np.asarray(gamma_s_eV, dtype=float)
    if np.all(gamma_s == 0):
        return eps_tab.copy()
    return (eps_tab - drude_epsilon(E_eV, omega_p_eV, gamma_b_eV)
            + drude_epsilon(E_eV, omega_p_eV, gamma_b_eV + gamma_s))


# --------------------------------------------------------------------------------------------
# Mie coefficients and efficiencies
# --------------------------------------------------------------------------------------------
def n_max_wiscombe(x):
    """Wiscombe, Appl. Opt. 19, 1505 (1980): ceil(2 + x + 4.05 x^(1/3)) for the largest x."""
    x = float(np.max(np.abs(x)))
    return int(np.ceil(2.0 + x + 4.05 * x ** (1.0 / 3.0)))


def _psi(n, z):
    """Riccati-Bessel psi_n(z) = z j_n(z) and psi_n'(z)."""
    j = spherical_jn(n, z)
    return z * j, j + z * spherical_jn(n, z, derivative=True)


def _xi(n, x):
    """Riccati-Bessel xi_n(x) = x h_n^(1)(x) and xi_n'(x) (real x)."""
    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    hp = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    return x * h, h + x * hp


def mie_coefficients(m, x, n_max=None):
    """Scattering (a_n, b_n) and internal (c_n, d_n) coefficients, arrays of shape (n_lambda, n_max).

    BH eq. 4.53 (mu_1 = mu):
      a_n = [m psi_n(mx) psi_n'(x) - psi_n(x) psi_n'(mx)] / [m psi_n(mx) xi_n'(x) - xi_n(x) psi_n'(mx)]
      b_n = [psi_n(mx) psi_n'(x) - m psi_n(x) psi_n'(mx)] / [psi_n(mx) xi_n'(x) - m xi_n(x) psi_n'(mx)]
    BH eq. 4.52 with the Wronskian psi_n xi_n' - xi_n psi_n' = i:
      c_n = i m / den_b,  d_n = i m / den_a.
    Bessel functions from scipy (AMOS, complex argument) -- safe for Im(mx) << 700.
    """
    m = np.atleast_1d(np.asarray(m, dtype=complex))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n_max = int(n_max or n_max_wiscombe(x) + 1)
    n = np.arange(1, n_max + 1)[None, :]
    mm, xx = m[:, None], x[:, None]
    psi_x, dpsi_x = _psi(n, xx)
    xi_x, dxi_x = _xi(n, xx)
    psi_mx, dpsi_mx = _psi(n, mm * xx)
    den_a = mm * psi_mx * dxi_x - xi_x * dpsi_mx
    den_b = psi_mx * dxi_x - mm * xi_x * dpsi_mx
    a = (mm * psi_mx * dpsi_x - psi_x * dpsi_mx) / den_a
    b = (psi_mx * dpsi_x - mm * psi_x * dpsi_mx) / den_b
    return a, b, 1j * mm / den_b, 1j * mm / den_a


def mie_efficiencies(m, x, n_max=None):
    """Q_ext, Q_sca, Q_abs (BH eqs. 4.61, 4.62); n_max = Wiscombe + 1 by default.

    Q_sca = 2/x^2 sum (2n+1)(|a_n|^2 + |b_n|^2); Q_ext = 2/x^2 sum (2n+1) Re(a_n + b_n);
    Q_abs = Q_ext - Q_sca clipped at >= 0. Returns dict with arrays over lambda.
    """
    a, b, _, _ = mie_coefficients(m, x, n_max)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = np.arange(1, a.shape[1] + 1)[None, :]
    q_ext = 2.0 / x ** 2 * np.sum((2 * n + 1) * np.real(a + b), axis=1)
    q_sca = 2.0 / x ** 2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2), axis=1)
    return {"Q_ext": q_ext, "Q_sca": q_sca, "Q_abs": np.clip(q_ext - q_sca, 0.0, None),
            "n_max": a.shape[1]}


def surface_normal_field_integral(m, x, n_max=None):
    """Closed form int_S |E_r(R^-)/E0|^2 dS in units of R^2 (inner side; only N_e1n has E_r):
    2 pi sum_n n(n+1)(2n+1) |d_n|^2 |j_n(mx)/(mx)|^2  (research notes B; quasi-static limit
    (4 pi/3)|E_in/E0|^2)."""
    _, _, _, d = mie_coefficients(m, x, n_max)
    n = np.arange(1, d.shape[1] + 1)[None, :]
    mx = (np.atleast_1d(m) * np.atleast_1d(x))[:, None]
    jr = spherical_jn(n, mx) / mx
    return 2 * np.pi * np.sum(n * (n + 1) * (2 * n + 1) * np.abs(d) ** 2 * np.abs(jr) ** 2, axis=1)


def field_weighted_gamma_s(m, x, q_abs, A, v_F_m_s, R_nm):
    """Shahbazyan (PRB 94, 235431 (2016) eq. 2) gamma_s = A v_F int_S|E_n|^2 dS / int_V|E|^2 dV.

    int_V|E_in/E0|^2 dV = sigma_abs / (k Im m^2) (absorption-field identity), so in R units
    R S/V = S_R2 x Im(m^2) / (pi Q_abs). Reduces to A v_F/R for a uniform interior field.
    """
    if not v_F_m_s:
        return np.zeros_like(np.atleast_1d(x), dtype=float)
    s_r2 = surface_normal_field_integral(m, x)
    im_m2 = np.imag(np.atleast_1d(m) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s_r2 * np.atleast_1d(x) * im_m2 / (np.pi * np.atleast_1d(q_abs))
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    return surface_gamma(A, v_F_m_s, R_nm) * ratio


# --------------------------------------------------------------------------------------------
# Hot-carrier channel split
# --------------------------------------------------------------------------------------------
def hot_electron_split(E_eV, eps_tab, sigma_abs, omega_p_eV, gamma_b_eV, gamma_s_eV, E_ib_eV):
    """Semi-classical damping-channel decomposition of sigma_abs (Govorov-type; NOT ab initio).

    All channels share one internal field, so sigma_ch / sigma_abs = Im eps_ch / Im eps_total
    (research notes C1.3):
      Im eps_ib   = max(0, Im[eps_tab - eps_D(gamma_b)])  (negative values clipped and reported)
      Im eps_D    = Im eps_D(gamma_b + gamma_s), split exactly by Matthiessen weights:
                    bulk intraband = Im eps_D gamma_b/gamma, Landau/surface = Im eps_D gamma_s/gamma
      interband   = Im eps_ib for hbar omega >= E_ib, residual = Im eps_ib below E_ib
      Im eps_total = Im eps_ib + Im eps_D  (equals Im eps_sc where nothing was clipped)
    Weights sum to 1 exactly; Im eps_total <= 0 -> all weights 0.
    Returns sigma_* (nm^2), f_* weights, eta_HE = f_interband + f_Landau, eta_Landau, clip info.
    """
    E = np.asarray(E_eV, dtype=float)
    gamma_s = np.broadcast_to(np.asarray(gamma_s_eV, dtype=float), E.shape)
    gamma = gamma_b_eV + gamma_s
    im_ib_raw = np.imag(np.asarray(eps_tab) - drude_epsilon(E, omega_p_eV, gamma_b_eV))
    im_ib = np.clip(im_ib_raw, 0.0, None)
    im_d = np.imag(drude_epsilon(E, omega_p_eV, gamma))
    im_bulk = im_d * gamma_b_eV / gamma
    im_surf = im_d * gamma_s / gamma
    above = E >= E_ib_eV
    im_inter = np.where(above, im_ib, 0.0)
    im_resid = np.where(above, 0.0, im_ib)
    im_tot = im_ib + im_d
    ok = im_tot > 0
    safe = np.where(ok, im_tot, 1.0)
    f = {k: np.where(ok, v / safe, 0.0) for k, v in
         (("interband", im_inter), ("Landau", im_surf), ("bulk", im_bulk), ("residual", im_resid))}
    sigma_abs = np.asarray(sigma_abs, dtype=float)
    out = {f"f_{k}": v for k, v in f.items()}
    out.update({f"sigma_{k}": v * sigma_abs for k, v in f.items()})
    out["eta_HE"] = f["interband"] + f["Landau"]
    out["eta_Landau"] = f["Landau"]
    neg = im_ib_raw < 0
    out["clip"] = {"n_clipped": int(np.sum(neg)), "n_total": int(E.size),
                   "min_im_eps_ib": float(np.min(im_ib_raw)),
                   "max_rel_clip": float(np.max(np.where(neg, -im_ib_raw / np.where(ok, safe, 1.0), 0.0)))}
    return out


# --------------------------------------------------------------------------------------------
# Full spectrum (what the UI calls)
# --------------------------------------------------------------------------------------------
def compute_spectrum(E_tab, eps_tab, lam_nm, D_nm, n_medium, size_correction=True,
                     omega_p_eV=9.066, gamma_b_eV=0.0735, v_F_m_s=1.40e6, A=DEFAULT_A,
                     E_ib_eV=1.9, field_weighted=False, n_iter=3):
    """Spectrum + channel split on the part of lam_nm covered by the table (no extrapolation).

    eps_tab is PCHIP-interpolated in eV, size-corrected (gamma_s = A v_F/R, or the Shahbazyan
    field-weighted gamma_s(lambda) iterated n_iter times when field_weighted), then Mie with
    m = sqrt(eps_sc)/n_m, x = 2 pi n_m R/lambda0. sigma = Q pi R^2 (nm^2).
    """
    lam_all = np.asarray(lam_nm, dtype=float)
    mask = table_wavelength_mask(lam_all, E_tab)
    lam = lam_all[mask]
    if lam.size < 1:
        raise ValueError("Wavelength range does not overlap the dielectric table")
    R = 0.5 * float(D_nm)
    E = HC_EV_NM / lam
    eps_t = interpolate_epsilon(E_tab, eps_tab, E)
    x = 2 * np.pi * n_medium * R / lam
    gs0 = surface_gamma(A, v_F_m_s, R) if size_correction else 0.0
    gamma_s = np.full(lam.shape, gs0)
    for it in range(n_iter if (field_weighted and gs0 > 0) else 1):
        eps_sc = size_corrected_epsilon(E, eps_t, omega_p_eV, gamma_b_eV, gamma_s)
        m = np.sqrt(eps_sc) / n_medium
        q = mie_efficiencies(m, x)
        if field_weighted and gs0 > 0 and it < n_iter - 1:
            gamma_s = field_weighted_gamma_s(m, x, q["Q_abs"], A, v_F_m_s, R)
    geo = np.pi * R ** 2
    sigma_abs = q["Q_abs"] * geo
    split = hot_electron_split(E, eps_t, sigma_abs, omega_p_eV, gamma_b_eV, gamma_s, E_ib_eV)
    out = {"lam_nm": lam, "E_eV": E, "n_dropped": int(np.sum(~mask)), "R_nm": R, "x": x, "m": m,
           "eps_tab": eps_t, "eps_sc": eps_sc, "gamma_s_eV": gamma_s, "n_medium": float(n_medium),
           "Q_ext": q["Q_ext"], "Q_sca": q["Q_sca"], "Q_abs": q["Q_abs"],
           "sigma_ext": q["Q_ext"] * geo, "sigma_sca": q["Q_sca"] * geo, "sigma_abs": sigma_abs}
    out.update(split)
    return out


# --------------------------------------------------------------------------------------------
# Near field
# --------------------------------------------------------------------------------------------
def n_max_near_field(m, x):
    """Near-field order: max(Wiscombe(x), |m x|) + 4 (miepython keeps extra terms for fields)."""
    return int(max(n_max_wiscombe(x), np.ceil(np.max(np.abs(np.asarray(m) * np.asarray(x)))))) + 4


def _pi_tau(n_max, mu):
    """Angular functions, BH eq. 4.47: pi_1 = 1, pi_2 = 3 mu,
    pi_n = (2n-1)/(n-1) mu pi_{n-1} - n/(n-1) pi_{n-2};  tau_n = n mu pi_n - (n+1) pi_{n-1}."""
    pi = np.zeros((n_max + 1,) + mu.shape)
    tau = np.zeros_like(pi)
    pi[1] = 1.0
    if n_max >= 2:
        pi[2] = 3.0 * mu
    for n in range(3, n_max + 1):
        pi[n] = (2 * n - 1) / (n - 1) * mu * pi[n - 1] - n / (n - 1) * pi[n - 2]
    for n in range(1, n_max + 1):
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi[1:], tau[1:]


def near_field_points(m, x, X, Y, Z, n_max=None):
    """Total complex field E/E0 (Cartesian) at points (X, Y, Z) given in units of R.

    BH eqs. 4.37/4.40/4.45/4.50, E_n = i^n E0 (2n+1)/(n(n+1)):
      r > R: E = E_inc + sum E_n (i a_n N_e1n^(3) - b_n M_o1n^(3)),  rho = k r, z_n = h_n^(1)
      r < R: E = sum E_n (c_n M_o1n^(1) - i d_n N_e1n^(1)),           rho = m k r, z_n = j_n
      M_o1n = (0, cos(phi) pi_n z_n, -sin(phi) tau_n z_n)
      N_e1n = (cos(phi) n(n+1) sin(theta) pi_n z_n/rho, cos(phi) tau_n [rho z_n]'/rho,
               -sin(phi) pi_n [rho z_n]'/rho)
    Points with r < R are inside (E_r jumps by m^2 across r = R). Scalar m, x (one wavelength).
    """
    m = complex(np.asarray(m).ravel()[0])
    x = float(np.asarray(x).ravel()[0])
    X, Y, Z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (X, Y, Z)))
    shape = X.shape
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    n_max = int(n_max or n_max_near_field(m, x))
    a, b, c, d = (v[0] for v in mie_coefficients(m, x, n_max))
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r = np.where(r == 0, 1e-12, r)
    mu = np.clip(Z / r, -1.0, 1.0)
    st = np.sqrt(1.0 - mu ** 2)
    phi = np.arctan2(Y, X)
    cp, sp = np.cos(phi), np.sin(phi)
    pi, tau = _pi_tau(n_max, mu)
    n = np.arange(1, n_max + 1)[:, None]
    En = np.power(1j, n) * (2 * n + 1) / (n * (n + 1))
    inside = r < 1.0
    P = r.size
    rho = np.empty(P, dtype=complex)
    z = np.empty((n_max, P), dtype=complex)
    zp = np.empty((n_max, P), dtype=complex)
    rho[inside] = m * x * r[inside]
    ri = rho[inside]
    z[:, inside] = spherical_jn(n, ri)
    zp[:, inside] = spherical_jn(n, ri, derivative=True)
    ro = x * r[~inside]
    rho[~inside] = ro
    z[:, ~inside] = spherical_jn(n, ro) + 1j * spherical_yn(n, ro)
    zp[:, ~inside] = spherical_jn(n, ro, derivative=True) + 1j * spherical_yn(n, ro, derivative=True)
    dz = z / rho + zp  # [rho z_n]'/rho
    cM = np.where(inside, c[:, None], -b[:, None])
    cN = np.where(inside, -1j * d[:, None], 1j * a[:, None])
    Er = cp * np.sum(En * cN * n * (n + 1) * st * pi * z / rho, axis=0)
    Et = cp * np.sum(En * (cM * pi * z + cN * tau * dz), axis=0)
    Ep = -sp * np.sum(En * (cM * tau * z + cN * pi * dz), axis=0)
    Ex = st * cp * Er + mu * cp * Et - sp * Ep
    Ey = st * sp * Er + mu * sp * Et + cp * Ep
    Ez = mu * Er - st * Et
    Ex = np.where(inside, Ex, Ex + np.exp(1j * x * Z))
    return Ex.reshape(shape), Ey.reshape(shape), Ez.reshape(shape)


PLANES = {
    # plane: (horizontal axis, vertical axis, labels)
    "xz": ("x", "z", "x (nm)  [E0 direction]", "z (nm)  [k direction]"),
    "yz": ("y", "z", "y (nm)  [H0 direction]", "z (nm)  [k direction]"),
    "xy": ("x", "y", "x (nm)  [E0 direction]", "y (nm)  [H0 direction]"),
}


def near_field_map(m, x, R_nm, plane="xz", half_width=2.5, n_grid=201, n_max=None):
    """|E|/|E0| on an n_grid x n_grid Cartesian grid through the sphere centre.

    plane: 'xz' (k-E plane, default), 'yz' (k-H plane) or 'xy' (perpendicular to k, z = 0).
    half_width in units of R. Returns dict(X, Y [nm, horizontal/vertical], E_norm, R [nm],
    inside mask, plane, xlabel, ylabel, n_max). Internal field included.
    """
    h_ax, v_ax, xlabel, ylabel = PLANES[plane]
    g = np.linspace(-half_width, half_width, int(n_grid))
    H, V = np.meshgrid(g, g)
    coords = {"x": np.zeros_like(H), "y": np.zeros_like(H), "z": np.zeros_like(H)}
    coords[h_ax], coords[v_ax] = H, V
    n_max = int(n_max or n_max_near_field(m, x))
    Ex, Ey, Ez = near_field_points(m, x, coords["x"], coords["y"], coords["z"], n_max)
    E_norm = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
    return {"X": H * R_nm, "Y": V * R_nm, "E_norm": E_norm, "R": float(R_nm),
            "inside": np.hypot(H, V) < 1.0, "plane": plane, "xlabel": xlabel, "ylabel": ylabel,
            "n_max": n_max}


def surface_max_field(m, x, n_theta=721, r_rel=1.0001, n_max=None):
    """Max |E|/|E0| on the outer surface r = r_rel R. Returns (max, theta_deg, phi_deg).

    Every field component carries cos(phi) (E_r, E_theta) or sin(phi) (E_phi), so
    |E|^2 = cos^2(phi)(|E_r|^2 + |E_theta|^2) + sin^2(phi)|E_phi|^2 is extremal at phi = 0 or 90 deg;
    scanning theta on those two meridians gives the exact surface maximum.
    """
    th = np.linspace(0.0, np.pi, int(n_theta))
    best = (-1.0, 0.0, 0.0)
    for ph in (0.0, 0.5 * np.pi):
        X = r_rel * np.sin(th) * np.cos(ph)
        Y = r_rel * np.sin(th) * np.sin(ph)
        Z = r_rel * np.cos(th)
        Ex, Ey, Ez = near_field_points(m, x, X, Y, Z, n_max)
        En = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        k = int(np.argmax(En))
        if En[k] > best[0]:
            best = (float(En[k]), float(np.degrees(th[k])), float(np.degrees(ph)))
    return best


def guess_base_metal(name):
    """Majority metal from a file name such as 'dielectric_function_Pd_0.47_Au_99.53.txt' or
    'AuPd0.1' (first symbol wins when no fractions are given). Returns 'Au'/'Ag'/'Cu'/'Pd' or None."""
    pairs = re.findall(r"(Au|Ag|Cu|Pd|Pt)_(\d+(?:\.\d+)?)", name)
    if pairs:
        sym = max(pairs, key=lambda p: float(p[1]))[0]
        return sym if sym in DRUDE_DEFAULTS else None
    hits = [(name.find(s), s) for s in DRUDE_DEFAULTS if s in name]
    return min(hits)[1] if hits else None


def peak_index(y):
    """Index of the highest interior local maximum of y (e.g. the LSPR in Q_abs); falls back to
    the global maximum when y is monotonic (a maximum at the range edge is only the edge of a
    rising background such as the Au interband tail)."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return int(np.argmax(y))
    interior = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    if interior.size == 0:
        return int(np.argmax(y))
    return int(interior[np.argmax(y[interior])])
