"""Streamlit "Mie toolkit" tab: single-sphere Mie spectra, hot-carrier channels and near field.

Physics lives in mie_toolkit.py; this module only builds inputs, cached calls and figures.
The figure builders (fig_spectra, fig_hot_electron, fig_near_field) take arrays and return a
matplotlib Figure, so they can be used without Streamlit.
"""
import io
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import mie_toolkit as mt

DIELECTRIC_DIR = Path(__file__).resolve().parent / "dielectric_permittivity"
BUILTIN = {
    "Au (Johnson & Christy 1972)": ("JohnsonChristy1972_Au_nk.txt", "Au"),
    "Ag (Johnson & Christy 1972)": ("JohnsonChristy1972_Ag_nk.txt", "Ag"),
    "Cu (Johnson & Christy 1972)": ("JohnsonChristy1972_Cu_nk.txt", "Cu"),
    "Pd (Johnson & Christy 1974)": ("JohnsonChristy1974_Pd_nk.txt", "Pd"),
}
UPLOAD = "Upload tabulated epsilon"
FORMAT_LABELS = {"auto": "auto-detect", "E_eV_eps": "E (eV), eps1, eps2",
                 "lambda_nm_nk": "lambda (nm), n, k", "lambda_um_nk": "lambda (um), n, k"}
HE_CAVEAT = ("Semi-classical damping-channel decomposition (Govorov-type); not an ab initio rate. "
             "Interband carriers are mainly d-band holes; the bulk-intraband channel also contains "
             "phonon-assisted carriers (Brown et al., ACS Nano 10, 957 (2016)).")

# Categorical slots (fixed order): blue, orange, aqua, yellow.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#777777"


# ------------------------------------------------------------------------------------------
# Pure helpers / figure builders
# ------------------------------------------------------------------------------------------
def material_options(directory=DIELECTRIC_DIR):
    builtin = [k for k, (f, _) in BUILTIN.items() if (directory / f).exists()]
    builtin_files = {f for f, _ in BUILTIN.values()}
    others = sorted(p.name for p in directory.glob("*") if p.is_file() and p.name not in builtin_files)
    return builtin + others + [UPLOAD]


def default_drude(base, E_tab, eps_tab, is_builtin):
    """Drude/damping pre-fill consistent with the loaded table (spec amendment 2)."""
    ref = mt.DRUDE_DEFAULTS.get(base or "Au")
    out = {"omega_p_eV": ref["omega_p_eV"], "gamma_b_eV": ref["gamma_b_eV"],
           "v_F_m_s": ref["v_F_m_s"] or 0.0, "E_ib_eV": ref["E_ib_eV"], "source": ref["source"]}
    if is_builtin or base == "Pd":
        return out
    fit = mt.fit_drude(E_tab, eps_tab)
    if fit is not None:
        out.update(omega_p_eV=fit["omega_p_eV"], gamma_b_eV=fit["gamma_b_eV"],
                   source=f"Drude auto-fit to this table ({fit['n_points']} points, E < 1.3 eV); "
                          f"v_F and E_ib of {base or 'Au'} (guessed base metal)")
    else:
        rk = mt.RAKIC_DRUDE[base or "Au"]
        out.update(omega_p_eV=rk["omega_p_eV"], gamma_b_eV=rk["gamma_b_eV"],
                   source=f"Rakic 1998 Drude term of {base or 'Au'}: table has < 4 points below "
                          f"1.3 eV, cannot fit. Check that it matches the table.")
    return out


def fig_spectra(lam, q_ext, q_sca, q_abs, R_nm=None, use_sigma=False, lam_peak=None):
    fig = Figure(figsize=(6.4, 3.6), dpi=110, layout="constrained")
    ax = fig.add_subplot()
    scale = np.pi * R_nm ** 2 if (use_sigma and R_nm) else 1.0
    pre = "sigma" if scale != 1.0 else "Q"
    for y, lab, c in ((q_ext, "ext", C1), (q_sca, "sca", C2), (q_abs, "abs", C3)):
        ax.plot(lam, np.asarray(y) * scale, color=c, lw=2, label=f"{pre}_{lab}")
    if lam_peak is not None:
        ax.axvline(lam_peak, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Cross section (nm$^2$)" if scale != 1.0 else "Efficiency Q")
    ax.set_xlim(lam[0], lam[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False)
    return fig


def fig_hot_electron(lam, split, E_ib_eV=None):
    fig = Figure(figsize=(6.4, 6.0), dpi=110, layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=[1.4, 1.0])
    chans = (("interband", "interband (d-band holes)", C1),
             ("Landau", "Landau / surface (hot e-h pairs)", C2),
             ("bulk", "bulk intraband (phonon/e-e)", C3),
             ("residual", "sub-threshold residual", C4))
    ax1.stackplot(lam, *[split[f"sigma_{k}"] for k, _, _ in chans], labels=[l for _, l, _ in chans],
                  colors=[c for _, _, c in chans], edgecolor="white", linewidth=0.6)
    ax1.set_ylabel("sigma_abs channels (nm$^2$)")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    for k, lab, c in chans:
        ax2.plot(lam, split[f"f_{k}"], color=c, lw=1.5, label=f"f_{k}")
    ax2.plot(lam, split["eta_HE"], color=INK, lw=2, ls="--", label="eta_HE = f_interband + f_Landau")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of sigma_abs")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    if E_ib_eV and E_ib_eV > 0:
        lam_ib = mt.HC_EV_NM / E_ib_eV
        if lam[0] < lam_ib < lam[-1]:
            for ax in (ax1, ax2):
                ax.axvline(lam_ib, color=MUTED, lw=1, ls=":")
            ax1.annotate(f"E_ib = {E_ib_eV:g} eV", (lam_ib, 0.5), xycoords=("data", "axes fraction"),
                         color=MUTED, fontsize=8, ha="left", xytext=(3, 0), textcoords="offset points")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(lam[0], lam[-1])
    ax1.set_title("Semi-classical damping-channel decomposition (Govorov-type); not an ab initio rate",
                  fontsize=8.5, color=MUTED)
    return fig


def fig_near_field(X, Y, E_norm, R_nm, xlabel, ylabel, log_scale=False, title=None):
    fig = Figure(figsize=(5.6, 4.8), dpi=110, layout="constrained")
    ax = fig.add_subplot()
    data = np.log10(E_norm) if log_scale else E_norm
    im = ax.imshow(data, origin="lower", cmap="magma", interpolation="nearest",
                   extent=(X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]))
    ax.add_patch(Circle((0, 0), R_nm, fill=False, ec="white", lw=1.2))
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("log10(|E|/|E0|)" if log_scale else "|E|/|E0|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def spectrum_csv(spec, meta_lines=()):
    cols = [("lambda_nm", spec["lam_nm"]), ("Q_ext", spec["Q_ext"]), ("Q_sca", spec["Q_sca"]),
            ("Q_abs", spec["Q_abs"]), ("sigma_abs_nm2", spec["sigma_abs"]),
            ("sigma_abs_interband_nm2", spec["sigma_interband"]),
            ("sigma_abs_Landau_nm2", spec["sigma_Landau"]),
            ("sigma_abs_bulk_intraband_nm2", spec["sigma_bulk"]),
            ("sigma_abs_residual_nm2", spec["sigma_residual"]),
            ("f_interband", spec["f_interband"]), ("f_Landau", spec["f_Landau"]),
            ("f_bulk_intraband", spec["f_bulk"]), ("f_residual", spec["f_residual"]),
            ("eta_HE", spec["eta_HE"]), ("eta_Landau", spec["eta_Landau"]),
            ("gamma_s_eV", spec["gamma_s_eV"])]
    buf = io.StringIO()
    for line in meta_lines:
        buf.write(f"# {line}\n")
    buf.write(",".join(c for c, _ in cols) + "\n")
    np.savetxt(buf, np.column_stack([v for _, v in cols]), delimiter=",", fmt="%.8g")
    return buf.getvalue()


def near_field_inputs(spec, lam_nf, E_tab, eps_tab, omega_p_eV, gamma_b_eV):
    """(m, x) at the near-field wavelength, using the same eps_sc construction as the spectrum
    (PCHIP table value at lam_nf; gamma_s interpolated from the spectrum grid)."""
    E = mt.HC_EV_NM / lam_nf
    eps_t = complex(mt.interpolate_epsilon(E_tab, eps_tab, np.array([E]))[0])
    gs = float(np.interp(lam_nf, spec["lam_nm"], spec["gamma_s_eV"]))
    eps_sc = complex(mt.size_corrected_epsilon(E, eps_t, omega_p_eV, gamma_b_eV, gs))
    n_m = spec["n_medium"]
    return np.sqrt(eps_sc) / n_m, 2 * np.pi * n_m * spec["R_nm"] / lam_nf


# ------------------------------------------------------------------------------------------
# Cached computations
# ------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(source, fmt):
    return mt.load_epsilon_table(source, fmt=fmt, return_info=True)


@st.cache_data(show_spinner=False)
def _spectrum(E_tab, eps_tab, lam_min, lam_max, step, D, n_m, size_corr, wp, gb, vF, A, E_ib, fw):
    lam = np.arange(lam_min, lam_max + 0.5 * step, step)
    return mt.compute_spectrum(E_tab, eps_tab, lam, D, n_m, size_corr, wp, gb, vF, A, E_ib, fw)


@st.cache_data(show_spinner=False)
def _near_field(m_re, m_im, x, R_nm, plane, half_width, n_grid):
    # m passed as two floats: st.cache_data cannot hash Python complex numbers.
    m = complex(m_re, m_im)
    nf = mt.near_field_map(m, x, R_nm, plane, half_width, n_grid)
    nf["surface_max"] = mt.surface_max_field(m, x)
    return nf


# ------------------------------------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------------------------------------
def render():
    st.subheader("Mie toolkit: single sphere")
    st.caption("Full Mie solution (Bohren & Huffman ch. 4) with size-corrected tabulated permittivity. "
               "The sidebar controls belong to the Monte Carlo fit tab.")
    left, right = st.columns([1.0, 2.6])

    with left:
        material = st.selectbox("Material", material_options(), key="mie_material")
        fmt = "auto"
        if material == UPLOAD:
            up = st.file_uploader("Tabulated epsilon file", type=["txt", "csv", "dat"], key="mie_upload")
            fmt = st.selectbox("Column format", list(FORMAT_LABELS), format_func=FORMAT_LABELS.get,
                               key="mie_upload_fmt")
            if up is None:
                st.info("Upload a 3-column table: E (eV), eps1, eps2 or lambda, n, k.")
                return
            source, name, is_builtin = up.getvalue(), up.name, False
        elif material in BUILTIN:
            fname, _ = BUILTIN[material]
            source, name, is_builtin = str(DIELECTRIC_DIR / fname), fname, True
        else:
            source, name, is_builtin = str(DIELECTRIC_DIR / material), material, False
        try:
            E_tab, eps_tab, info = _load(source, fmt)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the dielectric table: {exc}")
            return
        base = BUILTIN[material][1] if material in BUILTIN else mt.guess_base_metal(name)
        st.caption(f"Table: {info['n_rows']} rows, {info['lambda_min_nm']:.0f}-{info['lambda_max_nm']:.0f} nm; "
                   f"format {FORMAT_LABELS[info['format']]} ({info['reason']})"
                   + (f"; {info['n_dropped']} row(s) with E <= 0 dropped" if info["n_dropped"] else ""))
        if not is_builtin and base is None:
            st.warning("Base metal not recognised from the file name; Au damping defaults are used.")

        D = st.number_input("Diameter d (nm)", min_value=1.0, max_value=500.0, value=50.0, step=1.0,
                            key="mie_D")
        n_m = st.number_input("Medium refractive index n_m", min_value=1.0, max_value=3.0, value=1.33,
                              step=0.01, format="%.3f", key="mie_nm")
        c1, c2, c3 = st.columns(3)
        lam_min = c1.number_input("lambda min (nm)", min_value=100.0, value=300.0, step=10.0, key="mie_lmin")
        lam_max = c2.number_input("lambda max (nm)", min_value=101.0, value=900.0, step=10.0, key="mie_lmax")
        step = c3.number_input("step (nm)", min_value=0.1, value=1.0, step=0.5, key="mie_lstep")
        if lam_max <= lam_min or (lam_max - lam_min) / step > 20000:
            st.error("Need lambda max > lambda min and at most 20000 wavelength points.")
            return

        size_corr = st.checkbox("Size correction gamma = gamma_bulk + A v_F / R", value=True, key="mie_sc")
        dflt = default_drude(base, E_tab, eps_tab, is_builtin)
        k = f"{material}_{name}"
        with st.expander("Drude / damping parameters", expanded=False):
            st.caption(f"Defaults: {dflt['source']}")
            wp = st.number_input("hbar omega_p (eV)", min_value=0.1, value=float(dflt["omega_p_eV"]),
                                 format="%.4f", key=f"mie_wp_{k}", help=dflt["source"])
            gb = st.number_input("hbar gamma_bulk (eV)", min_value=1e-4, value=float(dflt["gamma_b_eV"]),
                                 format="%.4f", key=f"mie_gb_{k}", help=dflt["source"])
            vF6 = st.number_input("v_F (1e6 m/s), 0 = none", min_value=0.0,
                                  value=float(dflt["v_F_m_s"]) / 1e6, format="%.3f", key=f"mie_vf_{k}",
                                  help="Free-electron Fermi velocity, Ashcroft & Mermin Table 2.1")
            A = st.slider("A (surface scattering constant)", 0.25, 1.5, mt.DEFAULT_A, 0.01, key="mie_A",
                          help="3/4 = Kawabata-Kubo / Govorov; 6/pi^2 = Brown 2016; ~0.32 TDLDA (Lerme)")
            E_ib = st.number_input("Interband threshold E_ib (eV)", min_value=0.0,
                                   value=float(dflt["E_ib_eV"]), format="%.2f", key=f"mie_eib_{k}",
                                   help="Im eps_ib below E_ib is reported as sub-threshold residual")
            fw = st.checkbox("Field-weighted gamma_s (Shahbazyan 2016)", value=False, key="mie_fw",
                             help="gamma_s = A v_F int_S|E_n|^2 dS / int_V|E|^2 dV, 3 fixed-point steps")
        vF = vF6 * 1e6
        if vF == 0:
            st.info("Landau channel disabled: no free-electron v_F for this material "
                    "(Pd: d band at E_F, not a free-electron metal). Enter v_F to enable it.")

        st.markdown("**Near field**")
        use_peak = st.checkbox("Wavelength = lambda at max Q_abs", value=True, key="mie_nf_peak")
        lam_nf_in = st.number_input("Near-field wavelength (nm)", value=532.0, step=1.0,
                                    key="mie_nf_lam", disabled=use_peak,
                                    help="Clipped to the computed wavelength range")
        plane = st.radio("Plane", ["xz", "yz", "xy"], horizontal=True, key="mie_nf_plane",
                         format_func={"xz": "xz (k-E)", "yz": "yz (k-H)", "xy": "xy"}.get)
        hw = st.number_input("Grid half-width (units of R)", min_value=1.1, max_value=20.0, value=2.5,
                             step=0.5, key="mie_nf_hw")
        n_grid = st.number_input("Grid N", min_value=21, max_value=601, value=201, step=20, key="mie_nf_N")
        log_scale = st.radio("Colour scale", ["linear", "log10"], horizontal=True, key="mie_nf_scale") == "log10"
        use_sigma = st.checkbox("Spectra in sigma (nm^2) instead of Q", value=False, key="mie_sigma")

    with right:
        try:
            spec = _spectrum(E_tab, eps_tab, float(lam_min), float(lam_max), float(step), float(D),
                             float(n_m), bool(size_corr), float(wp), float(gb), float(vF), float(A),
                             float(E_ib), bool(fw))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Spectrum failed: {exc}")
            return
        lam = spec["lam_nm"]
        if spec["n_dropped"]:
            st.warning(f"{spec['n_dropped']} wavelength point(s) outside the table range were dropped "
                       f"(no extrapolation): computed {lam[0]:.1f}-{lam[-1]:.1f} nm.")
        clip = spec["clip"]
        if clip["n_clipped"]:
            st.caption(f"Im eps_ib < 0 clipped to 0 at {clip['n_clipped']}/{clip['n_total']} wavelengths "
                       f"(min {clip['min_im_eps_ib']:.3f}, max {100 * clip['max_rel_clip']:.1f} % of Im eps_total).")
        i = mt.peak_index(spec["Q_abs"])
        lam_nf = float(lam[i]) if use_peak else float(np.clip(lam_nf_in, lam[0], lam[-1]))
        m_nf, x_nf = near_field_inputs(spec, lam_nf, E_tab, eps_tab, wp, gb)
        with st.spinner("Near field..."):
            nf = _near_field(float(np.real(m_nf)), float(np.imag(m_nf)), float(x_nf), spec["R_nm"], plane, float(hw), int(n_grid))
        smax, sth, sph = nf["surface_max"]

        r1 = st.columns(4)
        r1[0].metric("lambda_peak (Q_abs)", f"{lam[i]:.1f} nm")
        r1[1].metric("Q_ext / Q_sca / Q_abs", f"{spec['Q_ext'][i]:.3f} / {spec['Q_sca'][i]:.3f} / {spec['Q_abs'][i]:.3f}")
        r1[2].metric("hbar gamma_surf", f"{1e3 * spec['gamma_s_eV'][i]:.1f} meV")
        r1[3].metric("eta_HE / eta_Landau", f"{spec['eta_HE'][i]:.3f} / {spec['eta_Landau'][i]:.3f}")
        r2 = st.columns(4)
        r2[0].metric("max surface |E|/|E0|", f"{smax:.2f}", help=f"at {lam_nf:.1f} nm, theta {sth:.1f} deg, phi {sph:.0f} deg (r = 1.0001 R)")
        r2[1].metric("sigma_abs at peak", f"{spec['sigma_abs'][i]:.4g} nm^2")
        r2[2].metric("sigma_HE at peak", f"{spec['sigma_interband'][i] + spec['sigma_Landau'][i]:.4g} nm^2")
        r2[3].metric("x = k R at peak", f"{spec['x'][i]:.3f}")

        st.pyplot(fig_spectra(lam, spec["Q_ext"], spec["Q_sca"], spec["Q_abs"], spec["R_nm"], use_sigma,
                              lam[i]))
        st.pyplot(fig_hot_electron(lam, spec, E_ib))
        st.caption(HE_CAVEAT)
        st.pyplot(fig_near_field(nf["X"], nf["Y"], nf["E_norm"], nf["R"], nf["xlabel"], nf["ylabel"],
                                 log_scale, f"|E|/|E0| at {lam_nf:.1f} nm, plane {plane}, n_max {nf['n_max']}"))
        meta = [f"material={name}", f"D_nm={D}", f"n_medium={n_m}", f"size_correction={size_corr}",
                f"omega_p_eV={wp}", f"gamma_b_eV={gb}", f"v_F_m_s={vF}", f"A={A}", f"E_ib_eV={E_ib}",
                f"field_weighted_gamma_s={fw}"]
        st.download_button("Download spectrum CSV", spectrum_csv(spec, meta),
                           file_name=f"mie_{Path(name).stem}_D{D:g}nm.csv", mime="text/csv", key="mie_csv")
