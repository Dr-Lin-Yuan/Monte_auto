# Monte_auto — plasmonic diffuse-reflectance fitting + Mie toolkit

Streamlit app with two tabs:

1. **Monte Carlo fit** — fits Kubelka-Munk F(R) diffuse-reflectance spectra of
   metal nanoparticles (Cu, AuPd, …) dispersed on a CeO2 or N-doped TiO2 support.
   Pipeline: tabulated dielectric function → vectorised Mie cross sections for a
   truncated-normal particle-size ensemble (LUT + quadrature) and the support sphere
   (full Mie) → GPU (numba.cuda) 1-D slab photon-transport Monte Carlo, CPU fallback →
   F(R) = (1−R)²/2R → Optuna TPE search (low-photon search, full-photon refinement,
   SQLite persistence, cancel button).
2. **Mie toolkit** — standalone single-sphere calculator (numpy/scipy only):
   Q_ext / Q_sca / Q_abs spectra, Kreibig size correction γ = γ_b + A v_F/R,
   semi-classical damping-channel split of σ_abs into interband / Landau-surface /
   bulk-intraband / sub-threshold-residual (shared internal field ⇒ weights =
   Im ε_channel / Im ε_total; logistic interband onset gate; optional Shahbazyan
   field-weighted γ_s), and Bohren–Huffman near-field |E|/E0 maps (inside + outside)
   in the xz, yz or xy plane. Validated against miepython 3.3.1 and scattnlay 2.4
   (≤ 6e-7) and against the Govorov hot-electron efficiency (≤ 0.002).

## Theory

Notation: vacuum wavelength $`\lambda`$, photon energy $`E = hc/\lambda`$ with
$`hc = 1239.84193\ \text{eV nm}`$, sphere diameter $`D = 2R`$, real host index $`n_m`$,
time dependence $`e^{-i\omega t}`$ (Bohren & Huffman 1983, "BH"), so absorbing materials
have $`\mathrm{Im}\,\varepsilon \ge 0`$.

### 1. Mie theory for one sphere (both tabs)

Size parameter and relative refractive index:

```math
x = k R = \frac{2\pi n_m R}{\lambda}, \qquad
m = \frac{\sqrt{\varepsilon(\lambda)}}{n_m} = \frac{n + i\kappa}{n_m}
```

Scattering coefficients (BH eq. 4.53, non-magnetic sphere):

```math
a_n = \frac{m\,\psi_n(mx)\,\psi_n'(x) - \psi_n(x)\,\psi_n'(mx)}
           {m\,\psi_n(mx)\,\xi_n'(x) - \xi_n(x)\,\psi_n'(mx)}, \qquad
b_n = \frac{\psi_n(mx)\,\psi_n'(x) - m\,\psi_n(x)\,\psi_n'(mx)}
           {\psi_n(mx)\,\xi_n'(x) - m\,\xi_n(x)\,\psi_n'(mx)}
```

with the Riccati–Bessel functions $`\psi_n(z) = z\,j_n(z)`$ and
$`\xi_n(z) = z\,h_n^{(1)}(z)`$. Efficiencies (BH eqs. 4.61–4.62):

```math
Q_\text{ext} = \frac{2}{x^2}\sum_{n=1}^{n_\max}(2n+1)\,\mathrm{Re}\,(a_n + b_n), \qquad
Q_\text{sca} = \frac{2}{x^2}\sum_{n=1}^{n_\max}(2n+1)\left(\lvert a_n\rvert^2 + \lvert b_n\rvert^2\right), \qquad
Q_\text{abs} = Q_\text{ext} - Q_\text{sca}
```

```math
C_j = Q_j\,\pi R^2 \quad (j = \text{ext},\ \text{sca},\ \text{abs}), \qquad
n_\max = \left\lceil x + 4.05\,x^{1/3} + 2 \right\rceil \ \ \text{(Wiscombe 1980)}
```

| Symbol | Meaning | Unit |
|---|---|---|
| $`x`$ | size parameter = sphere circumference / wavelength in the host | – |
| $`m`$ | complex refractive index of the sphere relative to the host | – |
| $`\varepsilon = \varepsilon_1 + i\varepsilon_2`$ | tabulated permittivity of the sphere material | – |
| $`a_n,\ b_n`$ | electric / magnetic multipole coefficients of order $`n`$ ($`a_1`$ = electric dipole = LSPR) | – |
| $`j_n,\ h_n^{(1)}`$ | spherical Bessel / Hankel functions of the first kind | – |
| $`Q_\text{ext},\ Q_\text{sca},\ Q_\text{abs}`$ | extinction / scattering / absorption efficiency = cross section ÷ $`\pi R^2`$ | – |
| $`C_\text{ext},\ C_\text{sca},\ C_\text{abs}`$ | cross sections ($`\sigma`$ in the Mie tab) | nm² (m² inside the MC) |
| $`n_\max`$ | series truncation order | – |

### 2. Monte Carlo fit: from cross sections to F(R)

**2.1 Metal-particle size ensemble.** Diameters follow a normal distribution
$`\mathcal N(\mu_D, \sigma_D)`$ truncated to $`[D_\text{lo}, D_\text{hi}]`$. Cross sections are
pre-computed once on a diameter grid $`D_i`$ (look-up table, default 160 points) and
averaged with the normalised truncated-normal weights $`w_i`$:

```math
w_i = \frac{p(D_i)}{\sum_j p(D_j)}, \qquad
p(D) \propto \exp\!\left[-\frac{(D-\mu_D)^2}{2\sigma_D^2}\right]\ \text{for } D_\text{lo}\le D\le D_\text{hi}
```

```math
\langle C_\text{ext}\rangle(\lambda) = \sum_i w_i\,C_\text{ext}(D_i,\lambda), \qquad
\langle C_\text{abs}\rangle(\lambda) = \sum_i w_i\,C_\text{abs}(D_i,\lambda)
```

**2.2 Support sphere.** One sphere of diameter $`D_s`$ with a parametric complex index
$`n_s(\lambda) + i k_s(\lambda)`$, solved with the same full Mie kernel in the same host
$`n_m`$ ($`\lambda`$ in nm; $`\Theta`$ = Heaviside step):

```math
\text{CeO}_2:\quad n_s = 2.2 + 0.15\,\exp\!\left[-\frac{(\lambda-350)^2}{2\cdot120^2}\right], \qquad
k_s = A_\text{uv}\,e^{-(\lambda-350)/\tau_\text{uv}}\,\Theta(\lambda_\text{cut}-\lambda)
      + k_0\,\exp\!\left[-\frac{(\lambda-\lambda_\text{vis})^2}{2w_\text{vis}^2}\right]
```

```math
\text{N-doped TiO}_2:\quad n_s = \sqrt{5.913 + \frac{0.2441}{\lambda_{\mu\text{m}}^2 - 0.0803}}\ \ \text{(rutile, DeVore 1951)}, \qquad
k_s = A_\text{uv}\,e^{-(\lambda-350)/\tau_\text{uv}}\,\Theta(400-\lambda)
      + k_0\,\exp\!\left[-\frac{(\lambda-\lambda_\text{vis})^2}{2w_\text{vis}^2}\right]
```

| Symbol | Code name (CeO2 / TiO2) | Meaning |
|---|---|---|
| $`D_s`$ | `ceo2_size_nm` / `support_size_nm` | support sphere diameter (nm) |
| $`A_\text{uv},\ \tau_\text{uv}`$ | `uv_amp`, `uv_decay_nm` / `tio2_uv_amp`, `tio2_uv_decay_nm` | amplitude and decay length (nm) of the band-edge absorption tail |
| $`\lambda_\text{cut}`$ | `uv_cut_nm` / fixed 400 nm | wavelength above which the band-edge tail is switched off |
| $`k_0,\ \lambda_\text{vis},\ w_\text{vis}`$ | `vis_k0`, `vis_peak_nm`, `vis_width_nm` / `tio2_vis_*` | height, centre and width (nm) of a Gaussian visible absorption band (defects / dopant states) |

**2.3 Powder optical coefficients.** With total number density $`N_\text{tot}`$ and
metal-to-support number ratio $`\rho`$:

```math
N_s = \frac{N_\text{tot}}{1+\rho}, \qquad N_\text{NP} = \frac{\rho\,N_\text{tot}}{1+\rho}
```

```math
\alpha_\text{NP} = N_\text{NP}\,\langle C_\text{ext}\rangle, \qquad
\alpha_s = N_s\,C_\text{ext}^{s}, \qquad
\alpha_\text{tot} = \alpha_\text{NP} + \alpha_s
```

```math
P_\text{NP} = \frac{\alpha_\text{NP}}{\alpha_\text{tot}}, \qquad
p_\text{abs}^\text{NP} = \frac{\langle C_\text{abs}\rangle}{\langle C_\text{ext}\rangle}, \qquad
p_\text{abs}^{s} = \frac{C_\text{abs}^{s}}{C_\text{ext}^{s}}
```

| Symbol | Code name | Meaning | Unit |
|---|---|---|---|
| $`N_\text{tot}`$ | `total_dens` | number of spheres (metal + support) per volume of powder | m⁻³ |
| $`\rho`$ | `number_ratio` | $`N_\text{NP}/N_s`$, metal particles per support sphere | – |
| $`\alpha_\text{tot}`$ | `alpha_tot` | attenuation coefficient; $`1/\alpha_\text{tot}`$ = mean free path between collisions | m⁻¹ |
| $`P_\text{NP}`$ | `choose_np` | probability that a collision is with a metal particle (otherwise support) | – |
| $`p_\text{abs}`$ | `p_abs_np_eff`, `p_abs_sup` | probability that the collision absorbs the photon (= 1 − single-scattering albedo) | – |
| $`h`$ | `h_sample_m` | powder-bed thickness (default 2 mm) | m |

**2.4 Photon random walk (1-D slab, each wavelength independent).** A photon enters at
$`z = 0`$ travelling in $`+z`$. With $`u`$ uniform in $`(0,1)`$:

```math
\Delta z = -\frac{\ln u}{\alpha_\text{tot}}, \qquad z \leftarrow z \pm \Delta z
```

1. $`z \le 0`$: the photon leaves through the front face → counted as **reflected**.
2. $`z \ge h`$: mirror at the back face, the photon turns around (no transmission).
3. Otherwise a collision: metal particle with probability $`P_\text{NP}`$, else support;
   absorbed with probability $`p_\text{abs}`$ → counted as **absorbed**; if not absorbed
   it is scattered into $`+z`$ or $`-z`$ with probability ½ each.
4. A photon still alive after `max_steps_per_photon` (3000) collisions counts as reflected.

`fast_hybrid_gpu` uses the ensemble-averaged $`p_\text{abs}^\text{NP}`$ above;
`legacy_exact_powder` (CPU) instead draws an individual particle $`j`$ with probability
$`C_{\text{ext},j}/\sum_j C_{\text{ext},j}`$ and uses its own $`C_{\text{abs},j}/C_{\text{ext},j}`$.

**2.5 Reflectance and Kubelka–Munk function.**

```math
R(\lambda) = \frac{N_\text{reflected}}{N_\text{photons}}, \qquad
F(R) = \frac{(1-R)^2}{2R} = \frac{K}{S}, \qquad
\hat F(\lambda) = \frac{F(R)}{\max_\lambda F(R)}
```

$`R`$ is the diffuse reflectance of the optically thick bed, $`K`$ and $`S`$ the
Kubelka–Munk absorption and scattering coefficients, and $`\hat F`$ the peak-normalised
spectrum that is compared with the (equally normalised) experiment.

**2.6 Loss functions** ($`\hat F_\text{sim}`$, $`\hat F_\text{exp}`$ on the simulation
wavelength grid $`\lambda_i`$; all wavelengths in nm).

`legacy` (Cu / CeO2):

```math
\mathcal L = \frac{1}{N}\sum_i \tilde w_i\left(\hat F_{\text{sim},i} - \hat F_{\text{exp},i}\right)^2
 + \left(\frac{\Delta\lambda_\text{pk}}{50}\right)^2
 + 0.7\left(\frac{\Delta\text{FWHM}}{80}\right)^2
 + 1.2\left(\frac{\Delta \hat F_\text{val}}{0.05}\right)^2
 + 0.8\left(\frac{\Delta\lambda_\text{val}}{10}\right)^2
 + 0.8\,\frac{(\delta/0.20)^2}{1 + \lvert\delta\rvert/0.20}
```

| Term | Meaning |
|---|---|
| $`\tilde w_i`$ | point weight: 3 for 535–555 nm, 1.5 for 620–690 nm, 1 elsewhere, rescaled to mean 1 |
| $`\Delta\lambda_\text{pk}`$ | simulated − experimental wavelength of the spectrum maximum |
| $`\Delta\text{FWHM}`$ | difference of the full widths at half maximum (measured above the spectrum minimum) |
| $`\Delta\hat F_\text{val},\ \Delta\lambda_\text{val}`$ | difference in depth and position of the minimum inside 535–555 nm |
| $`\delta`$ | difference of the mean $`\hat F`$ over 700–750 nm (baseline tail); pseudo-Huber form, quadratic for small and linear for large $`\delta`$ |

`AuPd_TiO2_v1`:

```math
\mathcal L = 0.20\,\frac{1}{N}\sum_i\left(\hat F_{\text{sim},i} - \hat F_{\text{exp},i}\right)^2
 + 0.15\,\frac{\lvert\Delta\lambda_\text{pk}\rvert}{40}
 + 0.20\,(\Delta \hat F_\text{pk})^2
 + 0.30\,\frac{1}{N_t}\sum_{i\in\text{tail}} \tilde w_i\left(\hat F_{\text{sim},i} - \hat F_{\text{exp},i}\right)^2
 + 0.15\,\frac{1}{N}\sum_i\left(\frac{d\hat F_\text{sim}}{d\lambda} - \frac{d\hat F_\text{exp}}{d\lambda}\right)_i^2
```

| Term | Meaning |
|---|---|
| $`\Delta\lambda_\text{pk},\ \Delta\hat F_\text{pk}`$ | difference in position / height of the maximum inside the 480–620 nm peak window |
| tail | 650–800 nm, weights $`\tilde w_i`$ rising linearly with wavelength (2:1 from end to start), rescaled to mean 1 |
| derivative term | mean squared difference of the spectral slopes (finite differences) |

**2.7 Optimisation.** Optuna (TPE) minimises $`\mathcal L`$ over

| Parameter | Symbol | Default search range |
|---|---|---|
| `cu_mean_nm`, `cu_sigma_nm` | $`\mu_D,\ \sigma_D`$ | 15–60 nm, 2–15 nm |
| `ceo2_size_nm` / `support_size_nm` | $`D_s`$ | 10–220 nm |
| `number_ratio` | $`\rho`$ | 0.05–1 |
| `total_dens` | $`N_\text{tot}`$ | 10¹⁷–5×10¹⁸ m⁻³ (log) |
| CeO2: `uv_cut_nm`, `uv_amp`, `uv_decay_nm` | $`\lambda_\text{cut},\ A_\text{uv},\ \tau_\text{uv}`$ | 500–560 nm, 0.02–0.20, 20–80 nm |
| CeO2: `vis_peak_nm`, `vis_width_nm`, `vis_k0` | $`\lambda_\text{vis},\ w_\text{vis},\ k_0`$ | 500–620 nm, 40–180 nm, 10⁻⁴–2×10⁻² (log) |
| TiO2: `tio2_uv_amp`, `tio2_uv_decay_nm` | $`A_\text{uv},\ \tau_\text{uv}`$ | 0.02–0.20, 10–80 nm |
| TiO2: `tio2_vis_peak_nm`, `tio2_vis_width_nm`, `tio2_vis_k0` | $`\lambda_\text{vis},\ w_\text{vis},\ k_0`$ | 480–650 nm, 60–260 nm, 10⁻⁴–2×10⁻² (log) |

Search trials run with a reduced photon number (default 10⁴); the best trial is then
re-evaluated with the full `n_photons`. $`n_m`$, $`h`$, $`D_\text{lo}`$, $`D_\text{hi}`$
are fixed inputs.

### 3. Mie toolkit: size correction, hot-carrier channels, near field

**3.1 Drude term and surface (Kreibig) damping.** All energies in eV:

```math
\varepsilon_D(E;\gamma) = -\frac{E_p^2}{E\,(E + i\gamma)}, \qquad
\gamma = \gamma_b + \gamma_s, \qquad
\gamma_s = A\,\frac{\hbar v_F}{R}
```

```math
\varepsilon_\text{sc}(E) = \varepsilon_\text{tab}(E) - \varepsilon_D(E;\gamma_b) + \varepsilon_D(E;\gamma_b+\gamma_s)
```

The bulk Drude term inside the tabulated permittivity is replaced by a surface-broadened
one; $`\varepsilon_\text{sc}`$ then enters Section 1 through $`m = \sqrt{\varepsilon_\text{sc}}/n_m`$.

| Symbol | Meaning | Unit |
|---|---|---|
| $`\varepsilon_\text{tab}`$ | tabulated bulk permittivity (PCHIP-interpolated in energy, never extrapolated) | – |
| $`\varepsilon_\text{sc}`$ | size-corrected permittivity actually used in the Mie calculation | – |
| $`E_p = \hbar\omega_p`$ | Drude plasma energy | eV |
| $`\gamma_b`$ | bulk Drude damping (electron–phonon, electron–electron, defects) | eV |
| $`\gamma_s`$ | surface-scattering (Landau) damping of a sphere of radius $`R`$ | eV |
| $`A`$ | surface-scattering constant, default 0.75 (Kawabata & Kubo 1966) | – |
| $`v_F`$ | Fermi velocity | m s⁻¹ |

Built-in defaults:

| Metal | $`E_p`$ (eV) | $`\gamma_b`$ (eV) | $`v_F`$ (10⁶ m s⁻¹) | $`E_\text{ib}`$ (eV) |
|---|---|---|---|---|
| Au | 9.066 | 0.0735 | 1.40 | 1.9 |
| Ag | 9.175 | 0.0190 | 1.39 | 3.9 |
| Cu | 8.871 | 0.1018 | 1.57 | 2.1 |
| Pd | 9.72·√0.330 | 0.008 | – (size correction off) | 0 |

Au/Ag/Cu: Drude fit to Johnson & Christy 1972 (0.64–1.3 eV), $`v_F`$ from
Ashcroft & Mermin; Pd: Rakić 1998 Lorentz–Drude intraband term.

Optional field-weighted surface damping (Shahbazyan 2016), iterated to self-consistency;
it reduces to $`A\hbar v_F/R`$ for a uniform internal field:

```math
\gamma_s(\lambda) = A\,\hbar v_F\,
\frac{\oint_S \lvert E_n\rvert^2\,dS}{\int_V \lvert \mathbf E\rvert^2\,dV}
```

with $`E_n`$ the normal field component just inside the surface $`S`$ and $`V`$ the
sphere volume.

**3.2 Damping-channel split of the absorption.** All channels are driven by the same
internal field, so each takes the share of $`\sigma_\text{abs}`$ given by its share of
$`\mathrm{Im}\,\varepsilon`$:

```math
\mathrm{Im}\,\varepsilon_\text{ib} = \max\!\left\{0,\ \mathrm{Im}\left[\varepsilon_\text{tab} - \varepsilon_D(\gamma_b)\right]\right\}, \qquad
\mathrm{Im}\,\varepsilon_\text{tot} = \mathrm{Im}\,\varepsilon_\text{ib} + \mathrm{Im}\,\varepsilon_D(\gamma)
```

```math
s(E) = \frac{1}{1 + \exp\!\left[-(E - E_\text{ib})/w\right]}
```

```math
f_\text{interband} = \frac{s\,\mathrm{Im}\,\varepsilon_\text{ib}}{\mathrm{Im}\,\varepsilon_\text{tot}}, \quad
f_\text{Landau} = \frac{\gamma_s}{\gamma}\,\frac{\mathrm{Im}\,\varepsilon_D(\gamma)}{\mathrm{Im}\,\varepsilon_\text{tot}}, \quad
f_\text{bulk} = \frac{\gamma_b}{\gamma}\,\frac{\mathrm{Im}\,\varepsilon_D(\gamma)}{\mathrm{Im}\,\varepsilon_\text{tot}}, \quad
f_\text{residual} = \frac{(1-s)\,\mathrm{Im}\,\varepsilon_\text{ib}}{\mathrm{Im}\,\varepsilon_\text{tot}}
```

```math
\sigma_\text{ch} = f_\text{ch}\,\sigma_\text{abs}, \qquad \sum_\text{ch} f_\text{ch} = 1, \qquad
\eta_\text{HE} = f_\text{interband} + f_\text{Landau}
```

| Symbol | Meaning |
|---|---|
| $`\mathrm{Im}\,\varepsilon_\text{ib}`$ | non-Drude (bound-electron) part of the tabulated loss |
| $`s(E)`$ | logistic interband onset gate: 0 well below $`E_\text{ib}`$, ½ at $`E_\text{ib}`$, 1 above |
| $`E_\text{ib},\ w`$ | interband threshold and gate width (default 0.1 eV; $`w \le 0`$ gives a hard step) |
| $`f_\text{interband}`$ | d → sp interband transitions (hot holes in the d band) |
| $`f_\text{Landau}`$ | surface-assisted intraband absorption (Landau damping) → energetic hot electrons |
| $`f_\text{bulk}`$ | bulk Drude (phonon / defect-assisted) intraband absorption → low-energy carriers, heat |
| $`f_\text{residual}`$ | non-Drude loss below the interband threshold (table / Drude-fit mismatch), not assigned to a physical channel |
| $`\eta_\text{HE}`$ | fraction of absorbed power going into the two hot-carrier channels |

This is a semi-classical bookkeeping of the dissipated power (Govorov-type; Besteiro et al.,
*ACS Photonics* **4**, 2759 (2017)), not an ab initio carrier-generation rate.

**3.3 Near field.** Incident plane wave
$`\mathbf E_\text{inc} = E_0\,\hat{\mathbf x}\,e^{ikz}`$ (polarised along $`x`$,
propagating along $`z`$). Vector-spherical-harmonic expansion (BH eqs. 4.40, 4.45) with
$`E_n = i^n E_0\,(2n+1)/[n(n+1)]`$:

```math
\mathbf E(r>R) = \mathbf E_\text{inc} + \sum_{n=1}^{n_\max} E_n\left(i a_n\,\mathbf N^{(3)}_{e1n} - b_n\,\mathbf M^{(3)}_{o1n}\right), \qquad
\mathbf E(r<R) = \sum_{n=1}^{n_\max} E_n\left(c_n\,\mathbf M^{(1)}_{o1n} - i d_n\,\mathbf N^{(1)}_{e1n}\right)
```

```math
c_n = \frac{i\,m}{\psi_n(mx)\,\xi_n'(x) - m\,\xi_n(x)\,\psi_n'(mx)}, \qquad
d_n = \frac{i\,m}{m\,\psi_n(mx)\,\xi_n'(x) - \xi_n(x)\,\psi_n'(mx)}
```

$`\mathbf M,\ \mathbf N`$ are the vector spherical harmonics (radial function
$`h_n^{(1)}(kr)`$ outside, superscript (3); $`j_n(mkr)`$ inside, superscript (1)), and
$`c_n,\ d_n`$ the internal-field coefficients. The maps show the enhancement
$`\lvert\mathbf E\rvert/E_0`$ in the $`xz`$ (contains $`\mathbf E_0`$ and $`\mathbf k`$),
$`yz`$ or $`xy`$ plane; the radial component jumps by $`m^2`$ across the surface.

## Run

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

GPU Monte Carlo needs an NVIDIA driver + numba CUDA; without one the app falls back to
the CPU `legacy_exact_powder` path (use small photon counts). Command-line fitting:
`python run_fit.py` (edit `base_params` in the file).

## Tests

```bash
python -m pytest tests/
```

`tests/test_mie.py` — vectorised Mie vs scalar reference, Rayleigh / extinction-paradox
limits, support-model smoothness, LUT consistency, pipeline smoke.
`tests/test_mie_toolkit.py` — Bohren–Huffman / Wiscombe benchmarks, size-correction
sign, channel weights, Govorov cross-check, near-field continuity and far-field limit.

## Data

`dielectric_permittivity/` — tabulated ε or n,k tables. Formats auto-detected:
`E_eV eps1 eps2`, `lambda_nm n k`, `lambda_um n k`; `#` comments; comma or whitespace.
Johnson & Christy 1972/1974 Au, Ag, Cu, Pd (`*_nk.txt`) ship with the repo; the
`dielectric_function_*.txt` files are experimental alloy permittivities.

## Layout

| File | Role |
|---|---|
| `sim_core.py` | Mie (shared vectorised Riccati–Bessel kernel), support models, MC kernel, `run_simulation` |
| `fit_agent.py` | loss functions (`legacy`, `AuPd_TiO2_v1`), Optuna objective, trial artefacts |
| `fit_pipeline.py` | parameter assembly, two-phase Optuna study, run directories |
| `streamlit_app.py` | UI shell and the Monte Carlo tab |
| `mie_toolkit.py` | pure-physics single-sphere library used by the Mie tab |
| `mie_tab.py` | Mie tab UI and figure builders |
| `run_fit.py`, `inspect_best.py` | CLI fitting and inspection |
| `*_v1.py`, `sim_core_cpu.py` | legacy references |

## Acknowledgement and references

The modelling ideas used here — treating a plasmonic photocatalyst powder as an ensemble
of Mie absorbers/scatterers on an oxide support, following photons through the bed with a
Monte Carlo random walk to separate absorption from multiple scattering, and splitting the
absorbed power into hot-carrier generation channels — were learned from the
**Nordlander and Halas groups at Rice University**, and from the papers below.

Besides my advisors, Prof. Naomi J. Halas and Prof. Peter Nordlander, I learned most of
this as a graduate student from Dr. Minhan Lou, Dr. Ming Zhang, Dr. Jingyi Zhou,
Dr. Yage Zhao and Anvy Kuriakose (Nordlander group alumni), and from my senior labmates
Dr. Linan Zhou and Dr. Chao Zhang (Halas group alumni, as am I). Thank you.

**Disclaimer.** This repository is an independent re-implementation written for my own
fitting work. It is not the original code of the Nordlander or Halas groups, and it has
not been reviewed or endorsed by anyone named above; any errors are mine alone. The
software is provided "as is" under the MIT license, without warranty of any kind —
please validate the results against your own data before relying on them.

1. L. Zhou, D. F. Swearer, C. Zhang, H. Robatjazi, H. Zhao, L. Henderson, L. Dong,
   P. Christopher, E. A. Carter, P. Nordlander, N. J. Halas, "Quantifying hot carrier and
   thermal contributions in plasmonic photocatalysis", *Science* **362**, 69–72 (2018).
   [doi:10.1126/science.aat6967](https://doi.org/10.1126/science.aat6967)
2. L. Yuan, M. Lou, B. D. Clark, M. Lou, L. Zhou, S. Tian, C. R. Jacobson, P. Nordlander,
   N. J. Halas, "Morphology-Dependent Reactivity of a Plasmonic Photocatalyst",
   *ACS Nano* **14**, 12054–12063 (2020).
   [doi:10.1021/acsnano.0c05383](https://doi.org/10.1021/acsnano.0c05383)
3. L. Zhou, M. Lou, J. L. Bao, C. Zhang, J. G. Liu, J. M. P. Martirez, S. Tian, L. Yuan,
   D. F. Swearer, H. Robatjazi, E. A. Carter, P. Nordlander, N. J. Halas, "Hot carrier
   multiplication in plasmonic photocatalysis", *Proc. Natl. Acad. Sci. U.S.A.* **118**,
   e2022109118 (2021).
   [doi:10.1073/pnas.2022109118](https://doi.org/10.1073/pnas.2022109118)
4. Y. Yuan, L. Zhou, H. Robatjazi, J. L. Bao, J. Zhou, A. Bayles, L. Yuan, M. Lou, M. Lou,
   S. Khatiwada, E. A. Carter, P. Nordlander, N. J. Halas, "Earth-abundant photocatalyst
   for H<sub>2</sub> generation from NH<sub>3</sub> with light-emitting diode
   illumination", *Science* **378**, 889–893 (2022).
   [doi:10.1126/science.abn5636](https://doi.org/10.1126/science.abn5636)
5. S. Lee, L. Yuan, E. Begin, D. Yang, C. Lim, Y. A. Zhang, L. Ma, C. Ophus, Y. Cui,
   J. L. Bao, J. A. Dionne, "Plasmonic Photocatalysis Enables Selective Oxidative Coupling
   of Methane with Nitrous Oxide under Ambient Conditions", arXiv:2604.18417 (2026).
   [arXiv:2604.18417](https://arxiv.org/abs/2604.18417)
