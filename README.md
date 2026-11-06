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
