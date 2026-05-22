import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna

from sim_core import run_simulation


def normalize(y):
    y = np.asarray(y, dtype=float)
    ymax = np.max(y)
    if ymax <= 0:
        return y.copy()
    return y / ymax


def extract_peak_and_fwhm(wl, y):
    """
    Extract peak wavelength and baseline-corrected FWHM.

    Parameters
    ----------
    wl : ndarray
        Wavelength array.
    y : ndarray
        Spectrum array.

    Returns
    -------
    peak_wl : float
        Peak wavelength.
    fwhm : float
        Estimated FWHM in nm.
    peak_val : float
        Peak intensity.
    baseline : float
        Baseline intensity.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(wl) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan, np.nan

    peak_idx = int(np.argmax(y))
    peak_wl = wl[peak_idx]
    peak_val = y[peak_idx]

    baseline = np.min(y)
    y_corr = y - baseline
    peak_val_corr = y_corr[peak_idx]

    # Degenerate case
    if peak_val_corr <= 0:
        return peak_wl, np.nan, peak_val, baseline

    half = peak_val_corr / 2.0

    # Left crossing
    left_candidates = np.where(y_corr[:peak_idx] < half)[0]
    if len(left_candidates) > 0:
        left_idx = left_candidates[-1]
    else:
        left_idx = 0

    # Right crossing
    right_candidates = np.where(y_corr[peak_idx:] < half)[0]
    if len(right_candidates) > 0:
        right_idx = right_candidates[0] + peak_idx
    else:
        right_idx = len(y_corr) - 1

    fwhm = wl[right_idx] - wl[left_idx]

    return peak_wl, fwhm, peak_val, baseline


def spectral_loss(sim_wl, sim_fr_norm, exp_wl, exp_fr):
    """
    Combined loss:
    1. normalized spectral-shape MSE
    2. peak wavelength error
    3. FWHM error

    Returns
    -------
    total_loss : float
    exp_interp_norm : ndarray
    details : dict
    """
    exp_interp = np.interp(sim_wl, exp_wl, exp_fr)
    exp_interp_norm = normalize(exp_interp)
    sim_fr_norm = normalize(sim_fr_norm)

    # ---------------------------------------------------------
    # 1. Shape loss
    # ---------------------------------------------------------
    shape_mse = np.mean((sim_fr_norm - exp_interp_norm) ** 2)

    # ---------------------------------------------------------
    # 2. Peak loss
    # ---------------------------------------------------------
    sim_peak_wl, sim_fwhm, sim_peak_val, sim_baseline = extract_peak_and_fwhm(sim_wl, sim_fr_norm)
    exp_peak_wl, exp_fwhm, exp_peak_val, exp_baseline = extract_peak_and_fwhm(sim_wl, exp_interp_norm)

    # Safe fallback if FWHM extraction fails
    if np.isnan(sim_fwhm):
        sim_fwhm = 0.0
    if np.isnan(exp_fwhm):
        exp_fwhm = 0.0

    # Normalize peak and width errors to avoid scale mismatch
    peak_loss = ((sim_peak_wl - exp_peak_wl) / 50.0) ** 2
    fwhm_loss = ((sim_fwhm - exp_fwhm) / 80.0) ** 2

    # ---------------------------------------------------------
    # 3. Weighted total loss
    # ---------------------------------------------------------
    w_shape = 1.0
    w_peak = 1.0
    w_fwhm = 0.7

    total_loss = (
        w_shape * shape_mse
        + w_peak * peak_loss
        + w_fwhm * fwhm_loss
    )

    details = {
        "shape_mse": float(shape_mse),
        "peak_loss": float(peak_loss),
        "fwhm_loss": float(fwhm_loss),
        "sim_peak_wl": float(sim_peak_wl),
        "exp_peak_wl": float(exp_peak_wl),
        "sim_fwhm": float(sim_fwhm),
        "exp_fwhm": float(exp_fwhm),
        "sim_peak_val": float(sim_peak_val),
        "exp_peak_val": float(exp_peak_val),
        "sim_baseline": float(sim_baseline),
        "exp_baseline": float(exp_baseline),
    }

    return total_loss, exp_interp_norm, details


def make_trial_dir(root, trial_number):
    trial_dir = os.path.join(root, f"trial_{trial_number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir


def save_trial_artifacts(trial_dir, result, exp_interp_norm, params, loss, details):
    np.savetxt(
        os.path.join(trial_dir, "sim_spectrum.txt"),
        np.column_stack([
            result["lambda_nm"],
            result["F_R"],
            result["F_R_norm"],
            exp_interp_norm,
        ]),
        header="wavelength_nm F_R F_R_norm exp_F_R_norm",
    )

    np.savetxt(
        os.path.join(trial_dir, "diameters_nm.txt"),
        result["diameters_nm"],
        header="diameters_nm",
    )

    with open(os.path.join(trial_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "time": datetime.now().isoformat(),
                "loss": float(loss),
                "params": params,
                "loss_details": details,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(7, 4.5))
    plt.plot(result["lambda_nm"], result["F_R_norm"], lw=2, label="Simulation")
    plt.plot(result["lambda_nm"], exp_interp_norm, lw=2, label="Experiment")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized F(R)")
    plt.title("Simulation vs Experiment")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(trial_dir, "compare.png"), dpi=160)
    plt.close()


def build_objective(base_params, exp_wl, exp_fr, log_root):
    def objective(trial):
        params = dict(base_params)

        params["cu_mean_nm"] = trial.suggest_float("cu_mean_nm", 35.0, 50.0)
        params["cu_sigma_nm"] = trial.suggest_float("cu_sigma_nm", 8.0, 16.0)

        params["ceo2_size_nm"] = trial.suggest_float("ceo2_size_nm", 120.0, 220.0)

        params["number_ratio"] = trial.suggest_float("number_ratio", 0.4, 0.8)

        params["total_dens"] = trial.suggest_float(
            "total_dens", 5e16, 5e17, log=True
        )

        params["uv_cut_nm"] = trial.suggest_float("uv_cut_nm", 525.0, 550.0)
        params["uv_amp"] = trial.suggest_float("uv_amp", 0.05, 0.14)
        params["uv_decay_nm"] = trial.suggest_float("uv_decay_nm", 20.0, 50.0)

        params["vis_peak_nm"] = trial.suggest_float("vis_peak_nm", 500.0, 550.0)
        params["vis_width_nm"] = trial.suggest_float("vis_width_nm", 100.0, 160.0)
        params["vis_k0"] = trial.suggest_float("vis_k0", 1e-3, 8e-3, log=True)

        print(f"\n[Trial {trial.number}] START", flush=True)
        print(params, flush=True)

        print("[Trial] running simulation...", flush=True)
        result = run_simulation(params)
        print("[Trial] simulation done.", flush=True)

        loss, exp_interp_norm, details = spectral_loss(
            result["lambda_nm"],
            result["F_R_norm"],
            exp_wl,
            exp_fr
        )

        print(f"[Trial {trial.number}] total loss = {loss:.6f}", flush=True)
        print(
            f"[Trial {trial.number}] shape={details['shape_mse']:.6f}, "
            f"peak={details['peak_loss']:.6f}, "
            f"fwhm={details['fwhm_loss']:.6f}",
            flush=True
        )
        print(
            f"[Trial {trial.number}] sim_peak={details['sim_peak_wl']:.2f} nm, "
            f"exp_peak={details['exp_peak_wl']:.2f} nm, "
            f"sim_fwhm={details['sim_fwhm']:.2f} nm, "
            f"exp_fwhm={details['exp_fwhm']:.2f} nm",
            flush=True
        )

        trial_dir = make_trial_dir(log_root, trial.number)

        print("[Trial] saving artifacts...", flush=True)
        save_trial_artifacts(
            trial_dir,
            result,
            exp_interp_norm,
            params,
            loss,
            details,
        )
        print("[Trial] saved.", flush=True)

        return loss

    return objective