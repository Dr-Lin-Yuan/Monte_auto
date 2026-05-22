import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from sim_core import run_simulation


def normalize(y):
    y = np.asarray(y, dtype=float)
    ymax = np.max(y)
    if ymax <= 0:
        return y.copy()
    return y / ymax


def interpolate_and_normalize(sim_wl, sim_y, exp_wl, exp_y):
    sim_wl = np.asarray(sim_wl, dtype=float)
    sim_y_norm = normalize(np.asarray(sim_y, dtype=float))
    exp_interp = np.interp(sim_wl, exp_wl, exp_y)
    exp_interp_norm = normalize(exp_interp)
    return sim_wl, sim_y_norm, exp_interp_norm


def extract_peak_and_fwhm(wl, y):
    """
    Extract peak wavelength and baseline-corrected FWHM.
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

    if peak_val_corr <= 0:
        return peak_wl, np.nan, peak_val, baseline

    half = peak_val_corr / 2.0

    left_candidates = np.where(y_corr[:peak_idx] < half)[0]
    if len(left_candidates) > 0:
        left_idx = left_candidates[-1]
    else:
        left_idx = 0

    right_candidates = np.where(y_corr[peak_idx:] < half)[0]
    if len(right_candidates) > 0:
        right_idx = right_candidates[0] + peak_idx
    else:
        right_idx = len(y_corr) - 1

    fwhm = wl[right_idx] - wl[left_idx]

    return peak_wl, fwhm, peak_val, baseline


def legacy_loss(sim_wl, sim_fr_norm, exp_wl, exp_fr):
    """
    Combined loss:
    1. weighted spectral-shape MSE
    2. peak wavelength error
    3. FWHM error
    4. valley depth error
    5. valley position error
    """
    sim_wl, sim_fr_norm, exp_interp_norm = interpolate_and_normalize(
        sim_wl, sim_fr_norm, exp_wl, exp_fr
    )

    weights = np.ones_like(sim_wl, dtype=float)

    dip_mask = (sim_wl >= 535.0) & (sim_wl <= 555.0)
    weights[dip_mask] = 3.0

    tail_mask = (sim_wl >= 620.0) & (sim_wl <= 690.0)
    weights[tail_mask] = 1.5

    weights = weights / np.mean(weights)

    shape_mse = np.mean(weights * (sim_fr_norm - exp_interp_norm) ** 2)

    sim_peak_wl, sim_fwhm, sim_peak_val, sim_baseline = extract_peak_and_fwhm(sim_wl, sim_fr_norm)
    exp_peak_wl, exp_fwhm, exp_peak_val, exp_baseline = extract_peak_and_fwhm(sim_wl, exp_interp_norm)

    if np.isnan(sim_fwhm):
        sim_fwhm = 0.0
    if np.isnan(exp_fwhm):
        exp_fwhm = 0.0

    peak_loss = ((sim_peak_wl - exp_peak_wl) / 50.0) ** 2
    fwhm_loss = ((sim_fwhm - exp_fwhm) / 80.0) ** 2

    valley_mask = (sim_wl >= 535.0) & (sim_wl <= 555.0)

    if np.any(valley_mask):
        sim_valley_region_wl = sim_wl[valley_mask]
        sim_valley_region_y = sim_fr_norm[valley_mask]
        exp_valley_region_y = exp_interp_norm[valley_mask]

        sim_valley_idx = int(np.argmin(sim_valley_region_y))
        exp_valley_idx = int(np.argmin(exp_valley_region_y))

        sim_valley_wl = sim_valley_region_wl[sim_valley_idx]
        exp_valley_wl = sim_valley_region_wl[exp_valley_idx]

        sim_valley_y = sim_valley_region_y[sim_valley_idx]
        exp_valley_y = exp_valley_region_y[exp_valley_idx]

        valley_depth_loss = ((sim_valley_y - exp_valley_y) / 0.05) ** 2
        valley_pos_loss = ((sim_valley_wl - exp_valley_wl) / 10.0) ** 2
    else:
        sim_valley_wl = np.nan
        exp_valley_wl = np.nan
        sim_valley_y = np.nan
        exp_valley_y = np.nan
        valley_depth_loss = 0.0
        valley_pos_loss = 0.0

    w_shape = 1.0
    w_peak = 1.0
    w_fwhm = 0.7
    w_valley_depth = 1.2
    w_valley_pos = 0.8

    total_loss = (
        w_shape * shape_mse
        + w_peak * peak_loss
        + w_fwhm * fwhm_loss
        + w_valley_depth * valley_depth_loss
        + w_valley_pos * valley_pos_loss
    )

    details = {
        "shape_mse": float(shape_mse),
        "peak_loss": float(peak_loss),
        "fwhm_loss": float(fwhm_loss),
        "valley_depth_loss": float(valley_depth_loss),
        "valley_pos_loss": float(valley_pos_loss),
        "sim_peak_wl": float(sim_peak_wl),
        "exp_peak_wl": float(exp_peak_wl),
        "sim_fwhm": float(sim_fwhm),
        "exp_fwhm": float(exp_fwhm),
        "sim_peak_val": float(sim_peak_val),
        "exp_peak_val": float(exp_peak_val),
        "sim_baseline": float(sim_baseline),
        "exp_baseline": float(exp_baseline),
        "sim_valley_wl": float(sim_valley_wl) if not np.isnan(sim_valley_wl) else None,
        "exp_valley_wl": float(exp_valley_wl) if not np.isnan(exp_valley_wl) else None,
        "sim_valley_y": float(sim_valley_y) if not np.isnan(sim_valley_y) else None,
        "exp_valley_y": float(exp_valley_y) if not np.isnan(exp_valley_y) else None,
    }

    return total_loss, exp_interp_norm, details


def _peak_metrics(wl, y, peak_window):
    mask = (wl >= peak_window[0]) & (wl <= peak_window[1])
    if not np.any(mask):
        idx = int(np.argmax(y))
        return float(wl[idx]), float(y[idx])

    wl_win = wl[mask]
    y_win = y[mask]
    idx = int(np.argmax(y_win))
    return float(wl_win[idx]), float(y_win[idx])


def aupd_tio2_v1_loss(sim_wl, sim_y, exp_wl, exp_y, loss_config=None):
    sim_wl, sim_y_norm, exp_interp_norm = interpolate_and_normalize(
        sim_wl, sim_y, exp_wl, exp_y
    )

    config = {
        "weights": {
            "shape": 0.20,
            "peak_position": 0.15,
            "peak_height": 0.20,
            "tail": 0.30,
            "derivative": 0.15,
        },
        "peak_window_nm": (480.0, 620.0),
        "tail_window_nm": (650.0, 800.0),
        "peak_position_scale_nm": 40.0,
    }
    if loss_config:
        config.update(loss_config)
        if "weights" in loss_config:
            config["weights"].update(loss_config["weights"])

    weights = config["weights"]

    shape_loss = np.mean((sim_y_norm - exp_interp_norm) ** 2)

    sim_peak_wl, sim_peak_height = _peak_metrics(sim_wl, sim_y_norm, config["peak_window_nm"])
    exp_peak_wl, exp_peak_height = _peak_metrics(sim_wl, exp_interp_norm, config["peak_window_nm"])
    peak_position_loss = abs(sim_peak_wl - exp_peak_wl) / config["peak_position_scale_nm"]
    peak_height_loss = (sim_peak_height - exp_peak_height) ** 2

    tail_mask = (sim_wl >= config["tail_window_nm"][0]) & (sim_wl <= config["tail_window_nm"][1])
    if np.any(tail_mask):
        tail_wl = sim_wl[tail_mask]
        tail_weights = np.linspace(1.0, 2.0, tail_wl.size)
        tail_weights = tail_weights / np.mean(tail_weights)
        tail_diff_sq = (sim_y_norm[tail_mask] - exp_interp_norm[tail_mask]) ** 2
        tail_loss = np.mean(tail_weights * tail_diff_sq)
    else:
        tail_loss = 0.0

    if sim_wl.size >= 2:
        sim_grad = np.gradient(sim_y_norm, sim_wl)
        exp_grad = np.gradient(exp_interp_norm, sim_wl)
        derivative_loss = np.mean((sim_grad - exp_grad) ** 2)
    else:
        derivative_loss = 0.0

    total_loss = (
        weights["shape"] * shape_loss
        + weights["peak_position"] * peak_position_loss
        + weights["peak_height"] * peak_height_loss
        + weights["tail"] * tail_loss
        + weights["derivative"] * derivative_loss
    )

    details = {
        "loss_mode": "AuPd_TiO2_v1",
        "shape_loss": float(shape_loss),
        "peak_pos_loss": float(peak_position_loss),
        "peak_height_loss": float(peak_height_loss),
        "tail_loss": float(tail_loss),
        "deriv_loss": float(derivative_loss),
        "total_loss": float(total_loss),
        "sim_peak_wl": float(sim_peak_wl),
        "exp_peak_wl": float(exp_peak_wl),
        "sim_peak_height": float(sim_peak_height),
        "exp_peak_height": float(exp_peak_height),
    }

    return float(total_loss), exp_interp_norm, details


def compute_loss(sim_wl, sim_y, exp_wl, exp_y, loss_mode="legacy", loss_config=None):
    if loss_mode == "AuPd_TiO2_v1":
        return aupd_tio2_v1_loss(sim_wl, sim_y, exp_wl, exp_y, loss_config=loss_config)

    loss, exp_interp_norm, details = legacy_loss(sim_wl, sim_y, exp_wl, exp_y)
    details["loss_mode"] = "legacy"
    return loss, exp_interp_norm, details


def make_trial_dir(root, trial_number):
    trial_dir = os.path.join(root, f"trial_{trial_number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir


def save_trial_artifacts(trial_dir, result, exp_interp_norm, params, loss, details, save_plot=True):
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

    if save_plot:
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
    state = {"best_loss": np.inf}

    def objective(trial):
        params = dict(base_params)
        support_type = str(params.get("support_type", "CeO2"))

        params["cu_mean_nm"] = trial.suggest_float(
            "cu_mean_nm",
            float(params.get("opt_cu_mean_min_nm", 15.0)),
            float(params.get("opt_cu_mean_max_nm", 60.0)),
        )
        params["cu_sigma_nm"] = trial.suggest_float(
            "cu_sigma_nm",
            float(params.get("opt_cu_sigma_min_nm", 2.0)),
            float(params.get("opt_cu_sigma_max_nm", 15.0)),
        )

        if support_type == "N-doped TiO2":
            params.pop("ceo2_size_nm", None)
            params.pop("uv_cut_nm", None)
            params.pop("uv_amp", None)
            params.pop("uv_decay_nm", None)
            params.pop("vis_peak_nm", None)
            params.pop("vis_width_nm", None)
            params.pop("vis_k0", None)

            params["support_size_nm"] = trial.suggest_float("support_size_nm", 10.0, 220.0)
            params["tio2_uv_amp"] = trial.suggest_float("tio2_uv_amp", 0.02, 0.20)
            params["tio2_uv_decay_nm"] = trial.suggest_float("tio2_uv_decay_nm", 10.0, 80.0)
            params["tio2_vis_peak_nm"] = trial.suggest_float("tio2_vis_peak_nm", 480.0, 650.0)
            params["tio2_vis_width_nm"] = trial.suggest_float("tio2_vis_width_nm", 60.0, 260.0)
            params["tio2_vis_k0"] = trial.suggest_float("tio2_vis_k0", 1e-4, 2e-2, log=True)
        else:
            params.pop("tio2_uv_amp", None)
            params.pop("tio2_uv_decay_nm", None)
            params.pop("tio2_vis_peak_nm", None)
            params.pop("tio2_vis_width_nm", None)
            params.pop("tio2_vis_k0", None)

            params["ceo2_size_nm"] = trial.suggest_float("ceo2_size_nm", 10.0, 220.0)

        params["number_ratio"] = trial.suggest_float("number_ratio", 0.05, 1.0)

        params["total_dens"] = trial.suggest_float(
            "total_dens", 1e17, 5e18, log=True
        )

        if support_type != "N-doped TiO2":
            params["uv_cut_nm"] = trial.suggest_float("uv_cut_nm", 500.0, 560.0)
            params["uv_amp"] = trial.suggest_float("uv_amp", 0.02, 0.20)
            params["uv_decay_nm"] = trial.suggest_float("uv_decay_nm", 20.0, 80.0)

            params["vis_peak_nm"] = trial.suggest_float("vis_peak_nm", 500.0, 620.0)
            params["vis_width_nm"] = trial.suggest_float("vis_width_nm", 40.0, 180.0)
            params["vis_k0"] = trial.suggest_float("vis_k0", 1e-4, 2e-2, log=True)

        print(f"\n[Trial {trial.number}] START", flush=True)
        print(params, flush=True)

        print("[Trial] running simulation...", flush=True)
        result = run_simulation(params)
        print("[Trial] simulation done.", flush=True)

        loss, exp_interp_norm, details = compute_loss(
            result["lambda_nm"],
            result["F_R_norm"],
            exp_wl,
            exp_fr,
            loss_mode=params.get("loss_mode", "legacy"),
            loss_config=params.get("loss_config"),
        )

        print(f"[Trial {trial.number}] total loss = {loss:.6f}", flush=True)
        if details.get("loss_mode") == "AuPd_TiO2_v1":
            print(
                f"[Trial {trial.number}] "
                f"shape={details['shape_loss']:.6f}, "
                f"peak_pos={details['peak_pos_loss']:.6f}, "
                f"peak_height={details['peak_height_loss']:.6f}, "
                f"tail={details['tail_loss']:.6f}, "
                f"derivative={details['deriv_loss']:.6f}",
                flush=True
            )
            print(
                f"[Trial {trial.number}] "
                f"sim_peak={details['sim_peak_wl']:.2f} nm @ {details['sim_peak_height']:.4f}, "
                f"exp_peak={details['exp_peak_wl']:.2f} nm @ {details['exp_peak_height']:.4f}",
                flush=True
            )
        else:
            print(
                f"[Trial {trial.number}] "
                f"shape={details['shape_mse']:.6f}, "
                f"peak={details['peak_loss']:.6f}, "
                f"fwhm={details['fwhm_loss']:.6f}, "
                f"valley_depth={details['valley_depth_loss']:.6f}, "
                f"valley_pos={details['valley_pos_loss']:.6f}",
                flush=True
            )
            print(
                f"[Trial {trial.number}] "
                f"sim_peak={details['sim_peak_wl']:.2f} nm, "
                f"exp_peak={details['exp_peak_wl']:.2f} nm, "
                f"sim_fwhm={details['sim_fwhm']:.2f} nm, "
                f"exp_fwhm={details['exp_fwhm']:.2f} nm",
                flush=True
            )

            if details["sim_valley_wl"] is not None and details["exp_valley_wl"] is not None:
                print(
                    f"[Trial {trial.number}] "
                    f"sim_valley={details['sim_valley_wl']:.2f} nm @ {details['sim_valley_y']:.4f}, "
                    f"exp_valley={details['exp_valley_wl']:.2f} nm @ {details['exp_valley_y']:.4f}",
                    flush=True
                )

        trial_dir = make_trial_dir(log_root, trial.number)
        save_plot = bool(params.get("save_all_trial_plots", False)) or loss <= state["best_loss"]

        print("[Trial] saving artifacts...", flush=True)
        save_trial_artifacts(
            trial_dir,
            result,
            exp_interp_norm,
            params,
            loss,
            details,
            save_plot=save_plot,
        )
        if loss <= state["best_loss"]:
            state["best_loss"] = float(loss)
        print("[Trial] saved.", flush=True)

        return loss

    return objective
