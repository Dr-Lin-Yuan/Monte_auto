import json
from datetime import datetime
from pathlib import Path

import optuna

from fit_agent import build_objective, compute_loss, save_trial_artifacts
from sim_core import load_experiment_txt, run_simulation


DEFAULT_DIELECTRIC_FILE = (
    Path("dielectric_permittivity")
    / "dielectric_function_Cu_smoothed_1p55to3p54eV.txt"
)
DEFAULT_SEARCH_PHOTONS = 10000


def default_n_medium_for_support(support_type):
    if str(support_type) == "N-doped TiO2":
        return 2.55
    return 1.75


def default_base_params():
    return {
        "seed": 1,
        "n_particles": 300,
        "cu_lower_nm": 20.0,
        "cu_upper_nm": 40.0,
        "cu_mean_nm": 30.0,
        "cu_sigma_nm": 5.0,
        "file_eps": str(DEFAULT_DIELECTRIC_FILE),
        "n_medium": 1.75,
        "mie_lambda_min_nm": 400.0,
        "mie_lambda_max_nm": 800.0,
        "mie_lambda_step_nm": 1.0,
        "fit_lambda_min_nm": 400.0,
        "fit_lambda_max_nm": 800.0,
        "n_wavelengths": 100,
        "n_photons": 100000,
        "h_sample_m": 0.002,
        "support_type": "CeO2",
        "support_size_nm": 30.0,
        "ceo2_size_nm": 30.0,
        "tio2_uv_amp": 0.08,
        "tio2_uv_decay_nm": 20.0,
        "tio2_vis_peak_nm": 520.0,
        "tio2_vis_width_nm": 150.0,
        "tio2_vis_k0": 5e-3,
        "simulation_mode": "fast_hybrid_gpu",
        "loss_mode": "legacy",
        "number_ratio": 0.3,
        "total_dens": 1e18,
        "uv_cut_nm": 525.0,
        "uv_amp": 0.10,
        "uv_decay_nm": 40.0,
        "vis_peak_nm": 525.0,
        "vis_width_nm": 90.0,
        "vis_k0": 4e-3,
    }


def create_run_directory(root="app_runs"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"run_{stamp}"
    inputs_dir = run_dir / "inputs"
    logs_dir = run_dir / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, inputs_dir, logs_dir


def write_uploaded_bytes(target: Path, content: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def dielectric_options(directory="dielectric_permittivity"):
    root = Path(directory)
    if not root.exists():
        return []
    # *_nk.txt files (lambda, n, k) belong to the Mie toolkit tab; the Monte Carlo loader
    # expects E_eV eps1 eps2, so they are not offered here.
    return sorted(path for path in root.iterdir() if path.is_file() and not path.name.endswith("_nk.txt"))


def build_app_params(
    experiment_file,
    dielectric_file,
    particle_lower_nm,
    particle_upper_nm,
    mean_nm,
    sigma_nm,
    n_particles,
    n_wavelengths,
    n_photons,
    support_type="CeO2",
    support_size_nm=30.0,
    simulation_mode="fast_hybrid_gpu",
    n_medium=None,
    tio2_uv_amp=0.08,
    tio2_uv_decay_nm=20.0,
    tio2_vis_peak_nm=520.0,
    tio2_vis_width_nm=150.0,
    tio2_vis_k0=5e-3,
    allow_cpu_fallback=True,
    use_mie_lut=True,
    mie_diameter_grid_size=160,
):
    params = default_base_params()
    resolved_n_medium = default_n_medium_for_support(support_type) if n_medium is None else float(n_medium)
    params.update(
        {
            "cu_lower_nm": float(particle_lower_nm),
            "cu_upper_nm": float(particle_upper_nm),
            "cu_mean_nm": float(mean_nm),
            "cu_sigma_nm": float(sigma_nm),
            "n_particles": int(n_particles),
            "n_wavelengths": int(n_wavelengths),
            "n_photons": int(n_photons),
            "file_eps": str(Path(dielectric_file)),
            "n_medium": resolved_n_medium,
            "support_type": str(support_type),
            "support_size_nm": float(support_size_nm),
            "ceo2_size_nm": float(support_size_nm),
            "tio2_uv_amp": float(tio2_uv_amp),
            "tio2_uv_decay_nm": float(tio2_uv_decay_nm),
            "tio2_vis_peak_nm": float(tio2_vis_peak_nm),
            "tio2_vis_width_nm": float(tio2_vis_width_nm),
            "tio2_vis_k0": float(tio2_vis_k0),
            "simulation_mode": str(simulation_mode),
            "allow_cpu_fallback": bool(allow_cpu_fallback),
            "use_mie_lut": bool(use_mie_lut),
            "mie_diameter_grid_size": int(mie_diameter_grid_size),
            "opt_cu_mean_min_nm": float(particle_lower_nm),
            "opt_cu_mean_max_nm": float(particle_upper_nm),
            "opt_cu_sigma_min_nm": max(0.5, float(sigma_nm) * 0.5),
            "opt_cu_sigma_max_nm": max(float(sigma_nm) + 1.0, float(sigma_nm) * 1.5),
        }
    )
    params["experiment_file"] = str(Path(experiment_file))
    return params


def run_fit_pipeline(
    experiment_file,
    dielectric_file,
    output_root,
    n_trials,
    particle_lower_nm,
    particle_upper_nm,
    mean_nm,
    sigma_nm,
    n_particles,
    n_wavelengths,
    n_photons,
    support_type="CeO2",
    support_size_nm=30.0,
    simulation_mode="fast_hybrid_gpu",
    n_medium=None,
    tio2_uv_amp=0.08,
    tio2_uv_decay_nm=20.0,
    tio2_vis_peak_nm=520.0,
    tio2_vis_width_nm=150.0,
    tio2_vis_k0=5e-3,
    loss_mode="legacy",
    n_search_photons=None,
    refine_best_full_precision=True,
    allow_cpu_fallback=True,
    use_mie_lut=True,
    mie_diameter_grid_size=160,
    save_all_trial_plots=False,
    cancel_event=None,
    progress_callback=None,
    study_name="plasmonic_streamlit_fit",
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    full_n_photons = int(n_photons)
    if n_search_photons is None:
        search_n_photons = min(full_n_photons, DEFAULT_SEARCH_PHOTONS)
    else:
        search_n_photons = min(full_n_photons, int(n_search_photons))
    search_n_photons = max(1, search_n_photons)

    exp_wl, exp_fr = load_experiment_txt(experiment_file)
    base_params = build_app_params(
        experiment_file=experiment_file,
        dielectric_file=dielectric_file,
        particle_lower_nm=particle_lower_nm,
        particle_upper_nm=particle_upper_nm,
        mean_nm=mean_nm,
        sigma_nm=sigma_nm,
        n_particles=n_particles,
        n_wavelengths=n_wavelengths,
        n_photons=search_n_photons,
        support_type=support_type,
        support_size_nm=support_size_nm,
        simulation_mode=simulation_mode,
        n_medium=n_medium,
        tio2_uv_amp=tio2_uv_amp,
        tio2_uv_decay_nm=tio2_uv_decay_nm,
        tio2_vis_peak_nm=tio2_vis_peak_nm,
        tio2_vis_width_nm=tio2_vis_width_nm,
        tio2_vis_k0=tio2_vis_k0,
        allow_cpu_fallback=allow_cpu_fallback,
        use_mie_lut=use_mie_lut,
        mie_diameter_grid_size=mie_diameter_grid_size,
    )
    base_params["loss_mode"] = str(loss_mode)
    base_params["n_photons_search"] = search_n_photons
    base_params["n_photons_full"] = full_n_photons
    base_params["save_all_trial_plots"] = bool(save_all_trial_plots)

    storage_url = f"sqlite:///{output_root / 'optuna_study.db'}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "cu_mean_nm": float(mean_nm),
                "cu_sigma_nm": float(sigma_nm),
            }
        )

    objective = build_objective(base_params, exp_wl, exp_fr, str(log_root))
    total_trials = int(n_trials)

    optuna_callbacks = []
    if progress_callback is not None:
        def on_trial_complete(study, trial):
            completed_trials = len(study.trials)
            best_value = study.best_value if study.best_trial is not None else None
            progress_callback(
                stage="running optimization",
                completed_trials=completed_trials,
                total_trials=total_trials,
                trial_number=trial.number,
                best_value=best_value,
            )
            if cancel_event is not None and cancel_event.is_set():
                progress_callback(stage="cancelled")
                study.stop()

        optuna_callbacks.append(on_trial_complete)
    elif cancel_event is not None:
        def on_trial_complete(study, trial):
            if cancel_event.is_set():
                study.stop()

        optuna_callbacks.append(on_trial_complete)

    study.optimize(objective, n_trials=total_trials, callbacks=optuna_callbacks)

    best_trial_dir = log_root / f"trial_{study.best_trial.number:04d}"
    best_value = study.best_value
    best_params = study.best_params

    if (
        refine_best_full_precision
        and full_n_photons > search_n_photons
        and (cancel_event is None or not cancel_event.is_set())
    ):
        if progress_callback is not None:
            progress_callback(stage="refining best result")
        refined_params = dict(base_params)
        refined_params.update(study.best_params)
        refined_params["n_photons"] = full_n_photons
        refined_params["refined_from_trial"] = int(study.best_trial.number)
        refined_result = run_simulation(refined_params)
        refined_loss, refined_exp_interp_norm, refined_details = compute_loss(
            refined_result["lambda_nm"],
            refined_result["F_R_norm"],
            exp_wl,
            exp_fr,
            loss_mode=refined_params.get("loss_mode", "legacy"),
            loss_config=refined_params.get("loss_config"),
        )
        best_trial_dir = log_root / "best_full_precision"
        best_trial_dir.mkdir(parents=True, exist_ok=True)
        save_trial_artifacts(
            str(best_trial_dir),
            refined_result,
            refined_exp_interp_norm,
            refined_params,
            refined_loss,
            refined_details,
            save_plot=True,
        )
        best_value = float(refined_loss)
        best_params = refined_params

    best_meta_file = best_trial_dir / "meta.json"
    best_meta = {}
    if best_meta_file.exists():
        best_meta = json.loads(best_meta_file.read_text(encoding="utf-8"))

    return {
        "study": study,
        "output_root": output_root,
        "log_root": log_root,
        "best_trial_dir": best_trial_dir,
        "best_value": best_value,
        "search_best_value": study.best_value,
        "best_params": best_params,
        "best_meta": best_meta,
        "compare_png": best_trial_dir / "compare.png",
        "meta_json": best_trial_dir / "meta.json",
        "sim_spectrum_txt": best_trial_dir / "sim_spectrum.txt",
        "diameters_txt": best_trial_dir / "diameters_nm.txt",
        "optuna_storage": output_root / "optuna_study.db",
        "n_photons_search": search_n_photons,
        "n_photons_full": full_n_photons,
    }
