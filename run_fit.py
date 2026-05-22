import os
import optuna

from sim_core import load_experiment_txt
from fit_agent import build_objective


def main():
    exp_wl, exp_fr = load_experiment_txt("Cu_exp.txt")

    base_params = {
        "seed": 1,
        "n_particles": 300,
        "cu_lower_nm": 20.0,
        "cu_upper_nm": 40.0,
        "cu_mean_nm": 30.0,
        "cu_sigma_nm": 5.0,
        "file_eps": "./dielectric_permittivity/dielectric_function_Cu_smoothed_1p55to3p54eV.txt",
        "n_medium": 1.75,
        "mie_lambda_min_nm": 350.0,
        "mie_lambda_max_nm": 800.0,
        "mie_lambda_step_nm": 1.0,
        "fit_lambda_min_nm": 500.0,
        "fit_lambda_max_nm": 700.0,
        "n_wavelengths": 100,
        "n_photons": 100000,   # 先降一点，先跑通
        "h_sample_m": 0.002,
        "ceo2_size_nm": 30.0,
        "number_ratio": 0.3,
        "total_dens": 1e18,
        "uv_cut_nm": 525.0,
        "uv_amp": 0.10,
        "uv_decay_nm": 40.0,
        "vis_peak_nm": 525.0,
        "vis_width_nm": 90.0,
        "vis_k0": 4e-3,
    }

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name="cu_ceo2_fit"
    )

    objective = build_objective(base_params, exp_wl, exp_fr, log_root)

    study.optimize(objective, n_trials=50)

    print("\nBest loss:")
    print(study.best_value)
    print("\nBest params:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()