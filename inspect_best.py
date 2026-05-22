import json
import os
from pathlib import Path


LOG_ROOT = Path("logs")
TOP_K = 10


def load_trial_meta(trial_dir: Path):
    meta_file = trial_dir / "meta.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        print(f"[Warning] Failed to read {meta_file}: {e}")
        return None


def collect_trials(log_root: Path):
    trials = []

    if not log_root.exists():
        print(f"[Error] Log directory not found: {log_root.resolve()}")
        return trials

    for item in sorted(log_root.iterdir()):
        if not item.is_dir():
            continue
        if not item.name.startswith("trial_"):
            continue

        meta = load_trial_meta(item)
        if meta is None:
            continue

        loss = meta.get("loss", None)
        if loss is None:
            continue

        trial_number = None
        try:
            trial_number = int(item.name.split("_")[-1])
        except Exception:
            trial_number = item.name

        trials.append(
            {
                "trial_dir": item,
                "trial_number": trial_number,
                "loss": float(loss),
                "meta": meta,
            }
        )

    return trials


def print_trial_summary(trial_info, rank=None):
    prefix = f"#{rank} " if rank is not None else ""
    print(f"{prefix}Trial {trial_info['trial_number']}")
    print(f"  Loss: {trial_info['loss']:.6f}")

    loss_details = trial_info["meta"].get("loss_details", {})
    if loss_details:
        if "shape_mse" in loss_details:
            print(f"  Shape MSE : {loss_details['shape_mse']:.6f}")
        if "peak_loss" in loss_details:
            print(f"  Peak loss : {loss_details['peak_loss']:.6f}")
        if "fwhm_loss" in loss_details:
            print(f"  FWHM loss : {loss_details['fwhm_loss']:.6f}")

        sim_peak = loss_details.get("sim_peak_wl", None)
        exp_peak = loss_details.get("exp_peak_wl", None)
        sim_fwhm = loss_details.get("sim_fwhm", None)
        exp_fwhm = loss_details.get("exp_fwhm", None)

        if sim_peak is not None and exp_peak is not None:
            print(f"  Peak      : sim {sim_peak:.2f} nm | exp {exp_peak:.2f} nm")
        if sim_fwhm is not None and exp_fwhm is not None:
            print(f"  FWHM      : sim {sim_fwhm:.2f} nm | exp {exp_fwhm:.2f} nm")

    print(f"  Folder    : {trial_info['trial_dir']}")
    print()


def print_best_params(best_trial):
    params = best_trial["meta"].get("params", {})
    print("Best params:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()


def print_best_files(best_trial):
    trial_dir = best_trial["trial_dir"]

    compare_png = trial_dir / "compare.png"
    meta_json = trial_dir / "meta.json"
    sim_txt = trial_dir / "sim_spectrum.txt"
    diam_txt = trial_dir / "diameters_nm.txt"

    print("Best trial files:")
    print(f"  compare.png      : {compare_png if compare_png.exists() else 'Not found'}")
    print(f"  meta.json        : {meta_json if meta_json.exists() else 'Not found'}")
    print(f"  sim_spectrum.txt : {sim_txt if sim_txt.exists() else 'Not found'}")
    print(f"  diameters_nm.txt : {diam_txt if diam_txt.exists() else 'Not found'}")
    print()


def main():
    trials = collect_trials(LOG_ROOT)

    if not trials:
        print("No valid trial logs found.")
        return

    trials_sorted = sorted(trials, key=lambda x: x["loss"])
    best_trial = trials_sorted[0]

    print("=" * 72)
    print("BEST TRIAL")
    print("=" * 72)
    print_trial_summary(best_trial)
    print_best_params(best_trial)
    print_best_files(best_trial)

    print("=" * 72)
    print(f"TOP {min(TOP_K, len(trials_sorted))} TRIALS")
    print("=" * 72)
    for i, t in enumerate(trials_sorted[:TOP_K], start=1):
        print_trial_summary(t, rank=i)


if __name__ == "__main__":
    main()