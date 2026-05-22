from pathlib import Path
import queue
import threading
import time

import streamlit as st

from fit_pipeline import (
    create_run_directory,
    DEFAULT_DIELECTRIC_FILE,
    default_n_medium_for_support,
    dielectric_options,
    run_fit_pipeline,
    write_uploaded_bytes,
)


st.set_page_config(page_title="Plasmonic Fit MVP", layout="wide")


def save_inputs(experiment_upload, dielectric_upload, selected_dielectric):
    run_dir, inputs_dir, _ = create_run_directory()

    experiment_path = write_uploaded_bytes(
        inputs_dir / experiment_upload.name,
        experiment_upload.getvalue(),
    )

    dielectric_source = "uploaded"
    dielectric_path = None

    if dielectric_upload is not None:
        dielectric_path = write_uploaded_bytes(
            inputs_dir / dielectric_upload.name,
            dielectric_upload.getvalue(),
        )
    elif selected_dielectric:
        dielectric_source = "existing"
        dielectric_path = Path(selected_dielectric)

    return run_dir, experiment_path, dielectric_path, dielectric_source


def file_rows(paths):
    rows = []
    for label, path in paths:
        if path and Path(path).exists():
            rows.append({"label": label, "path": str(Path(path).resolve())})
    return rows


def start_fit_job(job_kwargs, dielectric_source):
    messages = queue.Queue()
    cancel_event = threading.Event()

    def progress_callback(**event):
        messages.put(("progress", event))

    def worker():
        try:
            result = run_fit_pipeline(
                **job_kwargs,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )
            messages.put(("done", result))
        except Exception as exc:
            messages.put(("error", repr(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    st.session_state.fit_job = {
        "thread": thread,
        "messages": messages,
        "cancel_event": cancel_event,
        "stage": "queued",
        "completed_trials": 0,
        "total_trials": int(job_kwargs["n_trials"]),
        "trial_number": None,
        "best_value": None,
        "result": None,
        "error": None,
        "dielectric_source": dielectric_source,
    }
    thread.start()


def drain_fit_job():
    job = st.session_state.get("fit_job")
    if not job:
        return None

    while True:
        try:
            kind, payload = job["messages"].get_nowait()
        except queue.Empty:
            break

        if kind == "progress":
            job.update(payload)
        elif kind == "done":
            job["stage"] = "done"
            job["result"] = payload
        elif kind == "error":
            job["stage"] = "error"
            job["error"] = payload

    return job


def render_result(result, dielectric_source):
    st.success("Fit finished.")

    st.subheader("Run Summary")
    st.write(f"Run folder: `{Path(result['output_root']).resolve()}`")
    st.write(f"Best trial folder: `{Path(result['best_trial_dir']).resolve()}`")
    st.write(f"Optuna storage: `{Path(result['optuna_storage']).resolve()}`")
    st.write(f"Dielectric source: `{dielectric_source}`")
    st.write(f"Search photons: `{result['n_photons_search']}`")
    st.write(f"Full photons: `{result['n_photons_full']}`")
    st.write(f"Best loss: `{result['best_value']:.6f}`")

    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader("Best-Fit Plot")
        if Path(result["compare_png"]).exists():
            st.image(str(Path(result["compare_png"]).resolve()))
        else:
            st.warning("No comparison plot was found for the best trial.")

    with right:
        st.subheader("Best-Fit Parameters")
        best_params = result.get("best_meta", {}).get("params") or result.get("best_params", {})
        st.json(best_params)

    st.subheader("Output Files")
    st.dataframe(
        file_rows(
            [
                ("Comparison plot", result["compare_png"]),
                ("Trial metadata", result["meta_json"]),
                ("Simulated spectrum", result["sim_spectrum_txt"]),
                ("Diameter samples", result["diameters_txt"]),
                ("Optuna SQLite DB", result["optuna_storage"]),
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if Path(result["meta_json"]).exists():
        st.subheader("Best Trial Metadata")
        st.json(result.get("best_meta", {}))


def render_fit_job(job):
    stage = job.get("stage", "queued")
    completed = int(job.get("completed_trials") or 0)
    total = max(int(job.get("total_trials") or 1), 1)
    best_value = job.get("best_value")

    if stage == "done" and job.get("result") is not None:
        render_result(job["result"], job.get("dielectric_source", "unknown"))
        if st.button("Clear finished run"):
            st.session_state.pop("fit_job", None)
            st.rerun()
        return

    if stage == "error":
        st.error(f"Fit failed: {job.get('error')}")
        if st.button("Clear failed run"):
            st.session_state.pop("fit_job", None)
            st.rerun()
        return

    progress = min(95, max(5, int(10 + 80 * completed / total)))
    if stage == "refining best result":
        progress = 97
    elif stage == "cancelled":
        progress = 100
    st.progress(progress)

    message = f"{stage} | trial {completed}/{total}"
    if best_value is not None:
        message += f" | best loss {best_value:.6f}"
    st.info(message)

    if not job["cancel_event"].is_set():
        if st.button("Cancel after current trial"):
            job["cancel_event"].set()
            job["stage"] = "cancel requested"
            st.rerun()
    else:
        st.warning("Cancel requested. The current simulation will finish before Optuna stops.")

    if job["thread"].is_alive():
        time.sleep(1)
        st.rerun()


st.title("Plasmonic Spectrum Fitting MVP")
st.write("Minimal local Streamlit wrapper around the existing fitting pipeline.")

available_dielectrics = dielectric_options()
default_dielectric = str(DEFAULT_DIELECTRIC_FILE) if DEFAULT_DIELECTRIC_FILE.exists() else None
default_index = 0
if default_dielectric and default_dielectric in [str(path) for path in available_dielectrics]:
    default_index = [str(path) for path in available_dielectrics].index(default_dielectric)

with st.sidebar:
    st.header("Inputs")
    experiment_upload = st.file_uploader(
        "Experimental spectrum",
        type=["txt", "csv", "dat"],
        help="Expected format: two columns with wavelength and intensity.",
    )
    dielectric_upload = st.file_uploader(
        "Dielectric function file (optional upload)",
        type=["txt", "csv", "dat"],
        help="If you do not upload a file, the selected local file below will be used.",
    )

    dielectric_choices = [str(path) for path in available_dielectrics]
    if dielectric_choices:
        selected_dielectric = st.selectbox(
            "Or select an existing dielectric file",
            dielectric_choices,
            index=default_index,
        )
    else:
        selected_dielectric = ""
        st.caption("No local dielectric files were found in `dielectric_permittivity/`.")

    st.subheader("Particle settings")
    particle_lower_nm = st.number_input("Particle lower diameter bound (nm)", min_value=0.1, value=20.0)
    particle_upper_nm = st.number_input("Particle upper diameter bound (nm)", min_value=0.1, value=40.0)
    mean_nm = st.number_input("Initial / suggested mean diameter (nm)", min_value=0.1, value=30.0)
    sigma_nm = st.number_input("Initial / suggested sigma (nm)", min_value=0.1, value=5.0)
    support_type = st.selectbox(
        "Support material",
        ["CeO2", "N-doped TiO2"],
        index=0,
    )
    n_medium_state_key = "n_medium_value"
    support_state_key = "support_type_last"
    support_default_n_medium = default_n_medium_for_support(support_type)
    if support_state_key not in st.session_state:
        st.session_state[support_state_key] = support_type
    if n_medium_state_key not in st.session_state:
        st.session_state[n_medium_state_key] = support_default_n_medium
    elif st.session_state[support_state_key] != support_type:
        st.session_state[n_medium_state_key] = support_default_n_medium
        st.session_state[support_state_key] = support_type
    n_medium = st.number_input(
        "Effective medium refractive index",
        min_value=1.0,
        value=float(st.session_state[n_medium_state_key]),
        step=0.05,
    )
    st.session_state[n_medium_state_key] = float(n_medium)
    support_size_nm = st.number_input("Support particle diameter (nm)", min_value=0.1, value=30.0)
    tio2_uv_amp = 0.08
    tio2_uv_decay_nm = 20.0
    tio2_vis_peak_nm = 520.0
    tio2_vis_width_nm = 150.0
    tio2_vis_k0 = 0.005
    if support_type == "N-doped TiO2":
        st.subheader("TiO2 Support Controls")
        tio2_uv_amp = st.number_input("TiO2 UV amplitude", min_value=0.0, value=tio2_uv_amp, step=0.01)
        tio2_uv_decay_nm = st.number_input("TiO2 UV decay (nm)", min_value=0.1, value=tio2_uv_decay_nm, step=1.0)
        tio2_vis_peak_nm = st.number_input("TiO2 visible tail peak (nm)", min_value=0.1, value=tio2_vis_peak_nm, step=1.0)
        tio2_vis_width_nm = st.number_input("TiO2 visible tail width (nm)", min_value=0.1, value=tio2_vis_width_nm, step=1.0)
        tio2_vis_k0 = st.number_input("TiO2 visible tail k0", min_value=0.0, value=tio2_vis_k0, step=0.001, format="%.6f")
    simulation_mode = st.selectbox(
        "Simulation mode",
        ["fast_hybrid_gpu", "legacy_exact_powder"],
        index=0,
    )
    loss_mode = st.selectbox(
        "Loss mode",
        ["legacy", "AuPd_TiO2_v1"],
        index=0,
    )
    n_particles = st.number_input("Number of particles", min_value=1, value=300, step=1)
    n_wavelengths = st.number_input("Number of wavelengths", min_value=2, value=100, step=1)
    n_photons = st.number_input("Number of photons", min_value=1, value=100000, step=1000)
    n_search_photons = st.number_input(
        "Search photons",
        min_value=1,
        max_value=int(n_photons),
        value=min(10000, int(n_photons)),
        step=1000,
    )
    n_trials = st.number_input("Number of Optuna trials", min_value=1, value=50, step=1)
    refine_best_full_precision = st.checkbox("Refine best with full photons", value=True)
    allow_cpu_fallback = st.checkbox("Fallback to CPU if CUDA is unavailable", value=True)
    use_mie_lut = st.checkbox("Use Cu Mie LUT", value=True)
    mie_diameter_grid_size = st.number_input("Mie diameter grid size", min_value=8, value=160, step=8)
    save_all_trial_plots = st.checkbox("Save every trial plot", value=False)

    run_clicked = st.button("Run Fit", type="primary", use_container_width=True)

job = drain_fit_job()
if job is not None:
    render_fit_job(job)
elif run_clicked:
    if experiment_upload is None:
        st.error("Upload an experimental spectrum file before running the fit.")
        st.stop()

    if dielectric_upload is None and not selected_dielectric:
        st.error("Upload a dielectric function file or select an existing one.")
        st.stop()

    if particle_lower_nm > particle_upper_nm:
        st.error("Particle lower diameter bound must be less than or equal to the upper bound.")
        st.stop()

    run_dir, experiment_path, dielectric_path, dielectric_source = save_inputs(
        experiment_upload,
        dielectric_upload,
        selected_dielectric,
    )
    start_fit_job(
        {
            "experiment_file": experiment_path,
            "dielectric_file": dielectric_path,
            "output_root": run_dir,
            "n_trials": n_trials,
            "particle_lower_nm": particle_lower_nm,
            "particle_upper_nm": particle_upper_nm,
            "mean_nm": mean_nm,
            "sigma_nm": sigma_nm,
            "n_particles": n_particles,
            "n_wavelengths": n_wavelengths,
            "n_photons": n_photons,
            "n_search_photons": n_search_photons,
            "support_type": support_type,
            "support_size_nm": support_size_nm,
            "simulation_mode": simulation_mode,
            "n_medium": n_medium,
            "tio2_uv_amp": tio2_uv_amp,
            "tio2_uv_decay_nm": tio2_uv_decay_nm,
            "tio2_vis_peak_nm": tio2_vis_peak_nm,
            "tio2_vis_width_nm": tio2_vis_width_nm,
            "tio2_vis_k0": tio2_vis_k0,
            "loss_mode": loss_mode,
            "refine_best_full_precision": refine_best_full_precision,
            "allow_cpu_fallback": allow_cpu_fallback,
            "use_mie_lut": use_mie_lut,
            "mie_diameter_grid_size": mie_diameter_grid_size,
            "save_all_trial_plots": save_all_trial_plots,
        },
        dielectric_source,
    )
    st.rerun()
else:
    st.info("Upload inputs, set the basic fitting controls, and click Run Fit.")
