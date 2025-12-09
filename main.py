import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# EMG imports:
from backbone import Subject
from timesrsutils import TimeSrsTools, SUBJECT_EVENTS

# PPG imports:
from pyPPG import Fiducials
import pyPPG.ppg_sqi as SQI
from pyPPG.datahandling import plot_fiducials, save_data
from PPG_section.PPG_handle import PPGProcTools

# PPG Configuration (Specific to Recorder)
# 140 seconds = 28057 datapoints
RECORDER_PPG_FS = int(28057 / 140)
PPG_TIME_ARR = np.arange(0, 28057) / RECORDER_PPG_FS
DO_PPG_PLOT = False  # disable live plotting for batch PPG processing
PPG_SEGMENT = (0, 15)  # segment in seconds to visualize

# Output directory for PPG
PPG_BASE_DIR = os.path.join("temp_dir", "PPG_AllSubjects")
os.makedirs(PPG_BASE_DIR, exist_ok=True)

# EMG Configuration
FS_EMG = 500
FS_PPG_TARGET = float(RECORDER_PPG_FS)
TARGET_DURATION_S = 140
EMG_LEN = 70000
PPG_LEN = int(FS_PPG_TARGET * TARGET_DURATION_S)


# PPG pipeline calls:
def process_ppg_subject(index, signal):
    """
    Encapsulates the single subject analysis logic from post_pipeline_calls.py
    """
    subject_dir = os.path.join(PPG_BASE_DIR, signal.name)
    os.makedirs(subject_dir, exist_ok=True)
    manifest_path = os.path.join(subject_dir, "manifest.json")

    # Define Naming Convention for plots
    title_root = PPGProcTools.naming_convention(signal.name, PPG_SEGMENT)

    s, fp, bm = None, None, None

    # --- PHASE 1: LOAD OR CALCULATE ---
    if os.path.exists(manifest_path):
        print(f"\n--- Subject {index} found in cache. Loading... ---")
        s, fp, bm = PPGProcTools.load_subject_data(index, PPG_BASE_DIR)
        print(f"Loaded successfully from {manifest_path}")

    if s is None:
        print(f"\n--- Processing PPG Subject {index} ---")

        # 1. Preprocess
        signal = PPGProcTools.preprocess_signal(signal)

        # 2. Fiducials
        # returns 's' (PPG Object), 'fp' (Fiducials Object), 'fid' (dict)
        s, fp, fid = PPGProcTools.compute_fiducials(signal)

        # 3. SQI Check (Optional print)
        ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
        print(f"Subject {index} - Mean PPG SQI = {ppgSQI}%")

        # 4. Biomarkers
        bm_defs, bm_vals, bm_stats, bm = PPGProcTools.compute_biomarkers(s, fp)

        # 5. Save Results
        fp_new = Fiducials(fp.get_fp() + s.start_sig)
        save_data(s=s, fp=fp_new, bm=bm, savingformat='csv', savingfolder=subject_dir)

    # PLOTTING
    # 1. Raw Segment Plot
    PPGProcTools.plot_segment_for_signal(
        PPG_TIME_ARR, s.v, title_root + "_raw", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )

    # 2. Filtered Signal Variants (PPG, VPG, APG, JPG)
    PPGProcTools.plot_processed_signal_variants(
        PPG_TIME_ARR, s, title_root + "_filtered", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )

    # 3. Fiducials Plot
    try:
        canvas = plot_fiducials(s, fp, savefig=False, savingfolder=subject_dir, legend_fontsize=12, show_fig=False)
        PPGProcTools.fiducials_segment_plot(canvas, title_root + "_fiducials", xlim=PPG_SEGMENT,
                                            save_dir=subject_dir + os.sep)
    except Exception as e:
        print(f"Warning: Could not plot fiducials for {index}: {e}")

    print(f"Finished PPG Subject {index}")
    return s, fp, bm


if __name__ == '__main__':

    # PART 1: EMG Processing
    print("\n=== STARTING EMG PROCESSING ===")
    ids = ['P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    # moved subject events to timersutils as a global constant it seemed fitting. More spacious code.
    subject_events = SUBJECT_EVENTS

    # PART 1b: lift to rest segmentation
    subjects_emg: dict[str, Subject] = {}
    for subject_id in ids:
        # Note: data folder should exist
        path = os.path.join("local", "data", f"BSL-EMG-Book{subject_id}.csv")
        subject_event = subject_events[subject_id]
        # where padding
        subj = Subject.from_csv(name=subject_id, path=path, fs=FS_EMG, target_len=EMG_LEN)
        subjects_emg[subject_id] = subj

    # check load results
    print(subjects_emg.keys())

    # example with "P2"
    print(subjects_emg["P2"].df.info())
    s = subjects_emg["P2"]  # TODO naming may cause confusion or conflict: s = some ppg signal vs s = subject object
    TimeSrsTools.emg_preprocess_hilbert(s)
    plt.figure(figsize=(10, 4))
    plt.plot(s.df["time"], s.df["EMG_env"], label="Hilbert-envelope")
    plt.title("EMG Hilbert-envelope – " + s.name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.close()

    # PART 2: PPG Processing:
    print("\n=== STARTING PPG PROCESSING ===")

    # Load full dataset
    # Note: change this to ourData.csv location
    ppg_data_path = "./local/data/ourData.csv"
    if not os.path.exists(ppg_data_path):
        raise FileNotFoundError(f"PPG Data file not found at {ppg_data_path}. Can't process ppg.")

    # Load full dataset
    df = pd.read_csv(ppg_data_path)
    df = df.drop(columns=["tstamp"], index=4, errors="ignore")  # drop sub. 4 right away -> indexing will be better

    # Process all subjects
    for idx, sig in enumerate(df.to_numpy()):  # the idx should be aligned with both ppg and emg data
        try:
            key = ids[idx]
            subject = subjects_emg[key]
            events = subject_events[key]
            subject = subject.resample_data(200)
            sig = PPGProcTools.create_dotmaps_for_pyPPG(sig, ids[idx], RECORDER_PPG_FS)

            # Note: process_ppg_subject could return more if needed. Adjust if so
            signal, _, _ = process_ppg_subject(ids[idx], sig)
            # process_ppg_subject(idx + 2, sig)

            # Filter out first 5 seconds
            usable_signal = signal.vpg[signal.fs * 5: len(subject.df) + signal.fs * 5]
            # TODO should this be moved to process ppg block?
            # 5. NEW: Plot VPG vs EMG with Events
            emg_trace = subject.df["Integrated EMG"].values

            ppg_trace = usable_signal
            # Ensure array lengths match
            min_len = min(len(emg_trace), len(ppg_trace), len(PPG_TIME_ARR))
            vpg_emg_segment = (0, PPG_TIME_ARR[min_len])  # Not recommended to change.
            PPGProcTools.plot_vpg_emg(
                x=PPG_TIME_ARR[:min_len],
                signal_1=ppg_trace[:min_len],
                signal_2=emg_trace[:min_len],
                subject_id=key,
                title=PPGProcTools.naming_convention(signal.name, vpg_emg_segment) + "_EMG_and_PPG",
                xlim=vpg_emg_segment,
                save_dir=os.path.join(PPG_BASE_DIR, signal.name) + os.sep,
                do_plot=DO_PPG_PLOT
            )

            # STATISTICS WORKS
            # TODO but not: move these into something like the ppg process
            # 1. loads/sets
            subject.set_column("ppg", usable_signal).chunk_subject_by_events(events)
            # sort by muscle activity
            for active_chunk in subject.active_chunks:
                subject.active_windows.extend(TimeSrsTools.window_dataframe(active_chunk, subject.fs * 2, subject.fs))

            for passive_chunk in subject.passive_chunks:
                subject.passive_windows.extend(TimeSrsTools.window_dataframe(passive_chunk, subject.fs * 2, subject.fs))

            subject.apply_gaussian_windowing(["Integrated EMG"], 0.4)

            active_windows_rms = [np.sqrt(np.mean(window["Integrated EMG"] ** 2)) for window in subject.active_windows]
            active_windows_mav = [np.mean(np.abs(window["Integrated EMG"])) for window in subject.active_windows]
            active_windows_ppg_amplitude = [np.max(window["ppg"]) for window in subject.active_windows]

            passive_windows_rms = [np.sqrt(np.mean(window["Integrated EMG"] ** 2)) for window in
                                   subject.passive_windows]
            passive_windows_mav = [np.mean(np.abs(window["Integrated EMG"])) for window in subject.passive_windows]
            passive_windows_ppg_amplitude = [np.max(window["ppg"]) for window in subject.passive_windows]

            active_data = pd.DataFrame({
                'EMG_RMS': active_windows_rms,
                'EMG_MAV': active_windows_mav,
                'PPG_Amp': active_windows_ppg_amplitude,
                'Phase': 'Lift'
            })

            passive_data = pd.DataFrame({
                'EMG_RMS': passive_windows_rms,
                'EMG_MAV': passive_windows_mav,
                'PPG_Amp': passive_windows_ppg_amplitude,
                'Phase': 'Rest'
            })
            phases_to_process = [("Lift", active_data), ("Rest", passive_data)]
            # TODO unfinished correlation matrices. No negative correlation. We need to normalize?
            for phase_name, df_phase in phases_to_process:
                # 1. Calculate Correlation Matrix
                cols_to_corr = ['EMG_RMS', 'EMG_MAV', 'PPG_Amp']
                corr_matrix = df_phase[cols_to_corr].corr()

                print(f"\nSubject {ids[idx]} [{phase_name}] Correlation Matrix:\n{corr_matrix}")

                # 3. Plot Heatmap
                plt.figure(figsize=(6, 5))

                # Display matrix
                im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

                # Add colorbar
                cbar = plt.colorbar(im)
                cbar.set_label('Pearson Correlation (r)', rotation=270, labelpad=15)

                # Set labels
                labels = ['EMG (RMS)', 'EMG (MAV)', 'PPG (Amp)']
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.yticks(range(len(labels)), labels)

                # Dynamic Title based on Phase
                plt.title(f"Subject {ids[idx]} - {phase_name} Phase: Signal Correlation")

                # Annotate values
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        val = corr_matrix.iloc[i, j]
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        plt.text(j, i, f"{val:.2f}",
                                 ha='center', va='center', color=text_color, fontweight='bold')

                plt.tight_layout()

                save_name = f"{signal.name}_{phase_name}_Corr_Matrix.png"
                save_path = os.path.join(PPG_BASE_DIR, signal.name, save_name)
                plt.savefig(save_path)
                plt.close()

        except Exception as e:
            print(f" Error processing PPG subject {ids[idx]}: {e}")
            continue

    print("\nAll PPG subjects processed.")

    # plot
