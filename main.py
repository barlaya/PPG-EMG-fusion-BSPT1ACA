import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EMG imports:
from backbone import Subject
from timesrsutils import TimeSrsTools

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
PPG_SEGMENT = (0, 140)  # segment to visualize

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
    # ___SINGLE_SUBJECT_ANALYSIS___
    subject_dir = os.path.join(PPG_BASE_DIR, f"Subject_{index}")
    os.makedirs(subject_dir, exist_ok=True)

    print(f"\n--- Processing PPG Subject {index} ---")

    # Naming
    title_root = PPGProcTools.naming_convention(signal.name, PPG_SEGMENT)

    # 1. Raw segment plot
    PPGProcTools.plot_segment_for_signal(
        PPG_TIME_ARR, signal.v, title_root + "_raw", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )

    # 2. Preprocess
    signal = PPGProcTools.preprocess_signal(signal)

    # 3. Filtered plots
    PPGProcTools.plot_processed_signal_variants(
        PPG_TIME_ARR, signal, title_root + "_filtered", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )

    # 4. Fiducials
    s, fp, fid = PPGProcTools.compute_fiducials(signal)
    canvas = plot_fiducials(s, fp, savefig=False, savingfolder=subject_dir, legend_fontsize=12, show_fig=False)
    PPGProcTools.fiducials_segment_plot(canvas, title_root + "_fiducials", xlim=PPG_SEGMENT,
                                        save_dir=subject_dir + os.sep)

    # 5. SQI
    ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    print(f"Subject {index} - Mean PPG SQI = {ppgSQI}%")

    # 6. Biomarkers
    bm_defs, bm_vals, bm_stats, bm = PPGProcTools.compute_biomarkers(s, fp)

    # 7. Save results
    fp_new = Fiducials(fp.get_fp() + s.start_sig)
    save_data(s=s, fp=fp_new, bm=bm, savingformat='csv', savingfolder=subject_dir)

    print(f"Finished PPG Subject {index}")

if __name__ == '__main__':

    # PART 1: EMG Processing
    print("\n=== STARTING EMG PROCESSING ===")

    ids = ['P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    subject_events = {
        "P2": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.2640},
            {"phase": "lift", "start_s": 4.2640, "end_s": 17.9740},
            {"phase": "rest", "start_s": 17.9740, "end_s": 28.3400},
            {"phase": "lift", "start_s": 28.3400, "end_s": 42.1960},
            {"phase": "rest", "start_s": 42.1960, "end_s": 54.3080},
            {"phase": "lift", "start_s": 54.3080, "end_s": 68.5500},
        ],
        "P3": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.3520},
            {"phase": "lift", "start_s": 4.3520, "end_s": 17.9660},
            {"phase": "rest", "start_s": 17.9660, "end_s": 26.3900},
            {"phase": "lift", "start_s": 26.3900, "end_s": 40.6820},
            {"phase": "rest", "start_s": 40.6820, "end_s": 48.1880},
            {"phase": "lift", "start_s": 48.1880, "end_s": 62.6000},
        ],
        "P4": [
            {"phase": "rest", "start_s": 0.0, "end_s": 5.0020},
            {"phase": "lift", "start_s": 5.0020, "end_s": 20.8540},
            {"phase": "rest", "start_s": 20.8540, "end_s": 26.4480},
            {"phase": "lift", "start_s": 26.4480, "end_s": 43.6560},
            {"phase": "rest", "start_s": 43.6560, "end_s": 50.6500},
            {"phase": "lift", "start_s": 50.6500, "end_s": 66.7580},
        ],
        "P5": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.5480},
            {"phase": "lift", "start_s": 4.5480, "end_s": 20.0400},
            {"phase": "rest", "start_s": 20.0400, "end_s": 28.4940},
            {"phase": "lift", "start_s": 28.4940, "end_s": 45.4440},
            {"phase": "rest", "start_s": 45.4440, "end_s": 52.5660},
            {"phase": "lift", "start_s": 52.5660, "end_s": 69.3460},
        ],
        "P6": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.6600},
            {"phase": "lift", "start_s": 4.6600, "end_s": 18.6860},
            {"phase": "rest", "start_s": 18.6860, "end_s": 27.7940},
            {"phase": "lift", "start_s": 27.7940, "end_s": 42.8460},
            {"phase": "rest", "start_s": 42.8460, "end_s": 51.2260},
            {"phase": "lift", "start_s": 51.2260, "end_s": 67.7320},
        ],
        "P7": [
            {"phase": "rest", "start_s": 0.0, "end_s": 2.7880},
            {"phase": "lift", "start_s": 2.7880, "end_s": 20.0240},
            {"phase": "rest", "start_s": 20.0240, "end_s": 28.9380},
            {"phase": "lift", "start_s": 28.9380, "end_s": 45.5820},
            {"phase": "rest", "start_s": 45.5820, "end_s": 53.2720},
            {"phase": "lift", "start_s": 53.2720, "end_s": 67.0000},
        ],
        "P8": [
            {"phase": "rest", "start_s": 0.0, "end_s": 12.7400},
            {"phase": "lift", "start_s": 12.7400, "end_s": 28.1120},
            {"phase": "rest", "start_s": 28.1120, "end_s": 37.1860},
            {"phase": "lift", "start_s": 37.1860, "end_s": 51.2420},
            {"phase": "rest", "start_s": 51.2420, "end_s": 58.5760},
            {"phase": "lift", "start_s": 58.5760, "end_s": 73.1480},
        ],
        "P9": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.5920},
            {"phase": "lift", "start_s": 4.5920, "end_s": 18.8180},
            {"phase": "rest", "start_s": 18.8180, "end_s": 26.7700},
            {"phase": "lift", "start_s": 26.7700, "end_s": 40.2180},
            {"phase": "rest", "start_s": 40.2180, "end_s": 48.6620},
            {"phase": "lift", "start_s": 48.6620, "end_s": 63.0940},
        ],
        "P10": [
            {"phase": "rest", "start_s": 0.0, "end_s": 4.9460},
            {"phase": "lift", "start_s": 4.9460, "end_s": 18.6860},
            {"phase": "rest", "start_s": 18.6860, "end_s": 27.9280},
            {"phase": "lift", "start_s": 27.9280, "end_s": 43.1300},
            {"phase": "rest", "start_s": 43.1300, "end_s": 51.8840},
            {"phase": "lift", "start_s": 51.8840, "end_s": 65.1800},
        ]
    }
    # PART 1b: lift to rest segmentation
    subjects_emg: dict[str, Subject] = {}
    for subject_id in ids:
        # Note: data folder should exist
        path = os.path.join("local/data", f"BSL-EMG-Book{subject_id}.csv")
        subject_event = subject_events[subject_id]
        # where padding
        subj = Subject.from_csv(name=subject_id, path=path, fs=FS_EMG, target_len=EMG_LEN)
        subjects_emg[subject_id] = subj

    # check load results
    print(subjects_emg.keys())

    # example with "P2"
    print(subjects_emg["P2"].df.info())
    s = subjects_emg["P2"]
    TimeSrsTools.emg_preprocess_hilbert(s)
    plt.figure(figsize=(10, 4))
    plt.plot(s.df["time"], s.df["EMG_env"], label="Hilbert-envelope")
    plt.title("EMG Hilbert-envelope – " + s.name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # call emg_preprocess for all


    # PART 2: PPG Processing:
    print("\n=== STARTING PPG PROCESSING ===")
    # Load full dataset
    # Note: change this to ourData.csv location
    ppg_data_path = "./local/data/ourData.csv"
    if not os.path.exists(ppg_data_path):
        raise FileNotFoundError(f"PPG Data file not found at {ppg_data_path}. Can't process ppg.")

    # Load full dataset
    df = pd.read_csv(ppg_data_path)
    df = df.drop(columns=["tstamp"], errors="ignore")

    # Build list of subject signals
    signals = []

    patient_arrays = [arr for idx, arr in enumerate(df.to_numpy()) if idx != 4]
    # align patient naming convention
    for idx, (key, subject) in enumerate(subjects_emg.items()):
        events = subject_events[key]
        arr = patient_arrays[idx]

        subject = subject.resample_data(200)
        sig = PPGProcTools.create_dotmaps_for_pyPPG(arr, str(idx + 2), RECORDER_PPG_FS)
        signals.append(sig)

        try:
            process_ppg_subject(idx + 2, sig)

            signal = PPGProcTools.preprocess_signal(sig)

            # Filter out first 5 seconds
            usable_signal = signal.vpg[signal.fs * 5: len(subject.df) + signal.fs * 5]

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

            passive_windows_rms = [np.sqrt(np.mean(window["Integrated EMG"] ** 2)) for window in subject.passive_windows]
            passive_windows_mav = [np.mean(np.abs(window["Integrated EMG"])) for window in subject.passive_windows]
            passive_windows_ppg_amplitude = [np.max(window["ppg"]) for window in subject.passive_windows]
        except Exception as e:
            print(f" Error processing PPG subject {idx+2}: {e}")
            continue

    print("\nAll PPG subjects processed.")

    # plot