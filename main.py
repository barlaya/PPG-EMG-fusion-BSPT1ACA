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
FS_EMG = 500.
FS_PPG_TARGET = float(RECORDER_PPG_FS)
TARGET_DURATION_S = 120
EMG_LEN = 60000
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
    subjects_emg: dict[str, Subject] = {}
    for subject_id in ids:
        # Note: data folder should exist
        path = os.path.join("data", f"BSL-EMG-Book{subject_id}.csv")
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
    for i, arr in enumerate(df.to_numpy()):
        signals.append(PPGProcTools.create_dotmaps_for_pyPPG(arr, str(i), RECORDER_PPG_FS))

    # Process all subjects
    for idx, sig in enumerate(signals):
        if idx == 4:
            print("Ignore 4th subject (not our data)")
            continue
        try:
            process_ppg_subject(idx, sig)
        except Exception as e:
            print(f" Error processing PPG subject {idx}: {e}")
            continue

    print("\nAll PPG subjects processed.")