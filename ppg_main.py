import os
import numpy as np
import pandas as pd

from pyPPG import Fiducials
import pyPPG.ppg_sqi as SQI
from pyPPG.datahandling import plot_fiducials, save_data
from PPG_section.PPG_handle import PPGProcTools

# GLOBAL PARAMS:
# VERY SPECIFIC FOR OUR PPG RECORDER: https://readingtheheart.com/heartreader/
# All measurements ran for approx. 140 seconds = 28057 datapoints -> 1 sec = 28057/140
ppg_fs = int(28057 / 140)
t = np.arange(0, 28057) / ppg_fs
do_plot = False  # disable live plotting
from_to = (0, 140)  # segment to visualize

# Output dir
BASE_DIR = os.path.join("temp_dir", "PPG_AllSubjects")
os.makedirs(BASE_DIR, exist_ok=True)


def process_subject(index, signal):
    # ___SINGLE_SUBJECT_ANALYSIS___
    subject_dir = os.path.join(BASE_DIR, f"Subject_{index}")
    os.makedirs(subject_dir, exist_ok=True)

    print(f"\n--- Processing Subject {index} ---")

    # Naming
    title_root = PPGProcTools.naming_convention(signal.name, from_to)

    # 1. Raw segment plot
    PPGProcTools.plot_segment_for_signal(t, signal.v, title_root + "_raw", xlim=from_to,
                                         save_dir=subject_dir + os.sep, do_plot=do_plot)

    # 2. Preprocess
    signal = PPGProcTools.preprocess_signal(signal)

    # 3. Filtered plots
    PPGProcTools.plot_processed_signal_variants(t, signal, title_root + "_filtered", xlim=from_to,
                                                save_dir=subject_dir + os.sep, do_plot=do_plot)

    # 4. Fiducials
    s, fp, fid = PPGProcTools.compute_fiducials(signal)
    canvas = plot_fiducials(s, fp, savefig=False, savingfolder=subject_dir, legend_fontsize=12, show_fig=False)
    PPGProcTools.fiducials_segment_plot(canvas, title_root + "_fiducials", xlim=from_to, save_dir=subject_dir + os.sep)

    # 5. SQI
    ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    print(f"Subject {index} - Mean PPG SQI = {ppgSQI}%")

    # 6. Biomarkers
    bm_defs, bm_vals, bm_stats, bm = PPGProcTools.compute_biomarkers(s, fp)

    # 7. Save results
    fp_new = Fiducials(fp.get_fp() + s.start_sig)

    save_data(s=s, fp=fp_new, bm=bm, savingformat='csv', savingfolder=subject_dir)

    print(f"Finished Subject {index}")


if __name__ == "__main__":

    # Load full dataset
    df = pd.read_csv("./local/data/ourData.csv")
    df = df.drop(columns=["tstamp"], errors="ignore")

    # Build list of subject signals
    signals = []
    for i, arr in enumerate(df.to_numpy()):
        signals.append(PPGProcTools.create_dotmaps_for_pyPPG(arr, str(i), ppg_fs))

    # Process all subjects
    for idx, sig in enumerate(signals):
        if idx == 4:
            print("Ignore 4th subject (not our data)")
            continue
        try:
            process_subject(idx, sig)
        except Exception as e:
            print(f" Error processing subject {idx}: {e}")
            continue

    print("\nAll subjects processed.")
