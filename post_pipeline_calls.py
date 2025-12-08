import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyPPG import Fiducials
from PPG_section.PPG_handle import PPGProcTools
from main import PPG_BASE_DIR, PPG_TIME_ARR, RECORDER_PPG_FS
from pyPPG.datahandling import plot_fiducials, save_data, load_data

"""
sort of unfinished file. Supposed to run functions after processing for visualization.
"""

# config
# Match these with your main pipeline settings
DO_PPG_PLOT = False  # disable live plotting for batch PPG processing
PPG_SEGMENT = (0, 30)  # segment in seconds to visualize

# Output directory for PPG
os.makedirs(PPG_BASE_DIR, exist_ok=True)


def create_new_ppg_plots(index, signal):
    """
    Reloads raw and processed data for a specific subject and re-generates
    the plots (Raw, Filtered, Fiducials) without re-running the heavy processing pipeline.
    """

    # Setup paths
    subject_dir = os.path.join(PPG_BASE_DIR, f"Subject_{index}")

    # Naming
    title_root = PPGProcTools.naming_convention(signal.name, PPG_SEGMENT)

    # Raw segment plot
    PPGProcTools.plot_segment_for_signal(
        PPG_TIME_ARR, signal.v, title_root + "_raw", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )

    # Filtered plots
    PPGProcTools.plot_processed_signal_variants(
        PPG_TIME_ARR, signal, title_root + "_filtered", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep, do_plot=DO_PPG_PLOT
    )
    fid_title = PPGProcTools.naming_convention(signal.name + "_Fiducial", PPG_SEGMENT)
    df_fp = PPGProcTools.load_fiducials_from_csv(os.path.join(subject_dir, "Fiducial_points", fid_title))
    fp_loaded = Fiducials(df_fp)
    # Generate the canvas using pyPPG's tool
    canvas = plot_fiducials(signal, fp_loaded, savefig=False, savingfolder=subject_dir, legend_fontsize=12,
                            show_fig=False)

    # Pass canvas to our custom saver/plotter
    PPGProcTools.fiducials_segment_plot(
        canvas, title_root + "_fiducials", xlim=PPG_SEGMENT,
        save_dir=subject_dir + os.sep
    )


if __name__ == "__main__":
    ids = ['P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    print("Running Post-Pipeline Visualizations...\n")
    ppg_data_path = "./local/data/ourData.csv"
    # Load full dataset
    df = pd.read_csv(ppg_data_path)
    df = df.drop(columns=["tstamp"], errors="ignore")

    # Process all subjects
    for idx, sig in enumerate(df.to_numpy()):
        if idx == 4:
            print("Ignore 4th subject (not our data)")
            continue
        try:
            sig = PPGProcTools.create_dotmaps_for_pyPPG(sig, ids[idx], RECORDER_PPG_FS)
            create_new_ppg_plots(ids[idx], sig)
        except Exception as e:
            print(f" Error processing PPG subject {ids[idx]}: {e}")
            continue


    print("\nDone. Use these functions to inspect specific issues found in the full run.")
