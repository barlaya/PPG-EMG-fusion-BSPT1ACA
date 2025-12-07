import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your custom tools
from PPG_section.PPG_handle import PPGProcTools
from pyPPG.datahandling import load_data

# config
# Match these with your main pipeline settings
RECORDER_PPG_FS = int(28057 / 140)
BASE_OUTPUT_DIR = os.path.join("temp_dir", "PPG_AllSubjects")


def load_subject_results(subject_idx):
    """
    Helper to load processed data for a specific subject from the results folder.
    Returns the dataframe of signals and the fiducials if available.
    """
    subj_dir = os.path.join(BASE_OUTPUT_DIR, f"Subject_{subject_idx}")

    if not os.path.exists(subj_dir):
        raise FileNotFoundError(f"No processed data found for Subject {subject_idx}. Run main.py first.")

    # Assuming pyPPG standard saving format (usually saves as .csv or .mat)
    # We try to find the main data file. Adjust naming based on pyPPG output.
    try:
        # Example: pyPPG often saves a file containing signal data
        # This is a general loader; specific implementation depends on pyPPG version
        data_files = [f for f in os.listdir(subj_dir) if f.endswith('.csv')]
        if not data_files:
            raise FileNotFoundError("No CSV files found in subject directory.")

        print(f"Loading data for Subject {subject_idx} from {subj_dir}...")
        # Load the first matching CSV (or specific naming convention if known)
        df = pd.read_csv(os.path.join(subj_dir, data_files[0]))
        return df
    except Exception as e:
        print(f"Could not load data for Subject {subject_idx}: {e}")
        return None


def visualize_specific_segment(subject_idx, start_sec, end_sec, signal_col='ppg_filtered'):
    """
    Plots a specific time window for inspection.
    Useful for checking artifacts or specific events found in the report.
    """
    df = load_subject_results(subject_idx)
    if df is None: return

    # Ensure time column exists or create one from index
    if 'time' not in df.columns:
        df['time'] = np.arange(len(df)) / RECORDER_PPG_FS

    # Filter for the segment
    mask = (df['time'] >= start_sec) & (df['time'] <= end_sec)
    segment = df.loc[mask]

    if segment.empty:
        print(f"No data found between {start_sec}s and {end_sec}s")
        return

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(segment['time'], segment[signal_col], label=f"Subject {subject_idx} - {signal_col}")

    # Optional: Highlight peak markers if they exist in the DF
    if 'peak_idx' in segment.columns:
        peaks = segment[segment['peak_idx'] == 1]
        plt.scatter(peaks['time'], peaks[signal_col], c='red', marker='x', label="Detected Peaks")

    plt.title(f"Inspection: Subject {subject_idx} | Window: {start_sec}-{end_sec}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def print_subject_report(subject_idx):
    """
    Prints a summary of the biomarkers/SQI without plotting,
    useful for quick console checks.
    """
    subj_dir = os.path.join(BASE_OUTPUT_DIR, f"Subject_{subject_idx}")
    stats_file = os.path.join(subj_dir, "biomarkers.csv")  # Hypothetical filename

    if os.path.exists(stats_file):
        print(f"\n--- Report for Subject {subject_idx} ---")
        stats = pd.read_csv(stats_file)
        print(stats.to_string())
    else:
        print(f"No biomarker report found for Subject {subject_idx}")


# ==========================================
# PLAYGROUND EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Running Post-Pipeline Visualizations...\n")

    # --- EXAMPLE 1: Inspect a noisy segment ---
    # Perhaps Subject 2 had a low SQI report. Let's look at the first 10 seconds.
    # visualize_specific_segment(subject_idx=2, start_sec=0, end_sec=10)

    # --- EXAMPLE 2: Zoom in on a specific peak ---
    # Looking at seconds 50 to 52 for Subject 5
    # visualize_specific_segment(subject_idx=5, start_sec=50, end_sec=52)

    # --- EXAMPLE 3: Compare raw vs filtered (if columns exist) ---
    # Custom ad-hoc plotting script
    try:
        df_3 = load_subject_results(3)
        if df_3 is not None and 'ppg_raw' in df_3.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df_3['ppg_raw'][:500], label='Raw', alpha=0.5)
            plt.plot(df_3['ppg_filtered'][:500], label='Filtered', color='black')
            plt.title("Subject 3: Filter Check (First 500 samples)")
            plt.legend()
            plt.show()
    except Exception as e:
        print("Skipping Example 3 (Data not found)")

    print("\nDone. Use these functions to inspect specific issues found in the full run.")