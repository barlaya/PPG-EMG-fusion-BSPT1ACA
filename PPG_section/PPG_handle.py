# András: TODO tidy this up
# András: no
import numpy as np
import pandas as pd
from dotmap import DotMap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from timesrsutils import SUBJECT_EVENTS
from pyPPG.datahandling import load_data
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import os
import scipy.io
import json


# create separate PPG handler
class PPGProcTools:
    @staticmethod
    def naming_convention(name: str, from_to: tuple):
        return '%s_btwn_%s-%s' % (name, from_to[0], from_to[1])

    @staticmethod
    def create_dotmaps_for_pyPPG(array: np.ndarray, index: str, fs: int):
        # create dotmaps for pyPPG; Basically copy of load_data.py from pyPPG.
        s = DotMap()
        s.start_sig = 0
        s.end_sig = len(array)
        s.v = array.astype(float)  # pyPPG works with floats!
        s.fs = fs
        s.name = f"Subject_{index}"
        return s

    @staticmethod
    # TODO: how to set it so input is specific datatype
    def plot_segment_for_signal(x: np.ndarray, y: np.ndarray, title: str, xlim: tuple = (0, 10),
                                save_dir: str = "./", do_plot=False):
        fig, ax = plt.subplots()
        ax.plot(x, y, color='blue', linewidth=1)
        ax.set(xlabel='Time (s)', ylabel='Amplitude of PPG', title=title, xlim=xlim)
        if do_plot:
            plt.show()
        plt.savefig(save_dir + title + ".png")
        plt.close()

    @staticmethod
    def plot_processed_signal_variants(x: np.ndarray, y: DotMap, title: str, xlim: tuple = (0, 10),
                                       save_dir: str = "./", do_plot=False):
        # setup figure
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False)

        # plot filtered PPG signal
        ax1.plot(x, y.ppg)
        ax1.set(xlabel='', ylabel='PPG')

        # plot first derivative
        ax2.plot(x, y.vpg)
        ax2.set(xlabel='', ylabel='PPG\'')

        # plot second derivative
        ax3.plot(x, y.apg)
        ax3.set(xlabel='', ylabel='PPG\'\'')

        # plot third derivative
        ax4.plot(x, y.jpg)
        ax4.set(xlabel='Time (s)', ylabel='PPG\'\'\'')
        # spacing + naming +show plot
        plt.xlim(xlim)
        fig.suptitle(title)
        if do_plot:
            plt.show()
        plt.savefig(save_dir + title + ".png")
        plt.close(fig)

    @staticmethod
    def plot_vpg_emg(x: np.ndarray, signal_1: np.ndarray, signal_2: np.ndarray, subject_id: str,
                     title: str, xlim: tuple = (0, 10), save_dir: str = "./", do_plot=False):

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Signal 1 (e.g., PPG/VPG)
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal 1 (PPG/VPG)', color=color1, fontweight='bold')
        ax1.plot(x, signal_1, color=color1, linewidth=1.5, label='Signal 1')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Create secondary axis (Right Y-axis)
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Signal 2 (EMG)', color=color2, fontweight='bold')
        # Plot Signal 2 with transparency to see overlap
        ax2.plot(x, signal_2, color=color2, linewidth=1, alpha=0.6, label='Signal 2')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Highlight Lift/Rest Events
        # TODO fix duplicating legend text: rest lift rest lift etc. -> should be rest lift
        events = SUBJECT_EVENTS[subject_id]

        for event in events:
            start = event["start_s"]
            end = event["end_s"]
            phase = event["phase"].lower()

            # Skip drawing if event is entirely outside current view
            if end < xlim[0] or start > xlim[1]:
                continue
            mid_point = (start + end) / 2
            if phase == "lift":
                ax1.axvspan(start, end, color='green', alpha=0.15, label=phase)
                ax1.text(mid_point, ax1.get_ylim()[1], "LIFT",
                         ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
            elif phase == "rest":
                ax1.axvspan(start, end, color='gray', alpha=0.1, label=phase)
                ax1.text(mid_point, ax1.get_ylim()[1], "REST",
                         ha='center', va='bottom', fontsize=9, color='gray', fontweight='bold')
        # Set Limits and Title
        ax1.set_xlim(xlim)
        fig.suptitle(title, fontsize=14)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        fig.tight_layout()

        plt.savefig((save_dir + title + ".png"), dpi=150)

        if do_plot:
            plt.show()

        plt.close(fig)

    @staticmethod
    def fiducials_segment_plot(canvas: FigureCanvas, title: str, xlim: tuple = (0, 10), save_dir: str = ""):
        fs = 200
        # convert xlim seconds to indexes
        new_xlim = (int(xlim[0] * fs), int(xlim[1] * fs))  # !manual freq setup! TODO: should create self variables?

        # create 5 tick positions
        tick_labels = np.linspace(xlim[0], xlim[1], num=5)
        tick_positions = np.linspace(new_xlim[0], new_xlim[1], num=5)

        # set canvas parameters
        for ax in canvas.figure.axes:
            ax.set_xlim(new_xlim)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        canvas.draw()

        # Save segment
        canvas.figure.savefig(save_dir + title + ".png")

    @staticmethod
    def preprocess_signal(signal):
        # Apply filtering + smoothing.
        signal.filtering = True
        signal.fL = 0.5000001
        signal.fH = 12
        signal.order = 4
        signal.sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}

        prep = PP.Preprocess(
            fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins
        )
        signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)
        return signal

    @staticmethod
    def compute_fiducials(signal):
        # Compute fiducials and apply correction.
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction = pd.DataFrame()
        correction.loc[0, corr_on] = True
        signal.correction = correction

        s = PPG(signal)
        fpex = FP.FpCollection(s=s)
        fid = fpex.get_fiducials(s=s)

        fp = Fiducials(fp=fid)
        return s, fp, fid

    @staticmethod
    def compute_biomarkers(s, fp):
        # Extract biomarkers
        bmex = BM.BmCollection(s=s, fp=fp)
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
        return bm_defs, bm_vals, bm_stats, bm

    @staticmethod
    def load_subject_data(subject_id: str, base_dir: str):
        subj_name = f"Subject_{subject_id}"
        subject_dir = os.path.join(base_dir, subj_name)
        manifest_path = os.path.join(subject_dir, "manifest.json")

        # Load the File Map
        with open(manifest_path, "r") as f:
            files = json.load(f)

        # 1. LOAD SIGNAL (Reconstruct PPG Class)
        temp_s = DotMap()
        # struct_as_record=True gives us numpy structured arrays
        mat = scipy.io.loadmat(files['data_struct_mat'], squeeze_me=True, struct_as_record=True)
        for k, v in mat.items():
            if not k.startswith("__"):
                # This figured out what happened:
                # temp_s[k] = v
                # print("harrow?", temp_s)

                # Handle 'correction' (NumPy record -> DataFrame)
                if k == 'correction':
                    corr_dict = {name: [v[name].item()] for name in v.dtype.names}
                    temp_s[k] = pd.DataFrame(corr_dict)

                # Handle 'sm_wins' (NumPy record -> Dict)
                elif k == 'sm_wins':
                    # Convert to simple dict: {'ppg': 50, 'vpg': 10...}
                    temp_s[k] = {name: v[name].item() for name in v.dtype.names}

                # Handle 0-d arrays (scalars like fs, check_ppg_len) that might behave oddly
                elif np.ndim(v) == 0 and isinstance(v, np.generic):
                    temp_s[k] = v.item()

                else:
                    temp_s[k] = v

        # Pass the cleaned DotMap to the PPG constructor
        s = PPG(s=temp_s, check_ppg_len=False)

        # Load Fiducials (Fiducials Object)
        df_fp = pd.read_csv(files['fiducials_csv'], index_col=0)
        fp_obj = Fiducials(fp=df_fp)
        # Load Biomarkers (Biomarkers Object)
        bm_vals = {}
        bm_stats = {}
        bm_defs = {}
        # The 4 biomarker categories generated by pyPPG
        categories = ['ppg_sig', 'sig_ratios', 'ppg_derivs', 'derivs_ratios']

        for cat in categories:
            vals_key = f"{cat}_vals_csv"
            stats_key = f"{cat}_stats_csv"
            defs_key = f"{cat}_defs_csv"

            if vals_key in files:
                bm_vals[cat] = pd.read_csv(files[vals_key], index_col=0)

            if stats_key in files:
                bm_stats[cat] = pd.read_csv(files[stats_key], index_col=0)

            if defs_key in files:
                bm_defs[cat] = pd.read_csv(files[defs_key], index_col=0)

        bm_obj = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)

        return s, fp_obj, bm_obj
