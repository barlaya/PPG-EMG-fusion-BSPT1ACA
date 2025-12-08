import numpy as np
import pandas as pd
from dotmap import DotMap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from timesrsutils import TimeSrsTools, SUBJECT_EVENTS
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import os


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

    # note: do we even use this?
    @staticmethod  # modified version of load_fiducials: https://pyppg.readthedocs.io/en/latest/_modules/pyPPG/datahandling.html#load_fiducials
    def load_fiducials_from_csv(csv_path):
        """
        Loads fiducial points from a CSV file into a pandas DataFrame.

        Expected CSV columns:
        Index of pulse, on, sp, dn, dp, off, u, v, w, a, b, c, d, e, f, p1, p2
        """

        try:
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                return None

            # Read CSV
            df = pd.read_csv(csv_path)

            # Cleanup: Drop 'Index of pulse' if it exists as we just want the fiducial columns
            if 'Index of pulse' in df.columns:
                df = df.drop(columns=['Index of pulse'])

            # Ensure all standard pyPPG columns exist (fill missing with nan if necessary)
            expected_cols = ['on', 'sp', 'dn', 'dp', 'off', 'u', 'v', 'w', 'a', 'b', 'c', 'd', 'e', 'f', 'p1', 'p2']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan

            return df

        except Exception as e:
            print(f"Error loading fiducials from CSV: {e}")
            return None
