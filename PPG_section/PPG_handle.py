import numpy as np
import pandas as pd
from dotmap import DotMap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM


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
        s.name = f"patient {index}"
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