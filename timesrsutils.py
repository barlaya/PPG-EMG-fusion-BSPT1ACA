import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, hilbert, welch, resample
import pywt
from numpy.lib.stride_tricks import sliding_window_view


class TimeSrsTools:

    @staticmethod
    def emg_bandpass_rectify(x, fs, low=20, high=220, order=4):
        nyq = 0.5 * fs
        low_norm = low / nyq
        high_norm = high / nyq
        if high_norm >= 1.0:
            high_norm = 0.99
        if low_norm <= 0.0:
            low_norm = 1e-6
        if low_norm >= high_norm:
            raise ValueError(
                f"Invalid band: low={low}Hz, high={high}Hz for fs={fs}Hz "
                f"(normalized low={low_norm}, high={high_norm})"
            )
        b, a = butter(order, [low/nyq, high/nyq], btype='bandpass')
        x_filt = filtfilt(b, a, x)
        x_rect = np.abs(x_filt)
        return x_filt, x_rect

    @staticmethod
    def emg_hilbert_envelope(x_rect):
        calc = hilbert(x_rect)
        envelope = np.abs(calc)
        return envelope

    @staticmethod
    def emg_preprocess_hilbert(subject, col="EMG", low=20, high=220):
        df = subject.df
        fs = subject.fs
        # neurokit numeric-clean version
        x = nk.signal_sanitize(df[col].values.astype(float))
        x_filt, x_rect = TimeSrsTools.emg_bandpass_rectify(x, fs, low=low, high=high)
        x_env = TimeSrsTools.emg_hilbert_envelope(x_rect)
        subject.set_column("EMG_filt", x_filt)
        subject.set_column("EMG_rect", x_rect)
        subject.set_column("EMG_env", x_env)
        return subject.df

    @staticmethod
    def lowpass(x, fs, cutoff, order=4):
        nyq = 0.5 * fs
        if not (0 < cutoff < nyq):
            raise ValueError(f"Invalid lowpass cutoff={cutoff} Hz for fs={fs} Hz "
                f"(must satisfy 0 < cutoff < fs/2={nyq}).")
        b, a = butter(order, cutoff, btype="lowpass", fs=fs)
        y = filtfilt(b, a, x)
        return y

    @staticmethod
    def window_dataframe(df: pd.DataFrame, window_size: int,
                         overlap: int = 0) -> list[pd.DataFrame]:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        step = window_size - overlap
        windows: list[pd.DataFrame] = []

        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            window = df.iloc[start:end].copy()
            windows.append(window)
        return windows

    @staticmethod
    def welchsm(x, fs, nperseg=None):
        f, Pxx = welch(x, fs, nperseg=nperseg)
        return Pxx, f

    @staticmethod
    def median_freq(f, Pxx):
        cumsum = np.cumsum(Pxx)
        total = cumsum[-1]
        idx = np.searchsorted(cumsum, total/2.0)
        return f[idx]

    @staticmethod
    def dwt_energy(x, wavelet="db4", level=4):
        coeffs = pywt.wavedec(x, wavelet, level=level)
        energies = [np.sum(c**2) for c in coeffs]
        return np.array(energies)

    @staticmethod
    def window_rms(x):
        return np.sqrt(np.mean(x**2))

    @staticmethod
    def waveform_length(x):
        return np.sum(np.abs(np.diff(x)))

    @staticmethod
    def zero_crossing(x):
        return ((x[:-1] * x[1:]) < 0).sum()

    @staticmethod
    def sqi_power_ratio(x, fs, band=(20, 450)):
        f, Pxx = welch(x, fs=fs)
        total = np.sum(Pxx)
        mask = (f >= band[0]) & (f < band[1])
        band_power = np.sum(Pxx[mask])
        return band_power / total if total > 0 else 0.0

    @staticmethod
    def sqi_saturation(x, sat_min=-1.0, sat_max=1.0):
        """
        Fraction of samples within 1% of ADC range as a simple 'clipping' SQI.
        Tune sat_min/max to your ADC.
        """
        eps = 0.01 * (sat_max - sat_min)
        return np.mean((x < sat_min + eps) | (x > sat_max - eps))

    # resampling
    def resample_signal(x, fs_old, fs_new):
        """
        Simple Fourier-based resampling to fs_new.
        """
        n_old = len(x)
        n_new = int(n_old * fs_new / fs_old)
        return resample(x, n_new)

    @staticmethod
    def slide_window(x, window_samples, step_samples):
        """
        Return array of shape (n_windows, window_samples)
        """
        sw = sliding_window_view(x, window_shape=window_samples)
        return sw[::step_samples]


SUBJECT_EVENTS = {
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
