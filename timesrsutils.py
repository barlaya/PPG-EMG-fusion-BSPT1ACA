import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, hilbert, welch, resample
import pywt
from numpy.lib.stride_tricks import sliding_window_view

class TimeSrsTools:
    @staticmethod
    def emg_bandpass_rectify(x, fs, low=20, high=250, order=4):
        nyq = 0.5 * fs
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
    def emg_preprocess_hilbert(subject, col="EMG", low=20, high=250):
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


    '''
        @staticmethod
        def emg_envelope(x, fs, low=20, high=450, lp_cutoff=5):
            """
            1) bandpass EMG
            2) rectify
            3) lowpass envelope
            returns envelope, x filtered, x rectified
            """
            x_filt = TimeSrsTools.bandpass(x, fs, low, high)
            x_rect = np.abs(x_filt)
            env = TimeSrsTools.lowpass(x_rect, fs, lp_cutoff)
            return env, x_filt, x_rect
    '''
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
        mask = ( f>= band[0]) & (f<band[1])
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

    #resampling
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