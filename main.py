import os
from backbone import Subject
from timesrsutils import TimeSrsTools
import matplotlib.pyplot as plt

FS_EMG = 500.
FS_PPG = 200.
TARGET_DURATION_S = 120
EMG_LEN = 60000 #int(TARGET_DURATION_S * FS_EMG)
PPG_LEN = int(FS_PPG * TARGET_DURATION_S)

if __name__ == '__main__':
    ids = ['P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    subjects_emg: dict[str, Subject] = {}
    for subject_id in ids:
        path = os.path.join("data", f"BSL-EMG-Book{subject_id}.csv")
        # where padding
        subj = Subject.from_csv(name=subject_id, path=path, fs=FS_EMG, target_len=EMG_LEN)
        subjects_emg[subject_id] = subj
    print(subjects_emg.keys())
    print(subjects_emg["P2"].df.info())

    s = subjects_emg["P2"]

    TimeSrsTools.emg_preprocess_hilbert(s)

    plt.figure(figsize=(10,4))
    plt.plot(s.df["time"], s.df["EMG_env"], label="Hilbert-envelope")
    plt.title("EMG Hilbert-envelope – " + s.name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
