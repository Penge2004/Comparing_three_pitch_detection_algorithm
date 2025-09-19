import aubio
import matplotlib.pyplot as plt

# ================ CONFIG ================
wav_file = ""
win_s = 1024  # window size (FFT)
hop_s = 256  # hop size (frame length)
t_start = 0.0  # start time in seconds
t_end = 0.9  # end time in seconds

# ================ SETUP =================
# initialize aubio source
source = aubio.source(wav_file, 0, hop_s)
sr = source.samplerate

# pitch detectors
detectors = {
    "YIN": aubio.pitch("yin", win_s, hop_s, sr),
    "MComb": aubio.pitch("mcomb", win_s, hop_s, sr),
    "Schmitt": aubio.pitch("schmitt", win_s, hop_s, sr)
}

for det in detectors.values():
    det.set_unit("Hz")
    det.set_tolerance(0.8)
    det.set_silence(-40)

# ================== FILTER CONFIG ==================
# Between C3 and A5
expected_min = 120  # Hz
expected_max = 900  # Hz

# ================== PROCESS ==================
pitches = {name: [] for name in detectors}
times = []

total_frames = 0
start_frame = int(t_start * sr / hop_s)
end_frame = int(t_end * sr / hop_s)

while True:
    frame, read = source()
    if read == 0:
        break
    if total_frames < start_frame:
        total_frames += 1
        continue
    if total_frames > end_frame:
        break

    times.append(total_frames * hop_s / sr)
    for name, det in detectors.items():
        f = det(frame)[0]
        # FILTER: only keep pitches within expected range
        if expected_min <= f <= expected_max:
            pitches[name].append(f)
        else:
            pitches[name].append(None)
    total_frames += 1

# ================ PLOT =================
plt.figure(figsize=(10, 5))
for name, data in pitches.items():
    plt.plot(times, data, label=name)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Pitch detection ({t_start}s to {t_end}s)")
plt.grid(True)
plt.legend()
plt.show()
