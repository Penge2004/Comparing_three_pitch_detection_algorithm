import sounddevice as sd
import soundfile as sf
from pathlib import Path

# ================= CONFIG =================
OUTPUT_DIR = "real_sax_notes"  # folder to save recordings
DURATION = 1.0  # seconds per note
SAMPLE_RATE = 44100  # 44.1 kHz

# I deleted some notes because they are to high for the alto sax (and to my abilities)
NOTES = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3',
         'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4',
         'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5',
         'F#5', 'G5', 'G#5', 'A5']

NOTE_FREQUENCIES = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00,
    "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25,
    "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99,
    "G#5": 830.61, "A5": 880.00
}

# because alto saxophone is an Eb transposing instrument
concert_to_alto = {
    "C3": "A3", "C#3": "A#3", "D3": "B3", "D#3": "C4", "E3": "C#4",
    "F3": "D4", "F#3": "D#4", "G3": "E4", "G#3": "F4", "A3": "F#4",
    "A#3": "G4", "B3": "G#4", "C4": "A4", "C#4": "A#4", "D4": "B4",
    "D#4": "C5", "E4": "C#5", "F4": "D5", "F#4": "D#5", "G4": "E5",
    "G#4": "F5", "A4": "F#5", "A#4": "G5", "B4": "G#5", "C5": "A5",
    "C#5": "A#5", "D5": "B5", "D#5": "C6", "E5": "C#6", "F5": "D6",
    "F#5": "D#6", "G5": "E6", "G#5": "F6", "A5": "F#6"
}

# Create output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)


# ================= RECORD FUNCTION =================
def record_note(note_name: str, duration: float, sample_rate: int) -> None:
    print(f"\nGet ready to play {note_name} ({duration} sec)...")
    input("Press Enter to start recording...")

    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    filename = output_path / f"{note_name}.wav"
    sf.write(filename, audio, sample_rate)
    print(f"Saved {filename}")


# ================= MAIN =================
for note in ['A4']: # for separate notes, or you can make the whole recording at once with whole list
    alto_note = concert_to_alto[note]
    print(f"Please play {alto_note} on the alto sax (sounding concert {note})")
    record_note(note, DURATION, SAMPLE_RATE)

print("\nAll notes recorded successfully!")
