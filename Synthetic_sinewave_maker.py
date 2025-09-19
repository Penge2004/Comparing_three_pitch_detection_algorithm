"""
Enhanced Audio Dataset Generator for Pitch Detection Algorithm Testing
Generates synthetic audio files with various modifications for testing YIN, Schmitt trigger, and MCOMB algorithms.

Author: Refactored for research project
Purpose: Generate controlled test cases with known ground truth for pitch detection evaluation
"""

import os
import csv
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration class for audio generation parameters."""
    sample_rate: int = 44100
    bit_depth: str = 'PCM_16'
    fade_duration_ms: float = 5.0  # Smooths the edges
    output_amplitude: float = 0.9  # Maximum amplitude to prevent clipping


@dataclass
class SynthConfig:
    """Configuration for synthesis parameters."""
    # Harmonic settings
    add_harmonics: bool = False
    num_harmonics: int = 3
    harmonic_decay: float = 0.5  # Each harmonic is this factor weaker than previous

    # Vibrato settings
    add_vibrato: bool = False
    vibrato_rate_hz: float = 6.0  # Typical vibrato rate
    vibrato_depth_cents: float = 30.0  # Pitch variation in cents (100 cents = 1 semitone)

    # Noise settings
    add_noise: bool = False
    snr_db_levels: List[Optional[float]] = None  # [None] for clean, [30, 20, 10] for noisy

    def __post_init__(self):
        if self.snr_db_levels is None:
            self.snr_db_levels = [None]  # Clean signal by default


class AudioSynthesizer:
    """Main class for generating synthetic audio signals."""

    # Standard musical note frequencies (Hz)
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

    def __init__(self, audio_config: AudioConfig = None):
        self.audio_config = audio_config or AudioConfig()

    def generate_pure_sine(self, frequency: float, duration: float) -> np.ndarray:

        num_samples = int(self.audio_config.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        return np.sin(2 * np.pi * frequency * t)

    def add_harmonics(self, signal: np.ndarray, fundamental_freq: float,
                      num_harmonics: int, decay_factor: float,
                      duration: float) -> np.ndarray:

        num_samples = int(self.audio_config.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        enhanced_signal = signal.copy()

        for harmonic_num in range(2, num_harmonics + 2):  # Start from 2nd harmonic
            harmonic_freq = fundamental_freq * harmonic_num
            amplitude = decay_factor ** (harmonic_num - 1)
            harmonic_wave = amplitude * np.sin(2 * np.pi * harmonic_freq * t)
            enhanced_signal += harmonic_wave

        return enhanced_signal

    def apply_vibrato(self, frequency: float, duration: float,
                      vibrato_rate: float, vibrato_depth_cents: float) -> np.ndarray:

        num_samples = int(self.audio_config.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Convert cents to frequency ratio
        # 1200 cents = 1 octave = frequency doubling
        depth_ratio = 2 ** (vibrato_depth_cents / 1200.0) - 1.0

        # Create instantaneous frequency multiplier
        freq_modulation = 1.0 + depth_ratio * np.sin(2 * np.pi * vibrato_rate * t)

        # Integrate to get phase (cumulative sum approximates integration)
        instantaneous_freq = frequency * freq_modulation
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.audio_config.sample_rate

        return np.sin(phase)

    def add_noise(self, signal: np.ndarray, snr_db: Optional[float],
                  random_generator: np.random.Generator) -> np.ndarray:
        """
        SNR is the ratio of signal power to noise power, expressed in decibels.
        Higher SNR = cleaner signal, lower SNR = noisier signal.
        signal: Clean audio signal
        snr_db: Signal-to-Noise ratio in dB (None for no noise)
        random_generator: NumPy random generator for reproducible noise
        """

        if snr_db is None:
            return signal

        # Calculate signal RMS (Root Mean Square) power
        signal_rms = np.sqrt(np.mean(signal ** 2))

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 20.0)

        # Calculate required noise RMS
        noise_rms = signal_rms / snr_linear

        # Generate Gaussian white noise
        noise = random_generator.normal(0, noise_rms, size=signal.shape)

        return signal + noise

    def apply_fade(self, signal: np.ndarray) -> np.ndarray:

        fade_samples = int(self.audio_config.sample_rate *
                           (self.audio_config.fade_duration_ms / 1000.0))

        if fade_samples <= 0 or fade_samples >= len(signal) // 2:
            return signal

        # Create fade envelope
        fade_window = np.ones(len(signal))
        fade_ramp = np.linspace(0, 1, fade_samples)

        # Apply fade-in and fade-out
        fade_window[:fade_samples] = fade_ramp
        fade_window[-fade_samples:] = fade_ramp[::-1]  # Reverse for fade-out

        return signal * fade_window

    def normalize_amplitude(self, signal: np.ndarray) -> np.ndarray:

        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            return signal / max_amplitude * self.audio_config.output_amplitude
        return signal

    def synthesize_note(self, frequency: float, duration: float,
                        synth_config: SynthConfig, seed: int) -> np.ndarray:

        random_gen = np.random.default_rng(seed)

        # Generate base signal
        if synth_config.add_vibrato:
            signal = self.apply_vibrato(
                frequency, duration,
                synth_config.vibrato_rate_hz,
                synth_config.vibrato_depth_cents
            )
        else:
            signal = self.generate_pure_sine(frequency, duration)

        # Add harmonics if requested
        if synth_config.add_harmonics and synth_config.num_harmonics > 0:
            signal = self.add_harmonics(
                signal, frequency,
                synth_config.num_harmonics,
                synth_config.harmonic_decay,
                duration
            )

        # Normalize before adding noise (prevents clipping)
        signal = self.normalize_amplitude(signal)

        # Apply fade to prevent clicks
        signal = self.apply_fade(signal)

        return signal


class DatasetGenerator:
    """Manages the generation of complete audio datasets with metadata."""

    def __init__(self, output_dir: str, base_seed: int = 12345):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.base_seed = base_seed
        self.synthesizer = AudioSynthesizer()

        # Metadata file setup
        self.metadata_path = self.output_dir / "metadata.csv"
        self._initialize_metadata_file()

    def _initialize_metadata_file(self):
        """Initialize the CSV metadata file with headers."""
        with open(self.metadata_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "filename", "note", "frequency_hz", "duration_s", "seed",
                "has_harmonics", "num_harmonics", "harmonic_decay",
                "has_vibrato", "vibrato_rate_hz", "vibrato_depth_cents",
                "snr_db", "created_utc"
            ])

    def _generate_filename(self, name):
        return f"{name}.wav"

    def _write_metadata_row(self, filename: str, note: str, frequency: float,
                            duration: float, seed: int, snr_db: Optional[float],
                            synth_config: SynthConfig):
        """Write a row to the metadata CSV file."""
        with open(self.metadata_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                filename, note, frequency, duration, seed,
                synth_config.add_harmonics,
                synth_config.num_harmonics if synth_config.add_harmonics else 0,
                synth_config.harmonic_decay if synth_config.add_harmonics else 0,
                synth_config.add_vibrato,
                synth_config.vibrato_rate_hz if synth_config.add_vibrato else 0,
                synth_config.vibrato_depth_cents if synth_config.add_vibrato else 0,
                snr_db if snr_db is not None else "",
                datetime.utcnow().isoformat()
            ])

    def generate_dataset(self, durations: List[float],
                         synth_config: SynthConfig) -> int:

        file_count = 0

        for note, frequency in self.synthesizer.NOTE_FREQUENCIES.items():
            for duration in durations:
                for repeat_idx in range(1):
                    seed = self.base_seed + file_count

                    for snr_db in synth_config.snr_db_levels:
                        # Generate the audio signal
                        signal = self.synthesizer.synthesize_note(
                            frequency, duration, synth_config, seed
                        )

                        # Add noise if specified
                        if synth_config.add_noise and snr_db is not None:
                            random_gen = np.random.default_rng(seed)
                            signal = self.synthesizer.add_noise(signal, snr_db, random_gen)
                            signal = self.synthesizer.normalize_amplitude(signal)

                        # Generate filename and save
                        filename = self._generate_filename(note)
                        filepath = self.output_dir / filename

                        # Save audio file
                        sf.write(
                            str(filepath),
                            signal,
                            self.synthesizer.audio_config.sample_rate,
                            subtype=self.synthesizer.audio_config.bit_depth
                        )

                        # Save metadata
                        self._write_metadata_row(
                            filename, note, frequency, duration, seed, snr_db, synth_config
                        )

                        file_count += 1

        return file_count


def main():
    """Main function demonstrating how to use the audio dataset generator."""

    # Configuration
    OUTPUT_DIR = "Output_used_in_interesting_results/1sec_60harm_6Hz_100cent_0db"

    DURATIONS = [1.0] # Here we can make changes or more than 1 Duration per run

    # Synthesis configuration - customize these parameters as needed
    synth_config = SynthConfig(
        # Harmonic settings
        add_harmonics=False,
        num_harmonics=60,
        harmonic_decay=0.5,  # Each harmonic is 50% weaker than the previous - like in alto sax

        # Vibrato settings
        add_vibrato=False,
        vibrato_rate_hz=6.0,  # 6 Hz vibrato rate
        vibrato_depth_cents=30,  # 30 cents pitch variation

        # Noise settings
        add_noise=False,
        snr_db_levels=[None]  # None for not activating
    )
    # 40 not noticeable, 0 extreme noise

    # Generate the dataset
    print("Generating audio dataset for pitch detection research...")
    print(f"Configuration:")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Note durations: {DURATIONS}")
    print(f"  - Harmonics: {'Yes' if synth_config.add_harmonics else 'No'} "
          f"({synth_config.num_harmonics} harmonics)")
    print(f"  - Vibrato: {'Yes' if synth_config.add_vibrato else 'No'}")
    print(f"  - Noise levels: {synth_config.snr_db_levels}")
    print()

    generator = DatasetGenerator(OUTPUT_DIR)
    total_files = generator.generate_dataset(DURATIONS, synth_config)

    print(f"Dataset generation complete!")
    print(f"Generated {total_files} audio files in '{OUTPUT_DIR}'")
    print(f"Metadata saved to '{generator.metadata_path}'")


if __name__ == "__main__":
    main()
