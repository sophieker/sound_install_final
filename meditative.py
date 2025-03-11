import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment

# Parameters
duration = 60            # Duration in seconds
sample_rate = 44100      # Samples per second
num_samples = int(duration * sample_rate)
t = np.linspace(0, duration, num_samples, endpoint=False)

# Drone synthesis: Create multiple sine waves for a rich, meditative tone.
freq1 = 110              # Base frequency (Hz)
freq2 = 110.2            # Very slight detuning for natural richness
freq3 = 220              # Harmonic overtone

s1 = np.sin(2 * np.pi * freq1 * t)
s2 = np.sin(2 * np.pi * freq2 * t)
s3 = np.sin(2 * np.pi * freq3 * t)

# Mix the tones (overtone weighted lower)
drone = s1 + s2 + 0.5 * s3

# Removed the amplitude modulation envelope to avoid pulsation.
# Instead, we only apply a smooth fade-in to ease the transition.
fade_in_duration = 5    # seconds for fade-in
fade_in_samples = int(fade_in_duration * sample_rate)
fade_in_envelope = np.ones(num_samples)
fade_in_envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
drone *= fade_in_envelope

# Optional: Apply a subtle reverb effect via FFT convolution to add spatial depth.
apply_reverb = True
if apply_reverb:
    reverb_time = 0.3  # seconds for the impulse response length
    reverb_samples = int(reverb_time * sample_rate)
    # Create a decaying exponential impulse response
    ir = np.exp(-np.linspace(0, 3, reverb_samples))
    ir /= np.sum(ir)  # Normalize the impulse response
    # Convolve using FFT-based convolution for efficiency
    drone = fftconvolve(drone, ir, mode='full')[:num_samples]

# Normalize to prevent clipping
drone /= np.max(np.abs(drone))
drone *= 0.9

# Function to convert a NumPy array to a PyDub AudioSegment.
def numpy_to_audio_segment(samples, sample_rate=44100):
    samples_int16 = np.int16(samples * 32767)
    audio_segment = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return audio_segment

# Convert the NumPy audio to an AudioSegment and export as MP3.
audio = numpy_to_audio_segment(drone, sample_rate)
audio.export("meditative_music.mp3", format="mp3")
print("Exported meditative_music.mp3")
