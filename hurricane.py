import numpy as np
import scipy.signal as signal
from pydub import AudioSegment

duration = 10
sample_rate = 44100
num_samples = int(sample_rate * duration)

white_noise = np.random.normal(0, 1, num_samples)

cutoff = 300
order = 4
b, a = signal.butter(order, cutoff / (0.5 * sample_rate), btype='low')
filtered_noise = signal.lfilter(b, a, white_noise)

num_env_points = int(duration * 10)
env_times = np.linspace(0, duration, num_env_points)
random_envelope = np.random.uniform(0.7, 1.0, num_env_points)
time_array = np.linspace(0, duration, num_samples)
envelope = np.interp(time_array, env_times, random_envelope)

modulated_noise = filtered_noise * envelope

modulated_noise /= np.max(np.abs(modulated_noise))
modulated_noise *= 0.9

def numpy_to_audio_segment(samples, sample_rate=44100):
    samples_int16 = np.int16(samples * 32767)
    audio_segment = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return audio_segment

audio = numpy_to_audio_segment(modulated_noise, sample_rate)
audio.export("hurricane_sound.mp3", format="mp3")
print("Exported hurricane_sound.mp3")
