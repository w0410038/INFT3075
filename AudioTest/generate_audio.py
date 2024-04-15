import numpy as np
from scipy.io.wavfile import write

# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 5  # Duration in seconds
frequency = 440  # Frequency of the sine wave in Hz (A4 pitch)

# Generate time values
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# Generate sine wave
y = np.sin(2 * np.pi * frequency * t)

# Ensure amplitude is in int16 range
audio = np.int16(y * 32767)

# Write to WAV file
file_path = 'audioFile/sine_wave.wav'
write(file_path, sample_rate, audio)
print(f"Generated sine wave saved to {file_path}")