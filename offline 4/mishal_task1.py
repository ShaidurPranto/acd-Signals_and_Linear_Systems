import numpy as np
import matplotlib.pyplot as plt
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000

def dft(signal):
    n = len(signal)
    X = np.zeros(n, dtype=complex)
    for k in range(n):
        for t in range(n):
            X[k] += signal[t] * np.exp(-2j * np.pi * k * t / n)
    return X

def idft(freq_signal):
    n = len(freq_signal)
    x = np.zeros(n, dtype=complex)
    for t in range(n):
        for k in range(n):
            x[t] += freq_signal[k] * np.exp(2j * np.pi * k * t / n)
    return x / n

def cross_correlation(signal_a, signal_b):
    n = len(signal_a)
    dft_a = dft(signal_a)
    dft_b = dft(signal_b)
    cross_corr_freq =  np.conj(dft_b)*dft_a
    cross_corr = idft(cross_corr_freq)
    result = cross_corr.real
    result = np.roll(result, n // 2)
    return result
def sample_lag_detection(signal):
   
    return np.argmax(signal)

def distance_estimation(lag):
    return abs(lag) * wave_velocity / sampling_rate



#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = 3
    #shift_samples = -3
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
  
    
    return signal_A, signal_B

#implement other functions and logic
# Generate signals
signal_A, signal_B = generate_signals()

# Compute DFTs
dft_A = dft(signal_A)
dft_B = dft(signal_B)

# Compute magnitude spectra
magnitude_A = np.abs(dft_A)
magnitude_B = np.abs(dft_B)

# Compute cross-correlation
cross_corr = cross_correlation(signal_A, signal_B)
cross_corr2 = np.correlate(signal_A, signal_B, mode='same')
N = len(cross_corr)
lag=sample_lag_detection(cross_corr)

lag -= N//2
print(f"Sample Lag: {lag}")

# Sample lags for plotting cross-correlation
lags = np.arange(-n // 2, n // 2)

# Plot Signal A
plt.figure(figsize=(8, 6))
plt.stem(samples, signal_A, linefmt="blue", markerfmt="bo", basefmt="black")
plt.title("Signal A (Station A)")
plt.xlabel("Sample Number ")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

#Plot Frequency Spectrum of Signal A
plt.figure(figsize=(8, 6))
plt.stem(samples, magnitude_A, linefmt="blue", markerfmt="bo", basefmt="black")
plt.title("Frequency Spectrum of Signal A")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Plot Signal B
plt.figure(figsize=(8, 6))
plt.stem(samples, signal_B, linefmt="red", markerfmt="ro", basefmt="black")
plt.title("Signal B (Discrete)")
plt.xlabel("Sample Number (n)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Plot Frequency Spectrum of Signal B
plt.figure(figsize=(8, 6))
plt.stem(samples, magnitude_B, linefmt="red", markerfmt="ro", basefmt="black")
plt.title("Frequency Spectrum of Signal B")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Plot Cross-Correlation
plt.figure(figsize=(8, 6))
plt.stem(lags, cross_corr, linefmt="green", markerfmt="go", basefmt="black")
plt.title("Cross-Correlation Function (Discrete)")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.grid()
plt.show()

# Plot Cross-Correlation using built-in function
plt.figure(figsize=(8, 6))
plt.stem(lags, cross_corr2, linefmt="green", markerfmt="go", basefmt="black")
plt.title("Cross-Correlation Function (Built-in)")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.grid()
plt.show()
