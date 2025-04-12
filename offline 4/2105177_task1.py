import numpy as np
import matplotlib.pyplot as plt

n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000


# generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    # noise_freqs = [15, 30, 45]  
    # amplitudes = [0.5, 0.3, 0.1]  
    # noise_freqs2 = [10, 20, 40] 
    # amplitudes2 = [0.3, 0.2, 0.1]

    # noise_freqs = [15, 30, 45, 60]
    # amplitudes = [0.5, 0.3, 0.1, 0.05]
    # noise_freqs2 = [10, 20, 40, 50]
    # amplitudes2 = [0.3, 0.2, 0.1, 0.05]

    # noise_freqs = [15, 30, 45, 60, 75]
    # amplitudes = [0.5, 0.3, 0.1, 0.05, 0.02]
    # noise_freqs2 = [10, 20, 40, 50, 70]
    # amplitudes2 = [0.3, 0.2, 0.1, 0.05, 0.02]

    # noise_freqs = [15, 30, 45, 60, 75, 90]
    # amplitudes = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01]
    # noise_freqs2 = [10, 20, 40, 50, 70, 80]
    # amplitudes2 = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    noise_freqs = [5, 10, 15, 20, 25, 30]  
    amplitudes = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]  
    noise_freqs2 = [6, 12, 18, 24, 30, 36]  
    amplitudes2 = [0.7, 0.5, 0.3, 0.2, 0.1, 0.05] 
    
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
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)

    return signal_A, signal_B

# discrete fourier transform
def DFT(signal):
    size = len(signal)
    fourier_transform_real = np.zeros(size)
    fourier_transform_imag = np.zeros(size)
    for i in range(size):
        fourier_transform_real += signal[i] * np.cos(2 * np.pi * i * samples / size)
        fourier_transform_imag -= signal[i] * np.sin(2 * np.pi * i * samples / size)

    return fourier_transform_real, fourier_transform_imag

# inverse discrete fourier transform
def IDFT(fourier_transform_real, fourier_transform_imag):
    size = len(fourier_transform_real)
    signal = np.zeros(size)
    for i in range(size):
        signal += fourier_transform_real[i] * np.cos(2 * np.pi * i * samples / size)
        signal -= fourier_transform_imag[i] * np.sin(2 * np.pi * i * samples / size)

    return signal/size

# cross correlation using DFT
def cross_correlation(signal_A, signal_B):
    fourier_transform_real_A, fourier_transform_imag_A = DFT(signal_A)
    fourier_transform_real_B, fourier_transform_imag_B = DFT(signal_B)

    cross_correlation_real = fourier_transform_real_A * fourier_transform_real_B + fourier_transform_imag_A * fourier_transform_imag_B
    cross_correlation_imag = fourier_transform_imag_A * fourier_transform_real_B - fourier_transform_real_A * fourier_transform_imag_B

    result = IDFT(cross_correlation_real, cross_correlation_imag)
    result = np.roll(result, n // 2)

    return result

# low pass filter
def low_pass_filter(signal,cut_off_freq):
    fourier_transform_real, fourier_transform_imag = DFT(signal)
    size = len(signal)
    for i in range(size):
        if i > cut_off_freq:
            fourier_transform_real[i] = 0
            fourier_transform_imag[i] = 0

    return IDFT(fourier_transform_real, fourier_transform_imag)

# high pass filter
def high_pass_filter(signal,cut_off_freq):
    fourier_transform_real, fourier_transform_imag = DFT(signal)
    size = len(signal)
    for i in range(size):
        if i < cut_off_freq:
            fourier_transform_real[i] = 0
            fourier_transform_imag[i] = 0

    return IDFT(fourier_transform_real, fourier_transform_imag)
    
# processing filtered signals
def process_signals(signal_A, signal_B):
    # low pass filter
    low_pass_filtered_signal_A = low_pass_filter(signal_A, 10)
    low_pass_filtered_signal_B = low_pass_filter(signal_B, 10)

    # high pass filter
    high_pass_filtered_signal_A = high_pass_filter(signal_A, 10)
    high_pass_filtered_signal_B = high_pass_filter(signal_B, 10)

    # cross correlation
    cross_correlation_low_pass = cross_correlation(low_pass_filtered_signal_A, low_pass_filtered_signal_B)
    cross_correlation_high_pass = cross_correlation(high_pass_filtered_signal_A, high_pass_filtered_signal_B)

    # plot the cross-correlation results
    lag_samples = np.arange(-n // 2, n // 2)
    plt.stem(lag_samples, cross_correlation_low_pass, linefmt='blue', markerfmt='bo', basefmt=' ', label='Low-pass Filtered')
    plt.legend()
    plt.title('Cross-correlation of Low-pass Filtered Signals')
    plt.xlabel('Lag(samples)')
    plt.ylabel('Correlation')
    plt.grid()
    plt.show()

    plt.stem(lag_samples, cross_correlation_high_pass, linefmt='red', markerfmt='ro', basefmt=' ', label='High-pass Filtered')
    plt.legend()
    plt.title('Cross-correlation of High-pass Filtered Signals')
    plt.xlabel('Lag(samples)')
    plt.ylabel('Correlation')
    plt.grid()
    plt.show()



# main function
def main():
    signal_A, signal_B = generate_signals()

    # Plot Signal A as a stem plot
    plt.stem(samples, signal_A, linefmt='blue', markerfmt='bo', basefmt=' ', label='Signal A')
    plt.legend()
    plt.title('Signal A')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    # Plot Signal B as a line plot
    plt.stem(samples, signal_B, linefmt='red', markerfmt='ro', basefmt=' ', label='Signal B')
    plt.legend()
    plt.title('Signal B')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

    # Compute the Discrete Fourier Transform (DFT) of Signal A
    fourier_transform_real_A, fourier_transform_imag_A = DFT(signal_A)
    # Compute the Discrete Fourier Transform (DFT) of Signal B
    fourier_transform_real_B, fourier_transform_imag_B = DFT(signal_B)

    # Plot the magnitude of the DFT of Signal A
    plt.stem(samples, np.sqrt(fourier_transform_real_A**2 + fourier_transform_imag_A**2), linefmt='blue', markerfmt='bo', basefmt=' ', label='Signal A')
    plt.legend()
    plt.title('Magnitude of the DFT of Signal A')
    plt.xlabel('Sample index')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()
    # Plot the magnitude of the DFT of Signal B
    plt.stem(samples, np.sqrt(fourier_transform_real_B**2 + fourier_transform_imag_B**2), linefmt='red', markerfmt='ro', basefmt=' ', label='Signal B')
    plt.legend()
    plt.title('Magnitude of the DFT of Signal B')
    plt.xlabel('Sample index')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

    # Compute the cross-correlation of Signal A and Signal B
    cross_correlation_result = cross_correlation(signal_B, signal_A)
    lag_samples = np.arange(-n // 2, n // 2)

    # Plot the cross-correlation result
    plt.stem(lag_samples, cross_correlation_result, linefmt='green', markerfmt='go', basefmt=' ', label='Cross-correlation')
    plt.legend()
    plt.title('Cross-correlation of Signal A and Signal B')
    plt.xlabel('Lag(samples)')
    plt.ylabel('Correlation')
    plt.grid()
    plt.show()

    # find the sample lag
    lag = np.argmax(cross_correlation_result) - n // 2
    print(f"Sample Lag: {lag}")

    # calculate distance
    distance = abs(lag) * wave_velocity / sampling_rate
    print(f"Distance: {distance} m")

    # process the signals
    process_signals(signal_A, signal_B)

main()