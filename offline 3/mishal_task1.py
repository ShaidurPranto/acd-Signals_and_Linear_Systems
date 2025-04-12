import numpy as np
import matplotlib.pyplot as plt

# Define the interval and function and generate appropriate x values and y values
x_values = np.linspace(-10, 10, 1000)  # x values


# Define functions
def parabolic(x):
    return np.where((x >= -2) & (x <= 2), x ** 2, 0)


def triangular(x):
    return np.where((x >= -2) & (x <= 2), (2-abs(x))/2,0 )


def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), x+2, 0)


def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)


# Plot the original function, frequency spectrum and Fourier approximated function for different frequencies
def plot_function(equation, function):
    plt.figure(figsize=(12, 12))

    # Plot the original function
    plt.subplot(3, 2, 1)
    y_values = function(x_values)
    plt.plot(x_values, y_values, label="Original " + equation)
    plt.title("Original Function (" + equation + ")")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()

    # Frequency Spectrum and Reconstructed Signal for different frequencies
    freq_ranges = [1, 2, 3]  # Frequencies to consider for approximation
    for i, freq in enumerate(freq_ranges):
        frequencies = np.linspace(-freq, freq, 500)
        ft_data = fourier_transform(y_values, frequencies, x_values)

        # Frequency Spectrum Plot

        #  plot the FT data
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies, np.sqrt(ft_data[0] ** 2 + ft_data[1] ** 2))
        plt.title("Frequency Spectrum of y = x^2")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()


        # Reconstruct the signal from the FT data
        reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, x_values)

        plt.figure(figsize=(12, 4))
        plt.plot(x_values, y_values, label="Original y = x^2")
        plt.plot(x_values, reconstructed_y_values, label="Reconstructed y ")
        plt.title("Original vs Reconstructed Function (y = x^2)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()




# Fourier Transform
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)

    for i, f in enumerate(frequencies):
        # Compute the real and imaginary parts for this frequency
        cos_part = np.cos(2 * np.pi * f * sampled_times)  # Cosine term for real part
        sin_part = -1* np.sin(2 * np.pi * f * sampled_times)  # Sine term for imaginary part

        # Trapezoidal integration to calculate the real and imaginary components
        ft_result_real[i] = np.trapezoid(signal * cos_part, x=sampled_times)
        ft_result_imag[i] = np.trapezoid(signal * sin_part, x=sampled_times)

    return ft_result_real, ft_result_imag


# Inverse Fourier Transform
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)

    for i, t in enumerate(sampled_times):
        # Compute the real part of the inverse Fourier transform
        cos_part = np.cos(2 * np.pi * frequencies * t)
        sin_part = np.sin(2 * np.pi * frequencies * t)
        reconstructed_signal[i] = np.trapezoid(ft_signal[0] * cos_part - ft_signal[1] * sin_part, x=frequencies)

    return reconstructed_signal


# Call the plot function for all four functions
plot_function("Parabolic", parabolic)
plot_function("Triangular", triangular)
plot_function("Sawtooth", sawtooth_function)
plot_function("Rectangular", rectangular_function)