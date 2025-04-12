import numpy as np
import matplotlib.pyplot as plt

# the functions
def parabolic(x):
    return np.where((x >= -2) & (x <= 2), x ** 2, 0)

def triangular(x):
    return np.where((x >= -2) & (x <= 2), (2-abs(x))/2,0 )

def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), x+2, 0)

def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)

# original function
def plot_original_function(x_values, y_values,label):
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, y_values, label = label)
    plt.title("Original Function (" + label + ")")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# fourier transform
def plot_fourier_transform_function(frequencies,ft_data,label):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
    plt.title("Frequency Spectrum of " + label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()

def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    for i in range(num_freqs):
        ft_result_real[i] = np.trapezoid(signal * (np.cos(2*np.pi*frequencies[i]*sampled_times)), sampled_times)
        ft_result_imag[i] = -1 * np.trapezoid(signal * (np.sin(2*np.pi*frequencies[i]*sampled_times)), sampled_times)
    
    return ft_result_real, ft_result_imag

# inverse fourier transform
def plot_inverse_fourier_transform_function(x_values, y_values, sampled_times, reconstructed_y_values, label):
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
    plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed " + label, color="red", linestyle="--")
    plt.title("Original vs Reconstructed Function (" + label + ")")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()    

def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    for i in range(n):
        reconstructed_signal[i] = np.trapezoid(ft_signal[0]*np.cos(2*np.pi*frequencies*sampled_times[i]) - ft_signal[1]*np.sin(2*np.pi*frequencies*sampled_times[i]), frequencies)
    
    return reconstructed_signal


# main
def run(func,label) :
    # plotting original function from range -10 to 10
    x_values = np.linspace(-5, 5, 1000)
    y_values = func(x_values)
    plot_original_function(x_values, y_values, label)

    # plotting fruequency spectrum from range -1 to 1
    sampled_times = x_values
    frequencies1 = np.linspace(-1, 1, 1000)
    ft_data1 = fourier_transform(y_values, frequencies1, sampled_times)
    plot_fourier_transform_function(frequencies1, ft_data1, label)

    # plotting frequency spectrum from range -2 to 2
    frequencies2 = np.linspace(-2, 2, 1000)
    ft_data2 = fourier_transform(y_values, frequencies2, sampled_times)
    plot_fourier_transform_function(frequencies2, ft_data2, label)

    # plotting frequency spectrum from range -5 to 5
    frequencies3 = np.linspace(-5, 5, 1000)
    ft_data3 = fourier_transform(y_values, frequencies3, sampled_times)
    plot_fourier_transform_function(frequencies3, ft_data3, label)

    # reconstructed signal for range -1 to 1
    reconstructed_y_values = inverse_fourier_transform(ft_data1, frequencies1, sampled_times)
    plot_inverse_fourier_transform_function(x_values, y_values, sampled_times, reconstructed_y_values, label)

    # reconstructed signal for range -2 to 2
    reconstructed_y_values = inverse_fourier_transform(ft_data2, frequencies2, sampled_times)
    plot_inverse_fourier_transform_function(x_values, y_values, sampled_times, reconstructed_y_values, label)

    # reconstructed signal for range -5 to 5
    reconstructed_y_values = inverse_fourier_transform(ft_data3, frequencies3, sampled_times)
    plot_inverse_fourier_transform_function(x_values, y_values, sampled_times, reconstructed_y_values, label)



run(parabolic,"(parabolic) y = x^2")
run(triangular,"(triangular) y = (2-abs(x))/2")
run(sawtooth_function,"(sawtooth) y = x+2")
run(rectangular_function,"(rectangular) y = 1")