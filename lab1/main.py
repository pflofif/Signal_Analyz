import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import json

def f(x, n):
    """
    Exact analytical function f(x) = x^n * exp(-x^2 / n)
    """
    return x**n * np.exp(-x**2 / n)

def compute_a0(n):
    """
    Compute the a0 coefficient of the Fourier series.
    """
    integrand = lambda x: f(x, n)
    a0, _ = quad(integrand, -np.pi, np.pi)
    return (1/np.pi) * a0

def compute_ak(k, n):
    """
    Compute the ak coefficients of the Fourier series.
    """
    integrand = lambda x: f(x, n) * np.cos(k * x)
    ak, _ = quad(integrand, -np.pi, np.pi)
    return (1/np.pi) * ak

def compute_bk(k, n):
    """
    Compute the bk coefficients of the Fourier series.
    """
    integrand = lambda x: f(x, n) * np.sin(k * x)
    bk, _ = quad(integrand, -np.pi, np.pi)
    return (1/np.pi) * bk

def fourier_series(x, N, n, ak_values, bk_values):
    """
    Compute the Fourier series approximation up to order N.
    """
    result = ak_values[0] / 2  # a0 / 2
    for k in range(1, N + 1):
        result += ak_values[k] * np.cos(k * x) + bk_values[k - 1] * np.sin(k * x)
    return result

def plot_harmonics(N, ak_values, bk_values):
    """
    Plot the harmonics and the function in the frequency domain.
    """
    # Harmonic indices from 0 to N inclusive
    k_values = np.arange(0, N + 1)  # Length N + 1

    # ak_values includes a0 to aN (length N + 1)
    a_values = ak_values

    # bk_values includes b1 to bN (length N)
    # Prepend b0 = 0 to align with k_values
    b_values = np.concatenate(([0], bk_values))  # Length N + 1

    # Plot ak coefficients
    plt.figure(figsize=(12, 6))
    plt.stem(k_values, np.abs(a_values), linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title('Amplitude Spectrum of ak Coefficients')
    plt.xlabel('Harmonic k')
    plt.ylabel('|ak|')
    plt.grid(True)
    plt.show()

    # Plot bk coefficients
    plt.figure(figsize=(12, 6))
    plt.stem(k_values, np.abs(b_values), linefmt='g-', markerfmt='go', basefmt='r-')
    plt.title('Amplitude Spectrum of bk Coefficients')
    plt.xlabel('Harmonic k')
    plt.ylabel('|bk|')
    plt.grid(True)
    plt.show()

def compute_relative_error(actual_values, approx_values):
    """
    Compute the relative error of the approximation.
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.abs(actual_values - approx_values) / np.abs(actual_values)
        error[np.isnan(error)] = 0  # Set NaNs to zero
    return error

def save_results(N, ak_values, bk_values, max_error):
    """
    Save the results to a file.
    """
    results = {
        'Order N': N,
        'ak coefficients': ak_values.tolist(),  # Includes a0 to aN
        'bk coefficients': bk_values.tolist(),  # Includes b1 to bN
        'Maximum relative error': max_error
    }
    with open('fourier_series_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

def main():
    # Parameters
    n = int(input("Enter the value of n: "))
    N = int(input("Enter the order N of the Fourier series: "))
    x_values = np.linspace(-np.pi, np.pi, 1000)
    actual_values = f(x_values, n)

    # Compute coefficients
    a0 = compute_a0(n)
    ak_values = [a0]  # Start with a0
    bk_values = []

    for k in range(1, N + 1):
        ak_values.append(compute_ak(k, n))
        bk_values.append(compute_bk(k, n))

    ak_values = np.array(ak_values)  # Now length N + 1 (a0 to aN)
    bk_values = np.array(bk_values)  # Length N (b1 to bN)

    # Compute Fourier series approximation
    approx_values = np.array([fourier_series(x, N, n, ak_values, bk_values) for x in x_values])

    # Compute relative error
    error = compute_relative_error(actual_values, approx_values)
    max_error = np.max(error)

    # Plot harmonics
    plot_harmonics(N, ak_values, bk_values)

    # Save results
    save_results(N, ak_values, bk_values, max_error)

    # Plot actual vs approximation
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, actual_values, label='Actual Function f(x)')
    plt.plot(x_values, approx_values, label='Fourier Series Approximation', linestyle='--')
    plt.title('Actual Function vs Fourier Series Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot relative error
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, error)
    plt.title('Relative Error of Approximation')
    plt.xlabel('x')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.show()

    print(f"Maximum relative error: {max_error}")

if __name__ == "__main__":
    main()
