import numpy as np
import matplotlib.pyplot as plt
import time

# Global variables to count operations
multiplications = 0
additions = 0

def compute_dft_term(f_n, k, N):
    global multiplications, additions
    Ak = 0.0
    Bk = 0.0
    for n in range(N):
        angle = 2 * np.pi * k * n / N
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        Ak += f_n[n] * cos_val
        Bk += f_n[n] * sin_val
        # Counting operations
        multiplications += 4  # f_n[n]*cos_val, f_n[n]*sin_val, k*n, 2*pi*k*n/N
        additions += 2  # Ak += ..., Bk += ...
    Ck = Ak - 1j * Bk
    return Ck

def compute_dft(f_n):
    N = len(f_n)
    Ck = []
    start_time = time.time()
    for k in range(N):
        Ck.append(compute_dft_term(f_n, k, N))
    computation_time = time.time() - start_time
    return np.array(Ck), computation_time

def plot_spectra(Ck):
    N = len(Ck)
    k = np.arange(N)
    amplitude = np.abs(Ck)
    phase = np.angle(Ck)

    # Verify lengths
    print("Length of k:", len(k))
    print("Length of amplitude:", len(amplitude))
    print("Length of phase:", len(phase))

    # Check for NaN or Inf values
    if np.isnan(amplitude).any() or np.isinf(amplitude).any():
        print("Amplitude contains NaN or Inf values.")
    if np.isnan(phase).any() or np.isinf(phase).any():
        print("Phase contains NaN or Inf values.")

    # Plot amplitude spectrum
    plt.figure(figsize=(12, 6))
    plt.stem(k, amplitude)
    plt.title('Amplitude Spectrum')
    plt.xlabel('k')
    plt.ylabel('|Ck|')
    plt.grid(True)
    plt.show()

    # Plot phase spectrum
    plt.figure(figsize=(12, 6))
    plt.stem(k, phase)
    plt.title('Phase Spectrum')
    plt.xlabel('k')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.show()

def main():
    global multiplications, additions

    # Student number n (for N = 10 + n)
    n = 2  # Replace 2 with your student number
    N = 10 + n

    # Generate arbitrary input vector f_n
    f_n = [0.35778841, 0.47126741, 0.99193968 ,0.48107101 ,0.16193039 ,0.30178352,
 0.10909715 ,0.62862941 ,0.0270227 , 0.80352422 ,0.20834662 ,0.33267782]  # Random values between 0 and 1

    print(f"Input vector f_n (length {N}):")
    print(f_n)

    # Reset operation counters
    multiplications = 0
    additions = 0

    # Compute DFT coefficients
    Ck, computation_time = compute_dft(f_n)

    # Display computation time and number of operations
    print(f"\nComputation Time: {computation_time:.6f} seconds")
    print(f"Total Multiplications: {multiplications}")
    print(f"Total Additions: {additions}")

    # Display DFT coefficients
    print("\nDFT Coefficients Ck:")
    for k in range(N):
        print(f"C_{k} = {Ck[k]}")

    # Plot amplitude and phase spectra
    plot_spectra(Ck)

if __name__ == "__main__":
    main()
