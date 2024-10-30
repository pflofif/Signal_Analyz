import numpy as np
import matplotlib.pyplot as plt

def main():
    # Step 1: Generate the binary signal s(nT_delta)
    n = 1  # Replace with your student number
    N = 96 + n
    T = 1.0  # Total time period (seconds)
    T_delta = T / N  # Sampling interval

    # Time vector for discrete samples
    n_values = np.arange(N)
    t_values = n_values * T_delta

    # Generate N random binary samples (0 or 1)
    np.random.seed(0)  # For reproducibility; remove or change the seed as needed
    s_n = np.random.randint(0, 2, N)
    print(f"Binary signal s(nT_delta) with N={N} samples:")
    print(s_n)

    # Step 2: Compute the DFT coefficients Cn
    Cn = np.fft.fft(s_n) / N  # Normalize by N
    # Frequencies corresponding to Cn
    freqs = np.fft.fftfreq(N, d=T_delta)

    # Step 3: Calculate magnitudes and phases
    magnitudes = np.abs(Cn)
    phases = np.angle(Cn)

    # Display magnitudes and phases
    print("\nDFT Coefficients Cn:")
    for i in range(N):
        print(f"C_{i} = {Cn[i]:.4f} (Magnitude: {magnitudes[i]:.4f}, Phase: {phases[i]:.4f} radians)")

    # Step 4: Reconstruct the analog signal s(t) using IDFT
    s_t_reconstructed = np.fft.ifft(Cn * N)  # Multiply by N to reverse the normalization

    # Step 5: Plot the reconstructed signal s(t)
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, s_t_reconstructed.real, label='Reconstructed Signal')
    plt.stem(t_values, s_n, linefmt='r-', markerfmt='ro', basefmt='k-', label='Original Samples')
    plt.title('Reconstructed Analog Signal s(t)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Plot magnitudes and phases
    # Plot magnitude spectrum
    plt.figure(figsize=(12, 6))
    plt.stem(freqs, magnitudes)
    plt.title('Magnitude Spectrum |Cn|')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

    # Plot phase spectrum
    plt.figure(figsize=(12, 6))
    plt.stem(freqs, phases)
    plt.title('Phase Spectrum arg(Cn)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
