import numpy as np
import time

# Глобальні змінні для підрахунку операцій
mul_count = 0
add_count = 0

# --- Реалізація ШПФ ---
def fft_recursive(x):
    global mul_count, add_count
    N = len(x)
    if N <= 1:
        return x
    else:
        # Рекурсивні виклики для парних і непарних індексів
        X_even = fft_recursive(x[::2])
        X_odd = fft_recursive(x[1::2])
        
        # Обчислення коренів з одиниці
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        mul_count += N  # Множення для факторів

        X = np.zeros(N, dtype=complex)
        half_N = N // 2
        for k in range(half_N):
            temp = factor[k] * X_odd[k]
            mul_count += 1  # Множення factor[k] * X_odd[k]
            X[k] = X_even[k] + temp
            X[k + half_N] = X_even[k] - temp
            add_count += 2  # Додавання та віднімання
        return X

# Функція для доповнення вхідного сигналу до наступного степеня 2
def pad_to_power_of_two(x):
    N = len(x)
    next_power_of_two = 1 << (N - 1).bit_length()
    if N != next_power_of_two:
        padded_x = np.zeros(next_power_of_two)
        padded_x[:N] = x
        return padded_x
    else:
        return x

# --- Головна функція ---
def main():
    N = 12  # Початкова довжина даних
    f = np.array([0.35778841, 0.47126741, 0.99193968, 0.48107101, 0.16193039, 0.30178352,
                  0.10909715, 0.62862941, 0.0270227,  0.80352422, 0.20834662, 0.33267782])
    # Використовуємо ті самі дані з лабораторної роботи №2

    # Доповнюємо вхідний сигнал до наступного степеня 2
    f_padded = pad_to_power_of_two(f)
    N_padded = len(f_padded)

    # --- Обчислення ШПФ ---
    global mul_count, add_count
    mul_count = 0
    add_count = 0
    start_time = time.perf_counter()
    X_fft = fft_recursive(f_padded)
    time_fft = time.perf_counter() - start_time
    mul_count_fft = mul_count
    add_count_fft = add_count

    print("ШПФ:")
    print(f"Час обчислення: {time_fft:.10f} секунд")
    print(f"Кількість множень: {mul_count_fft}")
    print(f"Кількість додавань: {add_count_fft}")

    # --- Обчислення спектру амплітуд та фаз ---
    amplitudes = np.abs(X_fft)
    phases = np.angle(X_fft)

    # --- Побудова графіків ---
    import matplotlib.pyplot as plt

    k_values = np.arange(N_padded)

    plt.figure(figsize=(12, 6))
    plt.stem(k_values, amplitudes)
    plt.title('Спектр амплітуд (ШПФ)')
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.stem(k_values, phases)
    plt.title('Спектр фаз (ШПФ)')
    plt.xlabel('k')
    plt.ylabel('Фаза X(k)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
