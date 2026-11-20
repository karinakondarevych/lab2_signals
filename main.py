import time
import matplotlib.pyplot as plt
import numpy as np

n = 5

print("\n--- ЧАСТИНА І ---")
N_1 = 10 + n
np.random.seed(42)
f_signal = np.random.uniform(-0.2, 0.2, N_1)
print(f"Вхідний вектор сигналу f (N = {N_1}):\n", np.round(f_signal, 3))

def compute_k_trig_term(f, k, N): # підпрограма обчислення одного (k-го) члена ряду Фур’є для ДПФ у тригонометричній формі
  mult_count = 0
  add_count = 0
  Ak_part = 0.0
  Bk_part = 0.0

  for i in range(N):
    angle = (2 * np.pi * k * i) / N
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)

    Ak_part += f[i] * cos_val
    Bk_part -= f[i] * sin_val

    mult_count += 2  # множення f[i] на cos та sin
    add_count += 2  # додавання/накопичення сум

  return Ak_part, Bk_part, mult_count, add_count

def compute_fourier_coefficient_k(f, k, N): # підпрограма обчислення коефіцієнта Фур'є C_k = A_k + j * B_k як суми ряду з N членів
  Ak, Bk, mults, adds = compute_k_trig_term(f, k, N)
  C_k = complex(Ak, Bk)
  return C_k, Ak, Bk, mults, adds

def compute_full_dft_trig(f):
  N = len(f)
  C = np.zeros(N, dtype=complex)
  A = np.zeros(N)
  B = np.zeros(N)
  total_mult = 0
  total_add = 0

  start_time = time.perf_counter() # оцінка часу обчислення
  for k in range(N):
    Ck, Ak, Bk, m_ops, a_ops = compute_fourier_coefficient_k(f, k, N)
    C[k] = Ck
    A[k] = Ak
    B[k] = Bk
    total_mult += m_ops
    total_add += a_ops
  end_time = time.perf_counter()
  exec_time = end_time - start_time
  return C, A, B, exec_time, total_mult, total_add

C_coeffs, A_coeffs, B_coeffs, calc_time, num_mult, num_add = ( # виконання розрахунків Частини І
    compute_full_dft_trig(f_signal)
)

print("\nЗначення коефіцієнтів Фур'є та їх модулі:")
for k in range(N_1):
  real_p = A_coeffs[k]
  imag_p = B_coeffs[k]
  if abs(real_p) < 1e-10:
    real_p = 0.0
  if abs(imag_p) < 1e-10:
    imag_p = 0.0

  sign = "-" if imag_p < 0 else "+"
  mod_c = np.abs(C_coeffs[k])  # Обчислення модуля коефіцієнта C_k

  print(
      f"   k = {k:2d}: C_{k} = {real_p:7.3f} {sign} j{abs(imag_p):5.3f}"
      f"   |   Модуль |C_{k}| = {mod_c:6.3f}"
  )

print(f"Час обчислення ДПФ: {calc_time:.6f} секунд")
print(f"Кількість операцій множення: {num_mult}")
print(f"Кількість операцій додавання: {num_add}")

# побудова графіків спектру амплітуд та фаз для N
amplitudes = np.abs(C_coeffs)
phases = np.angle(C_coeffs)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.stem(range(N_1), amplitudes, basefmt=" ")
plt.title(f"Спектр амплітуд |C_k| (N = {N_1})")
plt.xlabel("Індекс k")
plt.ylabel("Амплітуда")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(range(N_1), phases, basefmt=" ")
plt.title(f"Спектр фаз arg(C_k) (N = {N_1})")
plt.xlabel("Індекс k")
plt.ylabel("Фаза (рад)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- ЧАСТИНА ІІ ---")
s_samples = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=float)
N_8 = len(s_samples)
print(f"Згенеровані 8 рівновіддалених відліків s(nT_δ): {s_samples}")

def compute_dft_complex_coefficients(signal):
  N = len(signal)
  C = np.zeros(N, dtype=complex)
  for k in range(N):
    sum_val = 0.0
    for i in range(N):
      angle = -2 * np.pi * k * i / N
      sum_val += signal[i] * np.exp(1j * angle)
    C[k] = sum_val
  return C


C_n = compute_dft_complex_coefficients(s_samples)
mod_C_n = np.abs(C_n)
arg_C_n = np.angle(C_n)

print("\nКомплексні коефіцієнти C_n, їх модулі та аргументи:")
for i in range(N_8):
  real_p = C_n[i].real
  imag_p = C_n[i].imag
  if abs(real_p) < 1e-10:
    real_p = 0.0
  if abs(imag_p) < 1e-10:
    imag_p = 0.0
  sign = "-" if imag_p < 0 else "+"
  print(
      f"   C_{i} = {real_p:.3f} {sign} j{abs(imag_p):.3f} | |C_{i}| ="
      f" {mod_C_n[i]:.3f} | arg(C_{i}) = {arg_C_n[i]:.3f} рад"
  )

T_d = 1.0
T_c = N_8 * T_d
t_fine = np.linspace(-T_c, T_c, 800)


def reconstruct_analog_signal_formula6(t_values, C_coeffs, T_interval):
  N = len(C_coeffs)
  reconstructed = np.zeros_like(t_values, dtype=float)
  for idx, t in enumerate(t_values):
    val = 0.0
    for k in range(N):
      term = C_coeffs[k] * np.exp(1j * 2 * np.pi * k * np.abs(t) / T_interval)
      val += term.real
    reconstructed[idx] = val / N
  return reconstructed


s_reconstructed = reconstruct_analog_signal_formula6(t_fine, C_n, T_c)


# Побудова симетричного графіка
plt.figure(figsize=(10, 4))
plt.plot(
    t_fine,
    s_reconstructed,
    label="Відтворений аналоговий сигнал s(t)",
    color="blue",
)

# Формуємо симетричні відліки для відображення в обидва боки від 0
t_samples_sym = np.arange(
    -N_8 + 1, N_8
) * T_d
s_samples_sym = np.concatenate([s_samples[1:][::-1], s_samples])

# Виводимо відліки за допомогою plt.stem
plt.stem(
    t_samples_sym,
    s_samples_sym,
    linefmt="r-",
    markerfmt="ro",
    basefmt=" ",
    label="Дискретні відліки s(nT_δ)",
)

plt.title("Відновлення аналогового сигналу за відліками")
plt.xlabel("Час t")
plt.ylabel("s(t)")

# Межі осей
max_val = max(np.max(np.abs(s_reconstructed)), np.max(np.abs(s_samples)))
plt.ylim(-max_val * 0.2, max_val * 1.3)
plt.xlim(-T_c * 1.1, T_c * 1.1)

plt.legend()
plt.grid(True)
plt.show()

print("\n--- ЧАСТИНА ІІІ ---")

def compute_idft_samples(C):
  N = len(C)
  s_recovered = np.zeros(N, dtype=float)
  for n in range(N):
    sum_val = 0.0
    for k in range(N):
      angle = 2 * np.pi * k * n / N
      sum_val += C[k] * np.exp(1j * angle)
    val = (sum_val / N).real
    if abs(val) < 1e-10:
      val = 0.0
    s_recovered[n] = val
  return s_recovered


recovered_samples = compute_idft_samples(C_n)
print("Відновлені відліки дискретизованого сигналу s(nT_δ) за ОДПФ:")
for idx, val in enumerate(recovered_samples):
  print(f"   s({idx} * T_δ) = {val:.5f}")

N_val = len(C_n)
terms_s0 = []
terms_s1 = []

for k in range(N_val):
  Ck = C_n[k]
  r_val = Ck.real
  i_val = Ck.imag
  if abs(r_val) < 1e-10:
    r_val = 0.0
  if abs(i_val) < 1e-10:
    i_val = 0.0
  sign = "-" if i_val < 0 else "+"

  terms_s0.append(f"      ({r_val:+.3f} {sign} j{abs(i_val):.3f}) · 1")

  angle_term = 2 * np.pi * k * 1 / N_val
  exp_val = np.exp(1j * angle_term)
  e_r = exp_val.real
  e_i = exp_val.imag
  if abs(e_r) < 1e-10:
    e_r = 0.0
  if abs(e_i) < 1e-10:
    e_i = 0.0
  e_sign = "-" if e_i < 0 else "+"
  terms_s1.append(
      f"      ({r_val:+.3f} {sign} j{abs(i_val):.3f}) · ({e_r:+.3f} {e_sign}"
      f" j{abs(e_i):.3f})"
  )

print("\nАналітичні вирази відліків:")
print(f"   s(0) = (1 / {N_val}) * [")
print("\n".join(terms_s0))
print(f"   ] = {recovered_samples[0]:.4f}\n")

print(f"   s(T_δ) = (1 / {N_val}) * [")
print("\n".join(terms_s1))
print(f"   ] = {recovered_samples[1]:.4f}\n")

print(
    f"   Числові значення: s(0) = {recovered_samples[0]:.4f}, s(T_δ) ="
    f" {recovered_samples[1]:.4f}"
)