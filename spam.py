import numpy as np
import matplotlib.pyplot as plt

# Define parameters
omega_0 = 1.0  # Center frequency
delta_omega_a = 1.0  # Homogeneous broadening width
ratios = [1/20, 1, 20]  # Different values of Δω_d / Δω_a

# Define frequency range for plotting (normalized)
omega_range = np.linspace(-5, 5, 1000)

# Define functions for χ' and χ''
def susceptibility_real(omega, delta_omega_d, delta_omega_a):
    term1 = np.log(1 + (delta_omega_d / (2 * delta_omega_a))**2)
    return (delta_omega_d / delta_omega_a) * term1 / ((omega - omega_0)**2 + delta_omega_a**2)

def susceptibility_imag(omega, delta_omega_d, delta_omega_a):
    term2 = np.arctan(delta_omega_d / (2 * delta_omega_a))
    return (delta_omega_d / delta_omega_a) * term2 / ((omega - omega_0)**2 + delta_omega_a**2)

# Create plots for different values of Δω_d / Δω_a
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot χ'(ω)
for ratio in ratios:
    delta_omega_d = ratio * delta_omega_a
    chi_real = susceptibility_real(omega_range, delta_omega_d, delta_omega_a)
    axs[0].plot(omega_range, chi_real, label=f'Δω_d / Δω_a = {ratio}')

axs[0].set_title(r"$\chi'(\omega)$ (Real Part)")
axs[0].set_xlabel(r"$(\omega - \omega_0) / \Delta \omega_a$")
axs[0].set_ylabel(r"$\chi'(\omega)$")
axs[0].legend()
axs[0].grid(True)

# Plot χ''(ω)
for ratio in ratios:
    delta_omega_d = ratio * delta_omega_a
    chi_imag = susceptibility_imag(omega_range, delta_omega_d, delta_omega_a)
    axs[1].plot(omega_range, chi_imag, label=f'Δω_d / Δω_a = {ratio}')

axs[1].set_title(r"$\chi''(\omega)$ (Imaginary Part)")
axs[1].set_xlabel(r"$(\omega - \omega_0) / \Delta \omega_a$")
axs[1].set_ylabel(r"$\chi''(\omega)$")
axs[1].legend()
axs[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()
