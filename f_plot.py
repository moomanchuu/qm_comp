import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# Parameters
beta = 1.0/100  # Inverse temperature
E_diff = np.linspace(-10, 2, 100)  # Range of (E - E') values

# Correct the plotting with better handling for LaTeX in legend labels
# Define beta values again
# Define a finer range of beta values for a gradient effect
beta_values_gradient = np.linspace(0.1, 5.0, 4000)  # Smooth gradient from low to high beta

# Create the plot
plt.figure(figsize=(10, 6))
for beta in beta_values_gradient:
    f_EE_prime = np.minimum(1, np.exp(beta * E_diff))
    # Adjust the color mapping for the brighter two-thirds of "inferno"
    brighter_inferno = plt.cm.inferno(np.linspace(0.23, 1, 256))
    brighter_inferno_map = plt.matplotlib.colors.ListedColormap(brighter_inferno)
    plt.plot(E_diff, f_EE_prime, color=brighter_inferno_map((beta - 0.1) / 4.9), alpha=1.0)


# Add labels and title
plt.xlabel(r"$(E - E')$")
plt.xlim(-10, 1)
plt.ylabel(r"$f_{EE'}$")
# Create a new colormap that only includes the brighter half of "inferno"


# Use the modified colormap in your colorbar
plt.colorbar(plt.cm.ScalarMappable(cmap=brighter_inferno_map, norm=plt.Normalize(vmin=0.1, vmax=5.0)),
             label=r"$\beta$ values")

plt.grid(True, color='white')
plt.gca().set_facecolor('#000000')
plt.show()
