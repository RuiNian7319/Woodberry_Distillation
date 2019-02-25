import matplotlib.pyplot as plt
import numpy as np

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

# Ideal response
x_ideal = [3, 3.1, 3.2, 3.5, 4.5, 4.8, 4.9, 5, 5, 5, 5, 5]

# Inverse response
x_inv = [3, 1.3, 0.9, 1.3, 1.9, 3.2, 4.8, 5, 5, 5, 5, 5]

# Overshoot response
x_overshoot = [3, 3.5, 4.7, 6.5, 6.6, 6.4, 5.8, 5.3, 5.1, 5, 5, 5, 5, 5, 5]

# Oscillations response
x_osc = [3, 3.8, 4.4, 5, 5.4, 5.5, 5.4, 5, 4.6, 4.5, 4.6, 5, 5.4, 5.5, 5.4, 5]

plt.plot(np.linspace(0, 10, len(x_ideal)), x_ideal, label=r'Ideal', color='black')
plt.plot(np.linspace(0, 10, len(x_inv)), x_inv, label=r'Inverse Response', linestyle='--', color='black')
plt.plot(np.linspace(0, 10, len(x_overshoot)), x_overshoot, label=r'Overshoot', linestyle=':', color='black')
plt.plot(np.linspace(0, 10, len(x_osc)), x_osc, label=r'Oscillations', linestyle='-.', color='black')

plt.legend(loc='best', frameon=False)

plt.xlabel(r'Time, \textit{t} (s)')
plt.ylabel(r'Response')

plt.show()
