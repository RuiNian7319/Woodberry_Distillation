import numpy as np
import matplotlib.pyplot as plt

# Load Data
transfer_function = np.loadtxt('step_test_tf.csv', delimiter=',')
matlab_state_space = np.loadtxt('step_test_ss.csv', delimiter=',')
python_state_space = np.loadtxt('step_test_python.csv')

"""
Math Plotting Library settings
"""

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

# Distillate Trajectory

plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(1, 2, 1)
plt.title(r"$X_D$ Trajectory")
plt.ylabel(r"\%MeOH in Distillate, \textit{$X_D$} (\%)")
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(transfer_function[:, 0], label='Transfer Function')
plt.plot(matlab_state_space[:, 0], label='State Space (MATLAB)')
plt.plot(python_state_space[6:, 0], label='State Space (Python)')

plt.xlim([0, 150])

plt.legend(loc=0, prop={'size': 10}, frameon=False)

# Bottoms Trajectory

plt.subplot(1, 2, 2)
plt.title(r"$X_B$ Trajectory")
plt.ylabel(r"\%MeOH in Bottoms, \textit{$X_B$} (\%)")
plt.xlabel(r"Time, \textit{T} (steps)")

plt.plot(transfer_function[:, 1], label='Transfer Function')
plt.plot(matlab_state_space[:, 1], label='State Space (MATLAB)')
plt.plot(python_state_space[6:, 1], label='State Space (Python)')

plt.xlim([0, 150])

plt.legend(loc=0, prop={'size': 10}, frameon=False)

plt.show()
