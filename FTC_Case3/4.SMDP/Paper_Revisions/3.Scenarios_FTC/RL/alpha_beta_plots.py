import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set()
sns.set_style('white')

# Plotting formats
fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

np.random.seed(2)

"""
Learning rate plots
"""

# Load data
alpha_001 = np.linspace(0.01, 0.01, 10)
alpha_002 = np.linspace(0.02, 0.02, 7)
alpha_004 = np.linspace(0.04, 0.04, 6)
alpha_006 = np.linspace(0.06, 0.06, 10)
alpha_013 = np.linspace(0.125, 0.125, 10)
alpha_025 = np.linspace(0.25, 0.25, 9)
alpha_05 = np.linspace(0.5, 0.5, 7)
alpha_1 = np.linspace(1, 1, 8)

alphas = np.concatenate([alpha_001, alpha_002, alpha_004, alpha_006, alpha_013, alpha_025, alpha_05, alpha_1])

# RMSE
rmse = [568, 562, 530, 532, 618, 499, 597, 503, 566, 509,
        514, 543, 522, 678, 518, 595, 518,
        597, 583, 495, 588, 612, 518,
        608, 487, 567, 687, 734, 527, 627, 614, 535, 507,
        502, 599, 517, 495, 722, 658, 818, 600, 748, 614,
        803, 612, 474, 499, 681, 703, 545, 586, 638,
        620, 6400, 769, 615, 708, 1957, 5000,
        675, 22500, 671, 535, 920, 2705, 626, 552]
rmse = np.sqrt(rmse)

# Mediation Time
med_time = [41, 45, 43, 53, 52, 43, 42, 44, 48, 49,
            46, 40, 47, 61, 45, 48, 44,
            47, 44, 43, 45, 64, 42,
            47, 41, 44, 52, 51, 44, 48, 45, 46, 43,
            45, 45, 49, 43, 56, 48, 55, 48, 53, 45,
            73, 49, 92, 44, 59, 52, 56, 59, 82,
            46, 306, 63, 45, 55, 107, 288,
            61, 496, 50, 46, 94, 223, 50, 44]


# Plotting
df = pd.DataFrame(np.array([alphas, rmse, med_time]).T, columns=['Learning Rate', 'RMSE', 'Med_time'])
ax = sns.lineplot(x=r'Learning Rate', y='RMSE', data=df, color='C0', label=r'RMSE')
ax2 = ax.twinx()

sns.lineplot(x='Learning Rate', y='Med_time', data=df, ax=ax2, color='C1', label=r'Fault Mediation Time')

ax.set_xlabel(r'Learning Rate, \textit{$\alpha$}')
ax.set_ylabel(r'RMSE, \textit{e}')
ax2.set_ylabel(r'Fault Mediation Time, \textit{t} (min)')

ax.legend(loc='upper left', frameon=False)
ax2.legend(loc='upper center', frameon=False)

plt.savefig('LR_tuning.pdf', dpi=1500, format='pdf')

plt.show()

"""
Beta plots
"""
