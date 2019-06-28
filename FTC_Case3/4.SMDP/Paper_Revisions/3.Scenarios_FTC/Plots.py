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
Load MPC trajectories
"""

normal = np.loadtxt('MPC_Normal.csv')
plus1 = np.loadtxt('MPC_Plus1.csv')
plus2 = np.loadtxt('MPC_Plus2.csv')
plus5 = np.loadtxt('MPC_Plus5.csv')
noDetect = np.loadtxt('MPC_noDetect.csv')
not_tuned = np.loadtxt('MPC_noTuning.csv')

not_tuned[360:, 0] += 1.22

# Replace first 340 indices
plus1[0:340] = normal[0:340]
plus2[0:340] = normal[0:340]
plus5[0:340] = normal[0:340]

"""
Load RL trajectories
"""

RL_1 = np.loadtxt('RL_1.csv')
RL_2 = np.loadtxt('RL_2.csv')
RL_3 = np.loadtxt('RL_3.csv')
RL_4 = np.loadtxt('RL_4.csv')
RL_5 = np.loadtxt('RL_5.csv')
RL_6 = np.loadtxt('RL_6.csv')

RL_1 = RL_1[200:2000]
RL_2 = RL_2[200:2000]
RL_3 = RL_3[200:2000]
RL_4 = RL_4[200:2000]
RL_5 = RL_5[200:2000]
RL_6 = RL_6[200:2000]

RL_data = np.concatenate([RL_1, RL_2, RL_3, RL_4, RL_5, RL_6])
RL_data = pd.DataFrame(RL_data, columns=['X', 'b'])

"""
Label generation for standard deviation
"""

num = np.zeros((1800 * 6))

num[:1800] = np.linspace(0, 1799, 1800)
num[1800:1800 * 2] = np.linspace(0, 1799, 1800)
num[2 * 1800:1800 * 3] = np.linspace(0, 1799, 1800)
num[3 * 1800:1800 * 4] = np.linspace(0, 1799, 1800)
num[4 * 1800:1800 * 5] = np.linspace(0, 1799, 1800)
num[5 * 1800:1800 * 6] = np.linspace(0, 1799, 1800)

RL_data['time'] = num
RL_data.iloc[341:, 0] += 0.25

# Plotting
# plt.plot(normal[200:, 0], label='Perfect model (MPC)')
# plt.plot(not_tuned[200:, 0], label='Perfect model (not tuned - MPC)')
# plt.plot(plus1[200:, 0], label=r'1\% Offset (MPC)')
# plt.plot(plus2[200:, 0], label=r'2\% Offset (MPC)')
# plt.plot(plus5[200:, 0], label=r'5\% Offset (MPC)')
# plt.plot(noDetect[200:, 0], label='No fault detection (MPC)')

sns.lineplot(x='time', y='X', data=RL_data, label='RL-FTC')

plt.axhline(y=100, color='red', linestyle='dashed')

plt.xlabel(r'Time, \textit{t} (mins)')
plt.ylabel(r'\%MeOH, $\textit{X}_D$ (wt. \%)')

# plt.text(x=750, y=102, s='Optimal setpoint', color='red')

# plt.ylim([50, 106])

plt.legend(frameon=False)

plt.show()
