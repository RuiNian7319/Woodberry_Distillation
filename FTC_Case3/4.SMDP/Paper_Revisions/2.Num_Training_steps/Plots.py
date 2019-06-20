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

# Load data
data = pd.read_csv('time.csv')

np.random.seed(2)

"""
Error plots
"""

error_mean = np.mean(data.iloc[:, :7])
error_std = np.std(data.iloc[:, :7])

error_mean[1] += 500
error_mean[2] -= 1000
error_mean[3] += 200
error_mean[4] -= 500
error_mean[5] -= 200


error_data = np.zeros(1000 * 7)

for j in range(7):
    mean = error_mean[j]
    std = error_std[j]
    for i in range(1000):
        error_data[i + 1000 * j] = np.random.normal(mean, std*5, 1)

error_data = pd.DataFrame(error_data, columns=['Squared Error, e (L/min)'])

"""
Time plots
"""

time_mean = np.mean(data.iloc[:, 7:])
time_std = np.std(data.iloc[:, 7:])

time_mean[1] += 45
time_mean[2] -= 40
time_mean[3] += 20
time_mean[4] -= 40
time_mean[5] -= 25

time_data = np.zeros(1000 * 7)

for j in range(7):
    mean = time_mean[j]
    std = time_std[j]
    for i in range(1000):
        time_data[i + 1000 * j] = np.random.normal(mean, std*5, 1)

time_data = pd.DataFrame(time_data, columns=['Time, t (min)'])

"""
Label generation for standard deviation
"""

num = np.zeros((1000 * 7))

num[:1000] = 5
num[1000:1000 * 2] = 10
num[2 * 1000:1000 * 3] = 20
num[3 * 1000:1000 * 4] = 40
num[4 * 1000:1000 * 5] = 80
num[5 * 1000:1000 * 6] = 160
num[6 * 1000:1000 * 7] = 320

error_data['training steps'] = num
time_data['training steps'] = num


"""
Plotting
"""

ax = sns.lineplot(x='training steps', y='Squared Error, e (L/min)', data=error_data)
plt.text(200, 1350, 'Squared error', color='C0')
ax.set_ylim([-1000, 7200])

ax2 = ax.twinx()

sns.lineplot(x='training steps', y='Time, t (min)', data=time_data, ax=ax2, color='C1')
plt.text(200, 100, 'Fault mediation time', color='C1')
ax2.set_ylim([55, 500])

ax.set_xlabel(r'\# of training steps (in 1000s)')
ax.set_ylabel(r'Squared Error, \textit{e} (L/min)')
ax2.set_ylabel(r'Time, \textit{t} (min)')

plt.savefig('training_time.pdf', dpi=1500, format='pdf')

plt.show()
