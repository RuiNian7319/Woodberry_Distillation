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

A = pd.read_csv('time_data/0.5.csv', names=['Time'])
B = pd.read_csv('time_data/1.csv', names=['Time'])
C = pd.read_csv('time_data/2.csv', names=['Time'])
D = pd.read_csv('time_data/4.csv', names=['Time'])
E = pd.read_csv('time_data/8.csv', names=['Time'])

"""
Normal plots
"""

A_mean = np.mean(A)
B_mean = np.mean(B)
C_mean = np.mean(C)
D_mean = np.mean(D)
E_mean = np.mean(E)

A_std = np.std(A)
B_std = np.std(B)
C_std = np.std(C)
D_std = np.std(D)
E_std = np.std(E)

A = np.zeros(1000)
B = np.zeros(1000)
C = np.zeros(1000)
D = np.zeros(1000)
E = np.zeros(1000)

aList = [A, B, C, D, E]

for i in range(len(A)):
    A[i] = np.random.normal(A_mean, A_std*10, 1)

for i in range(len(B)):
    B[i] = np.random.normal(B_mean, B_std*10, 1)

for i in range(len(C)):
    C[i] = np.random.normal(C_mean, C_std*10, 1)

for i in range(len(D)):
    D[i] = np.random.normal(D_mean, D_std*10, 1)

for i in range(len(E)):
    E[i] = np.random.normal(E_mean, E_std*10, 1)

data = np.concatenate([A, B, C, D, E]).reshape(-1, 1)

data = pd.DataFrame(data, columns=['Time, t (min)'])

num = np.zeros((1000 * 5))

num[:1000] = 0.5
num[1000:1000 * 2] = 1
num[2 * 1000:1000 * 3] = 2
num[3 * 1000:1000 * 4] = 4
num[4 * 1000:1000 * 5] = 8

data['Fault Magnitude, F (lb/min)'] = num

"""
Normalized Plots
"""

A_mean_norm = np.mean(A) / 1
B_mean_norm = np.mean(B) / 1
C_mean_norm = np.mean(C) / 2
D_mean_norm = np.mean(D) / 3
E_mean_norm = np.mean(E) / 5

A_std_norm = np.std(A) / 1
B_std_norm = np.std(B) / 1
C_std_norm = np.std(C) / 2
D_std_norm = np.std(D) / 3
E_std_norm = np.std(E) / 5

A_norm = np.zeros(1000)
B_norm = np.zeros(1000)
C_norm = np.zeros(1000)
D_norm = np.zeros(1000)
E_norm = np.zeros(1000)

aList = [A_norm, B_norm, C_norm, D_norm, E_norm]

for i in range(len(A_norm)):
    A_norm[i] = np.random.normal(A_mean_norm, A_std_norm, 1)

for i in range(len(B_norm)):
    B_norm[i] = np.random.normal(B_mean_norm, B_std_norm, 1)

for i in range(len(C_norm)):
    C_norm[i] = np.random.normal(C_mean_norm, C_std_norm, 1)

for i in range(len(D_norm)):
    D_norm[i] = np.random.normal(D_mean_norm, D_std_norm, 1)

for i in range(len(E_norm)):
    E_norm[i] = np.random.normal(E_mean_norm, E_std_norm, 1)

data_norm = np.concatenate([A_norm, B_norm, C_norm, D_norm, E_norm]).reshape(-1, 1)

data_norm = pd.DataFrame(data_norm, columns=['Time, t (min)'])

data_norm['Fault Magnitude, F (lb/min)'] = num

"""
Plotting
"""

sns.lineplot(x='Fault Magnitude, F (lb/min)', y='Time, t (min)', data=data)
plt.text(5, 69.5, 'Original',
         color='C0')

sns.lineplot(x='Fault Magnitude, F (lb/min)', y='Time, t (min)', data=data_norm)
plt.text(5, 25.5, 'Normalized',color='C1')

plt.xlabel(r'Fault Magnitude, \textit{F} (lb/min)')
plt.ylabel(r'Time, \textit{t} (min)')

plt.savefig('time_to_mediate.pdf', dpi=1000, format='pdf')

plt.show()
