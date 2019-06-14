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

data = pd.DataFrame(data, columns=['Time'])

num = np.zeros((1000 * 5))

num[:1000] = 0.5
num[1000:1000 * 2] = 1
num[2 * 1000:1000 * 3] = 2
num[3 * 1000:1000 * 4] = 4
num[4 * 1000:1000 * 5] = 8

data['Fault Magnitude'] = num

sns.lineplot(x='Fault Magnitude', y='Time', data=data)

plt.show()
