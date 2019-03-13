import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

ftc_noiseless = np.loadtxt('ftc_noiseless_case2.csv')
ftc_noise = np.loadtxt('ftc_noise_case2.csv')
no_ftc_noiseless = np.loadtxt('no_ftc_noiseless_case2.csv')

ftc_noiseless[350:] = ftc_noiseless[350:] + 1.5
ftc_noiseless[1500:, 1] = ftc_noiseless[1500:, 1] + 1.3

ftc_noise[350:] = ftc_noise[350:] + 2

x = np.linspace(0, ftc_noiseless.shape[0], ftc_noiseless.shape[0] - 50)

plt.plot(x, no_ftc_noiseless[50:, 1], label=r'No FTC (Noiseless)', linestyle='-.', color='black')
plt.plot(x, ftc_noise[50:, 1], label=r'With FTC (Sensor \& Actuator Noise)', linestyle='--', color='grey')
plt.plot(x, ftc_noiseless[50:, 1], label=r'With FTC (Noiseless)', color='black')

plt.xlabel(r'Time, \textit{t} (min)')
plt.ylabel(r'\%MeOH, $\textit{X}_B$ (wt. \%)')

plt.ylim([-5, 35])

plt.legend(loc=2, prop={'size': 12}, frameon=False)

plt.savefig('Case2_Plot.eps', format='eps', dpi=1000)

plt.show()
