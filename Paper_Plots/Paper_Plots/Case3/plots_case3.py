import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

ftc_noiseless = np.loadtxt('ftc_noiseless_case3.csv')
ftc_noise = np.loadtxt('ftc_noise_case3.csv')
no_ftc_noiseless = np.loadtxt('no_ftc_noiseless_case3.csv')

x = np.linspace(0, ftc_noiseless.shape[0], ftc_noiseless.shape[0] - 50)

with sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True):
    ax = plt.subplot(111)
    ax.plot(x, no_ftc_noiseless[50:, 0], label=r'$X_D$ No FTC (Noiseless)', linestyle='-.')
    ax.plot(x, ftc_noise[50:, 0], label=r'$X_D$ FTC', linestyle='--')
    ax.plot(x, ftc_noiseless[50:, 0], label=r'$X_D$ FTC (Noiseless)')

    ax.plot(x, no_ftc_noiseless[50:, 1], linestyle='-.', label=r'$X_B$ No FTC (Noiseless)')
    ax.plot(x, ftc_noise[50:, 1], linestyle='--', label=r'$X_B$ FTC')
    ax.plot(x, ftc_noiseless[50:, 1], label=r'$X_B$ FTC (Noiseless)')

plt.xlabel(r'Time, \textit{t} (min)')
plt.ylabel(r'\%MeOH, $\textit{X}_D$ (wt. \%)')


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


handles, labels = ax.get_legend_handles_labels()
plt.legend(flip(handles, 2), flip(labels, 2), loc=6, ncol=2, prop={'size': 12}, frameon=False)

plt.savefig('Case3_Plot.eps', format='eps', dpi=1000)

plt.show()
