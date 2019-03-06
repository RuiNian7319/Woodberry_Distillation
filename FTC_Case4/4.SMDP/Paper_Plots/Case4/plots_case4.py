import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

original = np.loadtxt('noiseless_case4.csv')
ep_100 = np.loadtxt('100ep_ftc_noise_case4.csv')
ep_300 = np.loadtxt('300ep_ftc_noise_case4.csv')

x = np.linspace(0, original.shape[0], original.shape[0] - 50)

with sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True):
    ax = plt.subplot(111)
    ax.plot(x, original[50:, 0], label=r'$X_D$ Original policy', linestyle='-.')
    ax.plot(x, ep_100[50:, 0], label=r'$X_D$ After 30K eval.', linestyle='--')
    ax.plot(x, ep_300[50:, 0], label=r'$X_D$ After 90K eval.')

    ax.plot(x, original[50:, 1], linestyle='-.', label=r'$X_B$ Original policy')
    ax.plot(x, ep_100[50:, 1], linestyle='--', label=r'$X_B$ After 30K eval.')
    ax.plot(x, ep_300[50:, 1], label=r'$X_B$ After 90K eval.')

plt.xlabel(r'Time, \textit{t} (min)')
plt.ylabel(r'\%MeOH, $\textit{X}$ (wt. \%)')

# plt.ylim([50, 105])


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


handles, labels = ax.get_legend_handles_labels()
plt.legend(flip(handles, 2), flip(labels, 2), loc=6, ncol=2, prop={'size': 12}, frameon=False)

plt.savefig('Case4_Plot.eps', format='eps', dpi=1000)

plt.show()

