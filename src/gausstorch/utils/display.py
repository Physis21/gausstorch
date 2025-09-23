import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# Parameters to automatically size the figure with respect to an article/report.
# Here I set the text width of my thesis, but you should write in the textwidth of your own article/report.
# Same for font sizes.
TEXTWIDTH = 455.24411  # unit: pt
INCH_PER_PT = 1. / 72.27  # 72.27 points to an inch.
CM_PER_INCH = 1 / 2.54  # ratio cm/inch
CM_PER_PT = 1 / 28.346456692913
TEXTWIDTH_INCH = TEXTWIDTH * INCH_PER_PT  # 6.3 inches
TEXTWIDTH_CM = TEXTWIDTH * CM_PER_PT  # 16.06 cm

FS = 12  # generally used for axis labels
FS_S = 10  # generally used for tick labels


def setup_tex(usetex=True):
    matplotlib.rcParams.update({
        'text.usetex': usetex,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{braket}'
                               r'\usepackage{physics}',
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'cm',
        'mathtext.it': 'cm',
        'mathtext.bf': 'cm',
        'font.family': 'cambria',
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'figure.constrained_layout.use': True,
    })


# region Plotting dynamics

def plot_evolution_N(tspan: np.ndarray, means: np.ndarray, width_ratio=0.8,
                     xlabel="time (ns)", ylabel=r"$\boldsymbol{N}$", yscale='linear'):
    setup_tex()
    M = means.shape[1]
    width = width_ratio * TEXTWIDTH_INCH
    figsize = (width, (5.5 / 9) * width)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
    ax.plot(tspan, means,
            label=[fr"$\boldsymbol{{N_{{{i + 1}}}}}$" for i in range(M)],
            linewidth=1
            )
    # ax.plot(tspan * 100, means, label=[f'$N_{chr(65 + i)}$' for i in range(self.M)])  # when R = 1e7
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_yscale(yscale)
    if np.max(means) > 10 ** 4:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both', labelsize=FS_S)
    ax.grid(True)
    # fig.suptitle(rf"<$ N_i $> for constant drive, $N_{{\text{{osc}}}}$={M}", fontsize=FS)
    ax.legend(fontsize=FS_S)
    return fig, ax


def plot_evolution_fock(tspan: np.ndarray, probs: np.ndarray, labels: list, width_ratio=0.8,
                        xlabel="time (ns)", ylabel=r"$\boldsymbol{N}$", yscale='linear'):
    setup_tex()
    M = probs.shape[1]
    width = width_ratio * TEXTWIDTH_INCH
    figsize = (width, (5.5 / 9) * width)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
    ax.plot(tspan, probs, label=labels, linewidth=1)
    # ax.plot(tspan * 100, means, label=[f'$N_{chr(65 + i)}$' for i in range(self.M)])  # when R = 1e7
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_yscale(yscale)
    if np.max(probs) > 10 ** 4:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both', labelsize=FS_S)
    ax.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FS)
    return fig, ax


# endregion Plotting dynamics


# region Fock combination manipulation

def fock_state_to_str_general(fock_combination, mode_comb):
    prob_notation = [f"{str(fock_num)}_{str(osc)}" for fock_num, osc in zip(fock_combination, mode_comb)]
    return '$P(' + ''.join(prob_notation) + ')$'


def fock_states_to_str_list(fock_combs_per_mode_comb):
    fock_str_list = []
    for mode_comb, fock_combinations in fock_combs_per_mode_comb.items():
        for fock_combination in fock_combinations:
            fock_str_list.append(fock_state_to_str_general(fock_combination, mode_comb))
    return fock_str_list

# endregion Fock combination manipulation
