"""
Functions providing templates for human-readable display of data.
By default, the axis dimensions are written with my PhD thesis text width and font size in mind.
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch

from gausstorch.utils.operations import (
    truncate_alpha,
    truncate_sigma,
    moments_to_quad_moments,
    wigner_2d_map,
    keep_quads_in_alpha_r,
    keep_quads_in_sigma_r,
)

# Parameters to automatically size the figure with respect to an article/report.
# Here I set the text width of my thesis, but you should write in the textwidth of your own article/report.
# Same for font sizes.
TEXTWIDTH = 455.24411  # unit: pt
INCH_PER_PT = 1.0 / 72.27  # 72.27 points to an inch.
CM_PER_INCH = 1 / 2.54  # ratio cm/inch
CM_PER_PT = 1 / 28.346456692913
TEXTWIDTH_INCH = TEXTWIDTH * INCH_PER_PT  # 6.3 inches
TEXTWIDTH_CM = TEXTWIDTH * CM_PER_PT  # 16.06 cm

FS = 12  # generally used for axis labels
FS_S = 10  # generally used for tick labels


def setup_tex(usetex: bool = True) -> None:
    """Sets up `matplotlib.rcParams` to write tex code.

    Args:
        usetex (bool, optional): If True, add `text.usetex` to rcParams. Defaults to True.
    """
    matplotlib.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{braket}"
            r"\usepackage{physics}",
            "mathtext.fontset": "cm",
            "mathtext.rm": "cm",
            "mathtext.it": "cm",
            "mathtext.bf": "cm",
            "font.family": "cambria",
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "figure.constrained_layout.use": True,
        }
    )


# region Folded print info on model
def print_state_dict(state_dict: dict) -> None:
    """Pretty prints a `torch.nn.Module` state dict, which is just a regular dictionnary.

    Args:
        state_dict (dict): State dict of a torch.nn.Module
    """
    print("state dict:")
    for key, values in state_dict.items():
        print(f"{key}: {values}")


def print_other_pars(other_pars: dict) -> None:
    """Pretty prints the `other_pars` instance attribute of the :py:class:`gausstorch.libs.qsyst.Qsyst` class

    Args:
        other_pars (dict): `other_pars` attribute of the :py:class:`gausstorch.libs.qsyst.Qsyst` class
    """
    print("other pars:")
    for key, values in other_pars.items():
        print(f"{key}: {values}")


def print_model_parameters(named_parameters: dict) -> None:
    """Pretty prints the parameters of an torch.nn.Module.named_parameters()

    Args:
        named_parameters (dict): `named_parameters()` evalutation of an torch.nn.Module
    """
    print(f"model parameters:")
    for name, param in named_parameters:
        if param.requires_grad:
            print(f"{name} = {param.data}")


# endregion Folded print info on model


# region Plotting Wigner


def plot_wigner(
    ax: plt.axes,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    quad1_key: str,
    quad2_key: str,
) -> matplotlib.collections.QuadMesh:
    """Plots the 2D cross section of the wigner quasi-probabiliy distribution of a (`alpha`, `sigma`) state, for quadratures `quad1_key` and `quad2_key`.

    Args:
        ax (plt.axes): Ax to plot the QuadMesh into
        alpha (torch.Tensor): Field operator displacement vector
        sigma (torch.Tensor): Field operator covariance matrix
        quad1_key (str): First quadrature chosen for the cross-section
        quad2_key (str): Second quadrature chosen for the cross-section

    Returns:
        matplotlib.collections.QuadMesh:
    """
    with torch.no_grad():
        M = alpha.shape[0] // 2
        alpha_r, sigma_r = moments_to_quad_moments(alpha, sigma)
        alpha_r_x1_x2 = keep_quads_in_alpha_r(alpha_r, quad1_key, quad2_key)
        sigma_r_x1_x2 = keep_quads_in_sigma_r(sigma_r, quad1_key, quad2_key)
        xvec = torch.linspace(-3, 3, 100)
        yvec = torch.linspace(-3, 3, 100)
        W = wigner_2d_map(alpha_r_x1_x2, sigma_r_x1_x2, xvec, yvec)
        assert W.dtype == torch.float64
        # plot
        c = ax.pcolormesh(
            xvec.detach().numpy(), yvec.detach().numpy(), W.detach().numpy()
        )
        ax.set_xlabel(quad1_key, fontsize=FS)
        ax.set_ylabel(quad2_key, fontsize=FS)
        ax.set_aspect("equal")
    return c


# endregion Plotting Wigner


# region Plotting dynamics


def plot_evolution_N(
    tspan: np.ndarray,
    means: np.ndarray,
    width_ratio=0.48,
    xlabel="time (ns)",
    ylabel=r"$\boldsymbol{N}$",
    yscale="linear",
):
    setup_tex()
    M = means.shape[1]
    width = width_ratio * TEXTWIDTH_INCH
    figsize = (width, (5.5 / 9) * width)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    ax.plot(
        tspan,
        means,
        label=[rf"$\boldsymbol{{N_{{{i + 1}}}}}$" for i in range(M)],
        linewidth=1,
    )
    # ax.plot(tspan * 100, means, label=[f'$N_{chr(65 + i)}$' for i in range(self.M)])  # when R = 1e7
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_yscale(yscale)
    if np.max(means) > 10**4:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax.tick_params(axis="both", labelsize=FS_S)
    ax.grid(True)
    # fig.suptitle(rf"<$ N_i $> for constant drive, $N_{{\text{{osc}}}}$={M}", fontsize=FS)
    ax.legend(fontsize=FS_S)
    return fig, ax


def plot_evolution_fock(
    tspan: np.ndarray,
    probs: np.ndarray,
    labels: list,
    width_ratio=0.48,
    xlabel="time (ns)",
    ylabel=r"Probability",
    yscale="linear",
):
    setup_tex()
    M = probs.shape[1]
    width = width_ratio * TEXTWIDTH_INCH
    figsize = (width, (5.5 / 9) * width)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    ax.plot(tspan, probs, label=labels, linewidth=1)
    # ax.plot(tspan * 100, means, label=[f'$N_{chr(65 + i)}$' for i in range(self.M)])  # when R = 1e7
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_yscale(yscale)
    if np.max(probs) > 10**4:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax.tick_params(axis="both", labelsize=FS_S)
    ax.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=FS)
    return fig, ax


# endregion Plotting dynamics


# region Fock combination manipulation


def fock_state_to_str_general(fock_combination, mode_comb):
    prob_notation = [
        f"{str(fock_num)}_{str(osc)}"
        for fock_num, osc in zip(fock_combination, mode_comb)
    ]
    return "$P(" + "".join(prob_notation) + ")$"


def fock_states_to_str_list(fock_combs_per_mode_comb):
    fock_str_list = []
    for mode_comb, fock_combinations in fock_combs_per_mode_comb.items():
        for fock_combination in fock_combinations:
            fock_str_list.append(fock_state_to_str_general(fock_combination, mode_comb))
    return fock_str_list


# endregion Fock combination manipulation
