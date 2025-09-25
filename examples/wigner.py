import matplotlib.pyplot as plt
import torch
import numpy as np

from gausstorch.libs.qsyst import Qsyst
from gausstorch.utils.display import plot_wigner, TEXTWIDTH_INCH, setup_tex

torch.set_default_dtype(torch.float64)

M = 4
g_shape = M * (M - 1) // 2  # number of possible couplings for a given coupling type (photon conversion or two-mode squeezing)
eA = M * 1e5
init_pars = {
    'M': M,  # Number of modes
    # learnable parameters
    # Output weight matrix, for the linear classifier on a reservoir output. Only use in child classes for learning, so unused here.
    'W_0': torch.ones(1),
    # Output weight bias, for the linear classifier on a reservoir output. Only use in child classes for learning, so unused here.
    'W_bias': torch.zeros(1),
    # Real value of the encoding parameter bias, called 'theta'. Set to 0 by default.
    'theta_bias_real': torch.zeros(1),
    # Imag value of the encoding parameter bias, called 'theta'. Set to 0 by default.
    'theta_bias_imag': torch.zeros(1),
    # (Real) value of the phase encoding parameter, if phase encoding is chosen.
    'phi_0': torch.zeros(1),
    # (Real) bias value of the phase encoding parameter, if phase encoding is chosen.
    'phi_bias': torch.zeros(1),
    # detuning of the drives with respect to the mode resonance frequency.
    'detuning': torch.zeros(M),
    # Real and imaginrary values of the drives
    'eA_real': eA.real * torch.ones(M),
    'eA_imag': eA.imag * torch.ones(M),
    # Real and imaginary values of the photon conversion rates
    'g_real': 2 * torch.pi * 100e6 * torch.ones(g_shape),
    'g_imag': torch.zeros(g_shape),
    # Real and imaginary values of the two-mode squeezing rates
    'gs_real': 2 * torch.pi * 20e6 * torch.ones(g_shape),
    'gs_imag': torch.zeros(g_shape),
    # Internal (caused by coupling with environment) dissipation rate
    'k_int': 0 * torch.ones(M),
    # External (caused by coupling with drive transmission lines) dissipation rate
    'k_ext': 2 * torch.pi * 2e6 * torch.ones(M),
    # other parameters
    # default time interval for dynamical simulation, in seconds
    't_i': torch.tensor(1e-7),
}
model = Qsyst(init_pars=init_pars)

theta_encoded = 0.4 * (model.syst_vars.eA_real + 1j * model.syst_vars.eA_imag).data
alpha, sigma = model.alpha_sigma_evolution(
    model.other_pars['t_i'], model.alpha0, model.sigma0,
    'eA', theta_encoded
)

# Plot
width = 0.48 * TEXTWIDTH_INCH
height = width
setup_tex()
fig, ax = plt.subplots(1, 1, figsize=(width, height))
c = plot_wigner(ax, alpha, sigma, 'X0', 'P0')
plt.colorbar(c)
plt.show()
