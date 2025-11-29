"""
Test file for qsyst.py
"""

import pytest
import torch
from gausstorch.libs.qsyst import Qsyst, init_pars_default
from gausstorch.utils.param_processing import rescale_pars

torch.set_default_dtype(torch.float64)


# @pytest.fixture
# def qsyst():
#     """
#     Creates a new instance of Qsyst before each test. For tests, use a 2-mode system
#     """
#     init_pars = init_pars_default(2)
#     return Qsyst(init_pars)

def test_alpha_after_time_evolution_g_only(M=2):
    # input parameters
    theta_key, encode_phase = 'eA', False
    x_val, x_min, x_max = 0.4, 0, 1

    # model parameters
    g_shape = M * (M - 1) // 2
    t = torch.tensor(1e-7)
    eA = torch.tensor(M * 1e5, dtype=torch.complex128)
    g = torch.tensor(2 * torch.pi * 100e6, dtype=torch.complex128)
    k_ext = torch.tensor(2 * torch.pi * 2e6)
    pars = {
        'M': M,
        # learnable parameters
        'W_0': torch.ones(1), 'W_bias': torch.zeros(1),
        'theta_bias_real': torch.zeros(1), 'theta_bias_imag': torch.zeros(1),
        'phi_0': torch.zeros(1), 'phi_bias': torch.zeros(1),
        'detuning': torch.zeros(M),
        'eA_real': eA.real * torch.ones(M),
        'eA_imag': eA.imag * torch.ones(M),
        'g_real': g.real * torch.ones(g_shape),
        'g_imag': g.imag * torch.ones(g_shape),
        'gs_real': torch.zeros(g_shape),
        'gs_imag': torch.zeros(g_shape),
        'k_int': torch.zeros(M),
        'k_ext': k_ext * torch.ones(M),
        # other parameters
        't_i': t,
    }
    pars = rescale_pars(pars, torch.tensor(2e6))
    model = Qsyst(pars)
    theta_0, x_mask_shape = model.return_theta_xmask(theta_key)
    x_mask = x_val * torch.ones(x_mask_shape, dtype=torch.complex128)
    theta_encoded = model.encode_theta(theta_0=theta_0, x_mask=x_mask, x_min=x_min, x_max=x_max,
                                       encode_phase=encode_phase)
    alpha_in_scalar = theta_encoded[0]
        # torch.concatenate((theta_encoded, theta_encoded.conj()), dim=0).unsqueeze(dim=1)
    alpha_1_theory = torch.sqrt(pars['k_ext'][0]) * (-alpha_in_scalar) * (
            (torch.exp((-1j * pars['g_real'][0] - (pars['k_ext'][0] / 2)) * pars['t_i']) - 1) /
            (-1j * pars['g_real'][0] - (pars['k_ext'][0] / 2))
    )
    alpha_theory = torch.tensor(
        [alpha_1_theory, alpha_1_theory, alpha_1_theory.conj(), alpha_1_theory.conj()]
    ).unsqueeze(dim=1)
    alpha_computed, sigma_computed = model.alpha_sigma_evolution(
        t=model.other_pars['t_i'],
        alpha_i=model.alpha0,
        sigma_i=model.sigma0,
        theta_key=theta_key,
        theta_encoded=theta_encoded
    )
    # print(f"alpha from theory: {alpha_theory}")
    # print(f"alpha from computation: {alpha_computed}")
    # print(alpha_theory.isclose(alpha_computed, rtol=1e-3, atol=1e-5))
    assert alpha_theory.isclose(alpha_computed).all()
