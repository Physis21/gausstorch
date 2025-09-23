import torch
import torch.nn as nn
from copy import deepcopy

from gausstorch.constants import SYST_VARS_KEYS_WITH_BIASES


def separate_syst_other(init_pars: dict):
    # separate contents of dict between parameters needing grad and not needing grad
    params_syst = nn.ParameterDict({
        key: nn.Parameter(init_pars[key], requires_grad=True)
        for key in SYST_VARS_KEYS_WITH_BIASES
    })
    params_others = {
        key: init_pars[key]
        for key in ['t_i']
    }
    return params_syst, params_others


def rescale_law(unscaled_val: torch.float64, par_key: str, R: torch.float64):
    """

    :param unscaled_val: unscaled value of parameter to set
    :param par_key: name of the parameter whose value to substitute
    :param R: rescaling factor
    :return: rescaled value
    """
    scaled_val = unscaled_val
    if par_key in ['detuning', 'g_real', 'g_imag', 'gs_real', 'gs_imag', 'k_int', 'k_ext']:
        scaled_val = unscaled_val / R
    elif par_key in ['eA_real', 'eA_imag']:
        scaled_val = unscaled_val / torch.sqrt(R)
    elif par_key == 't_i':
        scaled_val = unscaled_val * R
    return scaled_val


def unscale_law(unscaled_val: torch.float64, par_key: str, R: torch.float64):
    """

    :param unscaled_val: unscaled value of parameter to set
    :param par_key: name of the parameter whose value to substitute
    :param R: rescaling factor
    :return: rescaled value
    """
    if par_key in ['detuning', 'g_real', 'g_imag', 'gs_real', 'gs_imag', 'k_int', 'k_ext']:
        scaled_val = unscaled_val * R
    elif par_key in ['eA_real', 'eA_imag']:
        scaled_val = unscaled_val * torch.sqrt(R)
    else:
        raise NameError
    return scaled_val


def rescale_pars(pars: dict, R: torch.float64):
    rescaled_pars = {
        key: rescale_law(val, key, R) for key, val in pars.items()
    }
    return deepcopy(rescaled_pars)  # avoid memory shenanigans
