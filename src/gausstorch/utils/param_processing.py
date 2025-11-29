import torch
import torch.nn as nn
from copy import deepcopy

from gausstorch.constants import SYST_VARS_KEYS_WITH_BIASES


def rescale_law(
    unscaled_val: torch.Tensor, par_key: str, R: torch.Tensor
) -> torch.Tensor:
    """Returns rescaled value of a physical parameter.

    Args:
        unscaled_val (torch.Tensor): unscaled value of parameter to set
        par_key (str): name of the parameter whose value to substitute
        R (torch.Tensor): rescaling factor

    Returns:
        torch.Tensor: Rescaled parameter
    """
    scaled_val = unscaled_val
    if par_key in [
        "detuning",
        "g_real",
        "g_imag",
        "gs_real",
        "gs_imag",
        "k_int",
        "k_ext",
    ]:
        scaled_val = unscaled_val / R
    elif par_key in ["eA_real", "eA_imag"]:
        scaled_val = unscaled_val / torch.sqrt(R)
    elif par_key == "t_i":
        scaled_val = unscaled_val * R
    return scaled_val


def unscale_law(
    rescaled_val: torch.Tensor, par_key: str, R: torch.Tensor
) -> torch.Tensor:
    """Performs the inverse operation to :py:func:`rescale_law`

    Args:
        rescaled_val (torch.Tensor): rescaled value of parameter to unscale

        par_key (str): name of the parameter whose value to substitute
        
        R (torch.Tensor): rescaling factor

    Raises:
        NameError: If the `par_key` parameter key is not valid

    Returns:
        torch.Tensor: Unscaled value
    """
    if par_key in [
        "detuning",
        "g_real",
        "g_imag",
        "gs_real",
        "gs_imag",
        "k_int",
        "k_ext",
    ]:
        scaled_val = rescaled_val * R
    elif par_key in ["eA_real", "eA_imag"]:
        scaled_val = rescaled_val * torch.sqrt(R)
    else:
        raise NameError
    return scaled_val


def rescale_pars(pars: dict, R: torch.tensor) -> dict:
    """Returns a dict with all rescaled parameters from the `pars` argument

    Args:
        pars (dict): Contains parameter key-value pairs
        R (torch.tensor): Scaling parameter

    Returns:
        dict: Dict with rescaled parameters
    """
    rescaled_pars = {key: rescale_law(val, key, R) for key, val in pars.items()}
    return deepcopy(rescaled_pars)  # avoid memory shenanigans
