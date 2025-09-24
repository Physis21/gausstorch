import torch

from gausstorch.libs.qsyst import Qsyst, init_pars_default
from gausstorch.utils.param_processing import rescale_pars

init_pars = init_pars_default(2)
init_pars = rescale_pars(init_pars, torch.tensor(2e6))
fock_combs_per_mode_comb = {
    (0, 1): [(0, 0), (1, 0)],
}

model = Qsyst(init_pars=init_pars, init_print=True)
model.evolution_fock(fock_combs_per_mode_comb)