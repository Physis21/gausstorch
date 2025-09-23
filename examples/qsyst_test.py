import torch

from gausstorch.libs.qsyst import Qsyst, init_par_default
from gausstorch.utils.param_processing import rescale_pars

init_pars = init_par_default(4)
init_pars = rescale_pars(init_pars, torch.tensor(2e6))

model = Qsyst(init_pars=init_pars, init_print=True)
model.evolution_N()
