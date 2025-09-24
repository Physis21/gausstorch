import torch

from gausstorch.libs.qsyst import Qsyst, init_pars_default
from gausstorch.utils.param_processing import rescale_pars

init_pars = init_pars_default(4)
init_pars = rescale_pars(init_pars, torch.tensor(2e6))

model = Qsyst(init_pars=init_pars, init_print=True)
model.evolution_N()
