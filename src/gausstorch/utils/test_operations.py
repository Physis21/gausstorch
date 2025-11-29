import torch
from gausstorch.utils.operations import torch_block, wigner, wigner_2d_map

torch.set_default_dtype(torch.float64)


def test_torch_block(n=3):
    A, B, C, D = torch.zeros((n, n)), torch.zeros((n, n)), torch.zeros((n, n)), torch.zeros((n, n))
    assert torch_block(A, B, C, D).equal(torch.zeros(2 * n, 2 * n))


def test_wigner(d=torch.zeros((2, 1)), alpha=torch.zeros((2, 1)), sigma=0.5 * torch.eye(2)):
    # Check value at zero of vacuum state is equal to 1/pi
    assert wigner(d, alpha, sigma).isclose(torch.tensor(1 / torch.pi))
